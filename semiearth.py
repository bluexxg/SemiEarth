import argparse
from copy import deepcopy
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from evaluation import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.cee import extract_multi_class_edge, save_cee_debug
from util.ctsa import prepare_ctsa, cutmix_tensor, auxiliary_loss, ctsa_weight
from util.training_runtime import (autocast_settings, loader_options, update_ema,
                                   load_segmentation_state, StepTimer)

parser = argparse.ArgumentParser(
    description='Vision-Language Model Purified Semi-Supervised Semantic Segmentation for Remote Sensing Images')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--init-checkpoint', default=None, help='Initialize weights only; use a NEW save directory')

from util.benchmark_runtime import add_arguments as add_benchmark_arguments
add_benchmark_arguments(parser)

def get_vlm_purify(cfg, model, model_ema):
    vlm_type = cfg.get('vlm_type')
    class_names = CLASSES[cfg['dataset']]

    if vlm_type == 'qwen_vl':
        from model.semseg.vlm_runtime import BatchedQwenVLPurifiedSemi
        return BatchedQwenVLPurifiedSemi(cfg, model, model_ema, class_names)
    elif vlm_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown VLM type: {vlm_type}. "
                         f"Supported: 'qwen_vl', 'none'")

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    from util.benchmark_runtime import prepare as prepare_benchmark, Benchmark
    prepare_benchmark(args, cfg)

    # ---- CEE Edge-aware Module config (all have safe defaults) -----------
    use_cee = cfg.get('use_cee', False)
    use_edge_threshold = cfg.get('use_edge_threshold', True)
    use_cee_vlm = cfg.get('use_cee_vlm', True)
    edge_width = cfg.get('edge_width', 3)
    edge_conf_thresh = cfg.get('edge_conf_thresh', 0.85)
    vlm_pp_conf_threshold = cfg.get('vlm_pp_conf_threshold', 0.7)
    cee_debug = cfg.get('cee_debug', False)
    save_cee_debug_imgs = cfg.get('save_cee_debug', False)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if args.benchmark and world_size != 1:
        raise ValueError('Short benchmark supports exactly one GPU/process.')

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    use_ctsa = bool(cfg.get('use_ctsa', False))
    amp_enabled, amp_dtype = autocast_settings(cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    if rank == 0:
        logger.info('CTSA=%s, student AMP=%s/%s; CEE selection/fusion unchanged',
                    use_ctsa, amp_enabled, amp_dtype)

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]],
                   'nclass': cfg['nclass'], 'ctsa_config': cfg})

    if not args.benchmark:  # Benchmark restores the complete student and EMA checkpoint below.
        state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location='cpu', weights_only=True)
        model.backbone.load_state_dict(state_dict)

    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    model_ema = deepcopy(model.module)
    model_ema.ctsa = None  # teacher is the deployment network, without auxiliary parameters
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if rank == 0 and cfg.get('profile_model', False):
        from profile_model import measure_flops, measure_fps
        _H, _W = cfg['crop_size'] if isinstance(cfg['crop_size'], (list, tuple)) \
            else (cfg['crop_size'], cfg['crop_size'])
        _dummy = torch.randn(1, 3, _H, _W).cuda()
        model.eval()
        logger.info('\n[Profile] FLOPs & Params')
        measure_flops(model.module, _dummy, logger)  # .module 去掉 DDP 包装
        logger.info('[Profile] FPS')
        measure_fps(model.module, _dummy, warmup=30, runs=100,
                    device=torch.device('cuda'), logger=logger)
        model.train()


    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path,
        single_strong=True
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], drop_last=True, sampler=trainsampler_l, **loader_options(cfg)
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], drop_last=True, sampler=trainsampler_u, **loader_options(cfg)
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, drop_last=False, sampler=valsampler, **loader_options(cfg, validation=True)
    )

    if not len(trainloader_u) or len(trainloader_l) != len(trainloader_u):
        raise ValueError('Need nonempty, equally long labeled/unlabeled loaders; check split and batch size')
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    latest_path = (args.benchmark_checkpoint if args.benchmark else
                   os.path.join(args.save_path, 'latest.pth'))
    if args.init_checkpoint and os.path.exists(latest_path):
        raise ValueError('--init-checkpoint requires a new save directory')
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        load_segmentation_state(model, initial.get('model', initial), allow_missing_ctsa=True)
        load_segmentation_state(model_ema, initial.get('model_ema', initial.get('model', initial)))
        if rank == 0:
            logger.info('Initialized weights from %s; optimizer and schedule start fresh', args.init_checkpoint)

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        has_ctsa = any(k.removeprefix('module.').startswith('ctsa.') for k in checkpoint['model'])
        if has_ctsa != use_ctsa:
            raise ValueError('CTSA configuration differs from checkpoint. Use --init-checkpoint and a NEW save path.')
        load_segmentation_state(model, checkpoint['model'])
        load_segmentation_state(model_ema, checkpoint['model_ema'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    if args.benchmark:
        if not use_ctsa or epoch + 1 >= cfg['epochs']:
            raise ValueError('Need an in-progress CTSA checkpoint with at least one epoch remaining.')
        if ctsa_weight((epoch + 1) * len(trainloader_u), total_iters, cfg) <= 0:
            raise ValueError('Checkpoint is still in CTSA warmup. Supply a later checkpoint; do not shorten epochs.')
        if args.benchmark_steps + args.benchmark_warmup > len(trainloader_u):
            raise ValueError('Requested benchmark is longer than one epoch.')

    vlm_purify = None
    if cfg.get('use_vlm_pp', True):
        vlm_purify = get_vlm_purify(cfg, model, model_ema)

    benchmark = (Benchmark(args, cfg, vlm_purify, len(trainloader_u), epoch + 1)
                 if args.benchmark else None)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema,
                                                        best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_ctsa = AverageMeter()
        total_gate_ratio = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_throughput = AverageMeter()
        cee_edge_ratio = AverageMeter()
        cee_reliable_edge_ratio = AverageMeter()
        cee_low_conf_edge_ratio = AverageMeter()
        cee_vlm_target_ratio = AverageMeter()
        cee_vlm_target_pixels = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()
        epoch_start = time.perf_counter()
        log_every = max(1, int(cfg.get('log_interval', max(1, len(trainloader_u) // 8))))

        if benchmark:
            torch.cuda.synchronize()
            benchmark.previous_end = time.perf_counter()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, ignore_mask, cutmix_box)) in enumerate(loader):
            iters = epoch * len(trainloader_u) + i
            if benchmark:
                benchmark.start_step(i)
            timer = benchmark or StepTimer(i < int(cfg.get('profile_steps', 0)) and epoch == 0)
            ct_weight = ctsa_weight(iters, total_iters, cfg) if use_ctsa else 0.0
            # Same decision on every rank. During warmup skip the entire auxiliary
            # path (DDP find_unused_parameters=True); no rank-specific skipping.
            run_ctsa = use_ctsa and ct_weight > 0

            img_x, mask_x = img_x.cuda(non_blocking=True), mask_x.cuda(non_blocking=True)
            img_u_w, img_u_s = img_u_w.cuda(non_blocking=True), img_u_s.cuda(non_blocking=True)
            ignore_mask, cutmix_box = ignore_mask.cuda(non_blocking=True), cutmix_box.cuda(non_blocking=True)

            with torch.no_grad():
                with torch.autocast('cuda', enabled=amp_enabled and cfg.get('teacher_amp', False), dtype=amp_dtype):
                    teacher_output = model_ema(img_u_w, return_last_feature=run_ctsa)
                if run_ctsa:
                    pred_u_w, teacher_last = teacher_output
                else:
                    pred_u_w = teacher_output
                conf_u_w, mask_u_w = pred_u_w.float().softmax(dim=1).max(dim=1)
                raw_teacher_labels = mask_u_w.clone() if run_ctsa else None
                timer.mark('teacher_ms')
                # CEE: multi-class semantic edge map from teacher pseudo labels.
                # Pure tensor op, zero params, inside no_grad -> negligible cost.
                edge_map = extract_multi_class_edge(mask_u_w, edge_width=edge_width) if use_cee else None

                if benchmark:
                    timer.mark('cee_edge_ms')

                # ---- CEE statistics: computed on the ORIGINAL teacher conf
                # (before VLM overwrites conf_u_w) so the logged masks are
                # IDENTICAL to the priority_mask used inside get_qwen_purify.
                if use_cee and edge_map is not None:
                    valid = (ignore_mask != 255)
                    total_valid = valid.sum()
                    _edge = edge_map & valid
                    cee_edge_ratio.update(_edge.sum() / total_valid.clamp_min(1))
                    _reliable = _edge & (conf_u_w >= edge_conf_thresh)
                    cee_reliable_edge_ratio.update(_reliable.sum() / total_valid.clamp_min(1))
                    _low_conf_edge = _edge & (conf_u_w < edge_conf_thresh)
                    cee_low_conf_edge_ratio.update(_low_conf_edge.sum() / total_valid.clamp_min(1))
                    if cee_debug and (_reliable & _low_conf_edge).sum() != 0:
                        logger.warning('[CEE ERROR] reliable & low-conf edge overlap: threshold direction bug')
                    # VLM target == low-conf & edge, the SAME tensor the VLM
                    # uses as its priority_mask (conf < vlm_pp_conf_threshold).
                    _vlm_target = edge_map & (conf_u_w < vlm_pp_conf_threshold)
                    vlm_target_pixels = _vlm_target.sum()
                    cee_vlm_target_pixels.update(vlm_target_pixels)
                    cee_vlm_target_ratio.update(vlm_target_pixels / total_valid.clamp_min(1))
                else:
                    vlm_target_pixels = 0

                if benchmark:
                    timer.mark('cee_statistics_ms')
                    benchmark.capture_inputs(img_u_w, mask_u_w, conf_u_w, edge_map, ignore_mask)

                if vlm_purify is not None:
                    vlm_type = cfg.get('vlm_type')
                    if vlm_type in ['qwen_vl']:
                        # edge_map guides the VLM target selection & fusion when
                        # use_cee & use_cee_vlm are on; otherwise None keeps the
                        # original VLM-PP behaviour.
                        if cfg.get('use_vlm_on_mismatch', True):
                            conf_u_w, mask_u_w = vlm_purify.get_qwen_purify(
                                img_u_w, mask_u_w, conf_u_w, edge_map=edge_map)
                        else:
                            conf_u_w = vlm_purify.get_qwen_purify(
                                img_u_w, mask_u_w, conf_u_w, edge_map=edge_map)
                if benchmark:
                    benchmark.capture_outputs(conf_u_w, mask_u_w)
            timer.mark('vlm_ms')
            img_u_s = cutmix_tensor(img_u_s, cutmix_box)

            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            edge_map_cutmixed = edge_map.clone() if edge_map is not None else None

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]
            if edge_map_cutmixed is not None:
                edge_map_cutmixed[cutmix_box == 1] = edge_map.flip(0)[cutmix_box == 1]

            # Edge-aware pseudo label selection: interior pixels keep the
            # original conf_thresh, boundary pixels use the relaxed
            # edge_conf_thresh. Disabled -> exactly the original filter.
            if use_cee and use_edge_threshold and edge_map_cutmixed is not None:
                conf_mask = (~edge_map_cutmixed) & (conf_u_w_cutmixed >= cfg['conf_thresh'])
                conf_mask = conf_mask | (edge_map_cutmixed & (conf_u_w_cutmixed >= edge_conf_thresh))
            else:
                conf_mask = conf_u_w_cutmixed >= cfg['conf_thresh']
            accepted = conf_mask & (ignore_mask_cutmixed != 255)
            aux_logits = None
            if run_ctsa:
                raw_mixed = cutmix_tensor(raw_teacher_labels, cutmix_box)
                teacher_last, gate, active = prepare_ctsa(
                    teacher_last, raw_mixed, mask_u_w_cutmixed, conf_u_w_cutmixed,
                    accepted, cutmix_box, (img_u_s.shape[-2] // 14, img_u_s.shape[-1] // 14),
                    mode=cfg.get('ctsa_gate', 'agreement'),
                    reject_corrected=cfg.get('ctsa_reject_corrected', True))
            with torch.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                if run_ctsa:
                    predictions, aux_logits = model(
                        torch.cat((img_x, img_u_s)), teacher_feature=teacher_last,
                        ctsa_gate=gate, num_labeled=num_lb)
                else:
                    predictions = model(torch.cat((img_x, img_u_s)))
                pred_x, pred_u_s = predictions.split([num_lb, num_ulb])
            loss_x = criterion_l(pred_x.float(), mask_x)
            loss_u_s = criterion_u(pred_u_s.float(), mask_u_w_cutmixed)
            loss_u_s = (loss_u_s * accepted).sum() / (ignore_mask_cutmixed != 255).sum().clamp_min(1)
            loss_ctsa = (auxiliary_loss(aux_logits, mask_u_w_cutmixed, accepted,
                                       ignore_mask_cutmixed, active) if run_ctsa else loss_u_s.new_zeros(()))
            loss = (loss_x + loss_u_s) / 2.0 + ct_weight * loss_ctsa
            timer.mark('student_forward_ms')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= old_scale
            timer.mark('backward_optimizer_ms')

            total_loss.update(loss.detach())
            total_loss_x.update(loss_x.detach())
            total_loss_s.update(loss_u_s.detach())
            total_loss_ctsa.update(loss_ctsa.detach())
            if run_ctsa:
                total_gate_ratio.update((gate > 0).float().mean())
            # mask ratio (edge-aware when CEE is on) + CEE stats for logging
            if use_cee and use_edge_threshold and edge_map is not None:
                conf_mask_ratio = ((~edge_map) & (conf_u_w >= cfg['conf_thresh'])) | \
                                  (edge_map & (conf_u_w >= edge_conf_thresh))
                mask_ratio = (conf_mask_ratio & (ignore_mask != 255)).sum() / (ignore_mask != 255).sum().clamp_min(1)
            else:
                mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum() / (
                    ignore_mask != 255).sum().clamp_min(1)
            total_mask_ratio.update(mask_ratio.detach())

            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            if stepped:
                update_ema(model, model_ema, ema_ratio)
            timer.mark('ema_ms')
            timings = timer.finish()
            if timings and rank == 0:
                logger.info('[Profile step %d] %s', iters, timings)

            if rank == 0 and (i % log_every == 0 or i + 1 == len(trainloader_u)):
                writer.add_scalar('train/loss_ctsa', loss_ctsa.detach().item(), iters)
                writer.add_scalar('train/ctsa_weight', ct_weight, iters)
                writer.add_scalar('train/ctsa_gate_ratio', total_gate_ratio.avg, iters)
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                if use_cee:
                    writer.add_scalar('train/cee_edge_ratio', cee_edge_ratio.val, iters)
                    writer.add_scalar('train/cee_reliable_edge_ratio', cee_reliable_edge_ratio.val, iters)
                    writer.add_scalar('train/cee_low_conf_edge_ratio', cee_low_conf_edge_ratio.val, iters)
                    writer.add_scalar('train/cee_vlm_target_ratio', cee_vlm_target_ratio.val, iters)

            if (i % log_every == 0 or i + 1 == len(trainloader_u)) and rank == 0:
                elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                total_throughput.update((i + 1) * cfg['batch_size'] * world_size * 2 / elapsed)
                logger.info('[CTSA] weight=%.4f loss=%.4f gate_coverage=%.4f',
                            ct_weight, total_loss_ctsa.avg, total_gate_ratio.avg)
                logger.info(
                    'Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, '
                    'Loss s: {:.3f}, Mask ratio: {:.3f}, Throughput: {:.1f} img/s'.format(
                        i, optimizer.param_groups[0]['lr'],
                        total_loss.avg, total_loss_x.avg,
                        total_loss_s.avg, total_mask_ratio.avg,
                        total_throughput.avg
                    )
                )
                if use_cee:
                    logger.info(
                        '[CEE] Edge ratio: {:.2f}%, Reliable edge ratio: {:.2f}%, '
                        'Low-conf edge ratio: {:.2f}%'.format(
                            cee_edge_ratio.avg * 100,
                            cee_reliable_edge_ratio.avg * 100,
                            cee_low_conf_edge_ratio.avg * 100)
                    )
                    logger.info(
                        '[CEE-VLM] Target pixels: {:}, VLM refinement triggered: {}'.format(
                            int(cee_vlm_target_pixels.val),
                            'Yes' if cee_vlm_target_pixels.val > 0 else 'No')
                    )

            # Debug visualization (default off). Save the 7-panel CEE debug
            # figure once per epoch on the first iteration.
            if (rank == 0) and save_cee_debug_imgs and use_cee and edge_map is not None and i == 0:
                _low_conf_mask = conf_u_w < vlm_pp_conf_threshold
                _vlm_target_mask = _low_conf_mask & edge_map
                save_cee_debug(
                    os.path.join(args.save_path, 'debug_cee'),
                    'ep{}_it{}'.format(epoch, i),
                    img_u_w, mask_u_w, edge_map, conf_u_w,
                    _low_conf_mask, _vlm_target_mask,
                    refined_labels=mask_u_w, nclass=cfg['nclass'],
                )

            if benchmark and benchmark.end_step(ct_weight):
                writer.close()
                torch.distributed.destroy_process_group()
                return  # Never validate or save weights in benchmark mode.

        if rank == 0:
            logger.info('[Timing] training epoch seconds: %.1f', time.perf_counter() - epoch_start)
        validation_start = time.perf_counter()
        eval_mode = 'sliding_window' if cfg['dataset'] == '-' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        if rank == 0:
            logger.info('[Timing] student+EMA validation seconds: %.1f', time.perf_counter() - validation_start)
            logger.info(
                '[Epoch {:}] Avg Training Throughput (FPS): {:.1f} img/s'.format(
                    epoch, total_throughput.avg
                )
            )
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou,
                                                 iou_class_ema[cls_idx]))
            logger.info(
                '***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = mIoU >= previous_best

        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
