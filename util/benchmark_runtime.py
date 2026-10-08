"""Opt-in single-GPU short benchmark; never saves model weights.

Stage times are synchronized wall time. Nested method times are inclusive:
do not add them to the stage totals. Only non-detailed runs measure throughput.
"""
import functools
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def add_arguments(parser):
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-checkpoint')
    parser.add_argument('--benchmark-steps', type=int, default=20)
    parser.add_argument('--benchmark-warmup', type=int, default=5)
    parser.add_argument('--benchmark-qwen-batch', type=int, default=2)
    parser.add_argument('--benchmark-detail', action='store_true')
    parser.add_argument('--benchmark-compare-batch', type=int, default=0)
    parser.add_argument('--benchmark-seed', type=int, default=20260929)


def prepare(args, cfg):
    if not args.benchmark:
        return
    if args.init_checkpoint:
        raise ValueError('Use --benchmark-checkpoint, not --init-checkpoint: preserve schedule/optimizer.')
    if not args.benchmark_checkpoint or not Path(args.benchmark_checkpoint).is_file():
        raise ValueError('Benchmark requires an existing full training checkpoint.')
    if Path(args.save_path).exists():
        raise ValueError('Benchmark save-path must be NEW; existing experiment directories are protected.')
    if args.benchmark_steps < 1 or args.benchmark_warmup < 0 or args.benchmark_qwen_batch < 1:
        raise ValueError('Invalid benchmark step count or batch size.')
    if args.benchmark_compare_batch and (not args.benchmark_detail or args.benchmark_compare_batch < 1):
        raise ValueError('Output comparison is allowed only in a detailed, non-throughput run.')
    if not cfg.get('use_ctsa') or not cfg.get('use_vlm_pp', True) or cfg.get('vlm_type') != 'qwen_vl':
        raise ValueError('This benchmark requires CTSA and Qwen purification enabled.')
    if cfg.get('use_vlm_cache', not (cfg.get('use_cee') and cfg.get('use_cee_vlm', True))):
        raise ValueError('Cross-iteration VLM cache must be disabled for this paired benchmark.')
    cfg['qwen_batch_size'] = args.benchmark_qwen_batch
    cfg['profile_steps'] = 0
    cfg['save_cee_debug'] = False
    cfg['cee_debug'] = False
    random.seed(args.benchmark_seed)
    np.random.seed(args.benchmark_seed)
    torch.manual_seed(args.benchmark_seed)
    torch.cuda.manual_seed_all(args.benchmark_seed)


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('No measured steps')
    return {'mean_s': statistics.mean(ordered), 'median_s': statistics.median(ordered),
            'min_s': ordered[0], 'max_s': ordered[-1],
            'p90_s': ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            'samples': len(ordered)}


def accepted_mask(confidence, edge, ignore, cfg):
    if cfg.get('use_cee') and cfg.get('use_edge_threshold', True) and edge is not None:
        selection = ((~edge) & (confidence >= cfg['conf_thresh'])) | (
            edge & (confidence >= cfg.get('edge_conf_thresh', 0.85)))
    else:
        selection = confidence >= cfg['conf_thresh']
    return selection & (ignore != 255)


def compare_outputs(base, trial, edge, ignore, cfg):
    bc, bl = base
    tc, tl = trial
    valid = ignore != 255
    delta = (bc.float() - tc.float()).abs()[valid]
    label_changes = int(((bl != tl) & valid).sum().item())
    accepted_changes = int((accepted_mask(bc, edge, ignore, cfg) !=
                            accepted_mask(tc, edge, ignore, cfg)).sum().item())
    finite = bool(torch.isfinite(bc).all() and torch.isfinite(tc).all())
    maximum = float(delta.max().item()) if delta.numel() else 0.0
    return {'valid_pixels': int(valid.sum().item()), 'label_changed_pixels': label_changes,
            'accepted_changed_pixels': accepted_changes, 'confidence_max_abs': maximum,
            'confidence_mean_abs': float(delta.mean().item()) if delta.numel() else 0.0,
            'finite': finite, 'within_tolerance': finite and label_changes == 0 and
            accepted_changes == 0 and maximum <= 1e-6}


class Benchmark:
    def __init__(self, args, cfg, purifier, batches_per_epoch, start_epoch):
        self.args, self.cfg, self.purifier = args, cfg, purifier
        self.batches_per_epoch = batches_per_epoch
        self.start_epoch = start_epoch
        self.rows, self.comparisons = [], []
        self.pending = None
        self.measured = False
        self.suspended = False
        self.previous_end = time.perf_counter()
        self.stage_times, self.methods, self.counts = {}, {}, {}
        self.region_starts = {}
        if args.benchmark_detail:
            purifier._benchmark_trace = self
            for name, label in [('_tensor_to_pil', 'image_to_pil'),
                                ('_run_batch', 'qwen_batch_inclusive'),
                                ('_grounding_inference', 'grounding_inclusive')]:
                self.wrap(purifier, name, label)
            self.wrap(purifier.qwen_model, 'generate', 'qwen_generate', generation=True)
            self.wrap(purifier.processor, 'apply_chat_template', 'qwen_prompt')
            self.wrap(purifier.processor, 'batch_decode', 'qwen_decode')
            if purifier.use_sam:
                self.wrap(purifier.sam_predictor, 'set_image', 'sam_image_encode')
                self.wrap(purifier.sam_predictor, 'predict', 'sam_box_predict')
        self.metadata = {
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'seed': args.benchmark_seed,
            'checkpoint': str(Path(args.benchmark_checkpoint).resolve()),
            'start_epoch': start_epoch, 'batches_per_epoch': batches_per_epoch,
            'config': cfg, 'mode': 'detailed' if args.benchmark_detail else 'throughput',
            'qwen_device_map': {k: str(v) for k, v in
                                getattr(purifier.qwen_model, 'hf_device_map', {}).items()},
            'note': 'Detailed nested times are inclusive and perturb runtime. Output checks are untimed.'}

    def begin_region(self, name):
        if self.measured and not self.suspended:
            torch.cuda.synchronize()
            self.region_starts[name] = time.perf_counter()

    def end_region(self, name):
        if self.measured and not self.suspended:
            torch.cuda.synchronize()
            self.methods[name] = self.methods.get(name, 0.0) + time.perf_counter() - self.region_starts.pop(name)
            self.counts[name] = self.counts.get(name, 0) + 1

    def wrap(self, obj, name, label, generation=False):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.measured or self.suspended:
                return original(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.methods[label] = self.methods.get(label, 0.0) + time.perf_counter() - start
            self.counts[label] = self.counts.get(label, 0) + 1
            if generation:
                ids = kwargs.get('input_ids')
                if ids is not None:
                    # Decode work includes padding up to the longest sequence in a batch.
                    slots = int(result.shape[0] * (result.shape[1] - ids.shape[1]))
                    self.counts['qwen_generated_token_slots'] = self.counts.get('qwen_generated_token_slots', 0) + slots
                    self.counts['qwen_images'] = self.counts.get('qwen_images', 0) + int(ids.shape[0])
            return result
        setattr(obj, name, wrapped)

    def start_step(self, index):
        torch.cuda.synchronize()
        self.index = index
        self.measured = index >= self.args.benchmark_warmup
        self.begin = time.perf_counter()
        self.data_wait = self.begin - self.previous_end
        self.stage_start = self.begin
        self.stage_times, self.methods, self.counts = {}, {}, {}
        if self.measured:
            torch.cuda.reset_peak_memory_stats()

    def mark(self, name):
        if self.args.benchmark_detail and self.measured:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.stage_times[name.removesuffix('_ms') + '_s'] = now - self.stage_start
            self.stage_start = now

    def finish(self):
        return {}  # Compatible with StepTimer; writes structured records instead.

    def capture_inputs(self, images, labels, confidence, edge, ignore):
        if self.measured and self.args.benchmark_compare_batch:
            self.pending = (images.detach().clone(), labels.clone(), confidence.clone(),
                            edge.clone() if edge is not None else None, ignore.clone())

    def capture_outputs(self, confidence, labels):
        if self.pending is not None:
            self.reference = (confidence.clone(), labels.clone())

    def end_step(self, ct_weight):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.begin
        if self.measured:
            row = {'step': self.index, 'compute_step_s': elapsed,
                   'data_wait_s': self.data_wait, 'total_step_s': elapsed + self.data_wait,
                   'ctsa_weight': ct_weight,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                   'stages': dict(self.stage_times), 'methods_inclusive_s': dict(self.methods),
                   'calls_and_tokens': dict(self.counts)}
            self.rows.append(row)
            with (Path(self.args.save_path) / 'steps.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'[Benchmark] step={self.index} wall={row["total_step_s"]:.3f}s '
                  f'peak={row["peak_allocated_mib"]:.0f}MiB', flush=True)
        if self.pending is not None:
            self.suspended = True
            p = self.purifier
            saved = {key: getattr(p, key) for key in
                     ('qwen_batch_size', 'current_iter', 'cached_spatial_probs', 'cached_shape', '_prefetched')}
            try:
                images, labels, conf, edge, ignore = self.pending
                p.qwen_batch_size = self.args.benchmark_compare_batch
                with torch.no_grad():
                    output = p.get_qwen_purify(images, labels, conf, edge_map=edge)
                trial = output if p.use_vlm_on_mismatch else (output, labels)
                result = compare_outputs(self.reference, trial, edge, ignore, self.cfg)
                result['step'] = self.index
                self.comparisons.append(result)
                print('[Benchmark compare] ' + json.dumps(result), flush=True)
            finally:
                for key, value in saved.items():
                    setattr(p, key, value)
                self.pending = None
                self.reference = None
                self.suspended = False
        done = self.index + 1 >= self.args.benchmark_warmup + self.args.benchmark_steps
        if done:
            summary = {**self.metadata, 'timing': summarize([r['total_step_s'] for r in self.rows]),
                       'peak_allocated_mib': max(r['peak_allocated_mib'] for r in self.rows),
                       'peak_reserved_mib': max(r['peak_reserved_mib'] for r in self.rows),
                       'comparisons': self.comparisons}
            summary['estimated_epoch_hours'] = summary['timing']['mean_s'] * self.batches_per_epoch / 3600
            (Path(self.args.save_path) / 'summary.json').write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        torch.cuda.synchronize()
        self.previous_end = time.perf_counter()
        return done
