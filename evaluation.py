import torch
import torch.distributed as dist
import torch.nn.functional as F
from util.training_runtime import intersection_union, autocast_settings


@torch.no_grad()
def evaluate(model, loader, mode, cfg, multiplier=None):
    # Accumulate on the GPU; synchronize the small count array only once.
    model.eval()
    model = model.module if hasattr(model, 'module') else model
    device = next(model.parameters()).device
    assert mode in ('original', 'sliding_window')
    counts = torch.zeros(2, cfg['nclass'], dtype=torch.int64, device=device)
    enabled, dtype = autocast_settings(cfg)
    enabled = enabled and cfg.get('eval_amp', False) and device.type == 'cuda'
    for img, mask, _ in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        ori_h, ori_w = img.shape[-2:]
        with torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w, device=device)
                row = 0
                while True:
                    col = 0
                    while True:
                        tile = img[:, :, row:row + grid, col:col + grid]
                        th, tw = tile.shape[-2:]
                        if multiplier:
                            tile = F.interpolate(tile, (max(multiplier, round(th / multiplier) * multiplier),
                                                       max(multiplier, round(tw / multiplier) * multiplier)),
                                                 mode='bilinear', align_corners=True)
                        output = model(tile)
                        output = F.interpolate(output.float(), (th, tw), mode='bilinear', align_corners=True)
                        final[:, :, row:row + th, col:col + tw] += output.softmax(1)
                        if col >= max(w - grid, 0):
                            break
                        col = min(col + max(1, int(grid * 2 / 3)), max(w - grid, 0))
                    if row >= max(h - grid, 0):
                        break
                    row = min(row + max(1, int(grid * 2 / 3)), max(h - grid, 0))
                logits = final
            else:
                if multiplier is not None:
                    size = ((512, 512) if multiplier == 512 else
                            (max(multiplier, int(ori_h / multiplier + .5) * multiplier),
                             max(multiplier, int(ori_w / multiplier + .5) * multiplier)))
                    img = F.interpolate(img, size, mode='bilinear', align_corners=True)
                logits = model(img)
                if multiplier is not None:
                    logits = F.interpolate(logits, (ori_h, ori_w), mode='bilinear', align_corners=True)
        intersection, union = intersection_union(logits.argmax(1), mask, cfg['nclass'])
        counts[0] += intersection
        counts[1] += union
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    values = counts.cpu().numpy()
    iou = values[0] / (values[1] + 1e-10) * 100.0
    return iou.mean(), iou
