"""Alignment and loss helpers. The existing CEE acceptance mask is an input."""
import torch
from torch.nn import functional as F


def cutmix_tensor(tensor, box):
    """Use exactly the training loop's flip(0) partner and binary pixel mask."""
    mask = box.bool()
    if tensor.ndim == mask.ndim + 1:
        mask = mask.unsqueeze(1)
    return torch.where(mask, tensor.flip(0), tensor)


@torch.no_grad()
def prepare_ctsa(teacher, raw_labels, labels, confidence, accepted, box,
                 patch_hw, mode='agreement', reject_corrected=True):
    """Teacher is [B,N,C] BEFORE CutMix; the other maps are AFTER CutMix.

    Pooling a binary CutMix mask also identifies patches crossing its seam.
    Feature mixing after ViT encoding is an approximation (global context).
    """
    if mode not in ('none', 'confidence', 'agreement'):
        raise ValueError('ctsa_gate must be none, confidence or agreement')
    ph, pw = patch_hw
    b, n, c = teacher.shape
    if n != ph * pw:
        raise ValueError('Teacher token count does not match patch grid')
    pool = lambda x: F.adaptive_avg_pool2d(x.float().unsqueeze(1), (ph, pw))
    fraction = pool(box)
    pure_patch = (fraction == 0) | (fraction == 1)
    feature_box = fraction >= 0.5
    feature_map = teacher.transpose(1, 2).reshape(b, c, ph, pw)
    mixed = torch.where(feature_box, feature_map.flip(0), feature_map)
    if mode == 'none':
        gate = torch.ones_like(fraction)
    else:
        reliability = accepted.float() * confidence.float()
        if mode == 'agreement':
            reliability = reliability * (raw_labels == labels)
        gate = pool(reliability)
        if mode == 'agreement' and reject_corrected:
            gate = gate * (pool(raw_labels != labels) == 0)
    gate = gate * pure_patch
    gate = gate.flatten(2).transpose(1, 2).contiguous()
    active = gate.flatten(1).sum(1) > 0
    return mixed.flatten(2).transpose(1, 2).contiguous(), gate, active


def auxiliary_loss(logits, labels, accepted, ignore, active):
    # Keep CEE's threshold selection AND the baseline non-ignore denominator.
    per_pixel = F.cross_entropy(logits.float(), labels, reduction='none', ignore_index=255)
    weight = accepted & (ignore != 255) & active[:, None, None]
    return (per_pixel * weight).sum() / (ignore != 255).sum().clamp_min(1)


def ctsa_weight(step, total_steps, cfg):
    progress = step / max(total_steps, 1)
    warmup = float(cfg.get('ctsa_warmup_ratio', 0.1))
    ramp = float(cfg.get('ctsa_ramp_ratio', 0.1))
    maximum = float(cfg.get('ctsa_weight', 0.1))
    if not (0 <= warmup < 1 and 0 <= ramp <= 1 - warmup and maximum >= 0):
        raise ValueError('Invalid CTSA warmup/ramp/weight')
    if progress < warmup:
        return 0.0
    return maximum * min(1.0, (progress - warmup) / max(ramp, 1e-12)) if ramp else maximum
