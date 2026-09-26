"""CEE (Coarse Edge Extraction) -- lightweight multi-class semantic edge extraction.

The idea is borrowed from MSEONet's coarse boundary extraction (`_get_boundry`
in `epfo_head.py`): from a multi-class coarse segmentation (e.g. teacher pseudo
labels), explicitly locate pixels lying on the boundary between *any two*
different semantic classes.

This module is intentionally kept as a pure tensor operation:
  * no learnable parameters,
  * no extra CNN / encoder,
  * negligible extra GPU memory (only a bool [B, H, W] mask),
  * runs entirely on the device of the input.

It is NOT a copy of the MSEONet network -- only the idea is reused, expressed
with cheap neighbor-difference + max-pool dilation instead of one-hot morpho-
logical opening. When `use_cee=False` nothing in this file is ever invoked.
"""

import os

import torch
import torch.nn.functional as F


def extract_multi_class_edge(labels, edge_width=3):
    """Extract a multi-class semantic edge map from integer pseudo labels.

    A pixel is an edge pixel iff it has a different semantic class from at
    least one of its 4-neighbors. Optionally the raw 1-pixel-wide boundary is
    dilated into a boundary band via max-pool2d.

    Args:
        labels: [B, H, W] int64 (or any integral dtype) pseudo labels, i.e.
            argmax(teacher_logits). All values are expected to be in
            [0, num_classes); the ignore value 255 is treated as "another
            class", i.e. it also produces edges (conservative choice).
        edge_width: dilation window (kernel size of max_pool2d). ``1`` (or
            ``0``) disables dilation. Even values are bumped to the next odd
            kernel so the band is symmetric.

    Returns:
        edge_map: [B, H, W] bool tensor, ``True`` == semantic-boundary pixel.
    """
    labels = labels.long()
    B, H, W = labels.shape

    # 1) single-pixel class-change detection along both spatial axes.
    h_diff = labels[:, :, 1:] != labels[:, :, :-1]   # [B, H, W-1]
    v_diff = labels[:, 1:, :] != labels[:, :-1, :]   # [B, H-1, W]

    edge = torch.zeros(B, H, W, dtype=torch.bool, device=labels.device)
    edge[:, :, 1:] |= h_diff
    edge[:, :, :-1] |= h_diff
    edge[:, 1:, :] |= v_diff
    edge[:, :-1, :] |= v_diff

    # 2) optional morphological dilation -> boundary band.
    if edge_width is not None and int(edge_width) > 1:
        k = int(edge_width)
        if k % 2 == 0:
            k += 1
        pad = (k - 1) // 2
        edge = F.max_pool2d(
            edge.float().unsqueeze(1), kernel_size=k, stride=1, padding=pad
        ).squeeze(1) > 0

    return edge


def _save_gray_pil(arr, path):
    from PIL import Image
    Image.fromarray(arr, mode='L').save(path)


def save_cee_debug(save_dir, tag, images, pseudo_labels, edge_map, conf_scores,
                   low_conf_mask, vlm_target_mask, refined_labels=None, nclass=8):
    """Save a debug panel of the CEE pipeline (default off).

    Args:
        save_dir: output directory (``<save_path>/debug_cee``).
        tag:       a string label, e.g. ``ep{epoch}_it{iter}``.
        images:    [B, 3, H, W] normalized input images.
        pseudo_labels / edge_map / conf_scores / low_conf_mask /
        vlm_target_mask / refined_labels: all [B, H, W].

    Saves per-batch-first-image:
        1. Original Image   2. Teacher Pseudo Label   3. CEE Edge Map
        4. Confidence Map   5. Low-confidence Mask    6. VLM Target Mask
        7. Refined Pseudo Label
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) original (unnormalized) RGB image
    from PIL import Image as _PILImage
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (images[0].detach().float().cpu() * std + mean).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    _PILImage.fromarray(img, mode='RGB').save(os.path.join(save_dir, f'{tag}_image.png'))

    def dump_map(t, name, kind):
        t = t[0].detach().float().cpu()
        if kind == 'label':                      # class id -> gray level
            t = t / max(nclass - 1, 1)
        elif kind == 'binary':                   # bool -> 0/1
            t = t.float()
        elif kind == 'conf':                     # confidence [0,1]
            t = t.clamp(0, 1)
        arr = (t.clamp(0, 1).numpy() * 255).astype('uint8')
        _save_gray_pil(arr, os.path.join(save_dir, f'{tag}_{name}.png'))

    dump_map(pseudo_labels, 'pseudo_label', 'label')
    dump_map(edge_map, 'edge_map', 'binary')
    dump_map(conf_scores, 'conf_map', 'conf')
    dump_map(low_conf_mask, 'low_conf_mask', 'binary')
    dump_map(vlm_target_mask, 'vlm_target_mask', 'binary')
    if refined_labels is not None:
        dump_map(refined_labels, 'refined_pseudo_label', 'label')
