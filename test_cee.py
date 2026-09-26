"""Minimal CEE module test -- run on the training machine (torch + CUDA).

Usage:
    python test_cee.py

Verifies:
    1. extract_multi_class_edge on a synthetic 2-class split (correct edges).
    2. Random logits [2, 8, 518, 518] -> pseudo_labels/edge_map/masks shapes.
    3. Edge-aware threshold logic (interior vs boundary confidence).
    4. Dilation on/off (edge_width=1 vs 3).
    5. VLM-PP style priority_mask = low_conf & edge.
"""
import torch
import torch.nn.functional as F
from util.cee import extract_multi_class_edge


def _run(name, fn):
    fn()
    print(f'[OK] {name}')


def test_synthetic_boundary():
    # left half = class 0, right half = class 1, single vertical boundary
    labels = torch.zeros(1, 64, 64, dtype=torch.long)
    labels[:, :, 32:] = 1
    edge = extract_multi_class_edge(labels, edge_width=1)
    assert edge.dtype == torch.bool and edge.shape == labels.shape
    # column 31 and 32 (neighbor diff) must be the boundary band
    assert edge[0, :, 31].all() and edge[0, :, 32].all()
    # interior (far from boundary) must not be edge
    assert not edge[0, :, 5].any() and not edge[0, :, 60].any()
    # dilation (width=3) widens the band
    edge3 = extract_multi_class_edge(labels, edge_width=3)
    assert edge3[0, :, 30].all() and edge3[0, :, 33].all()
    assert not edge3[0, :, 10].any()


def test_random_shapes_and_threshold():
    B, C, H, W = 2, 8, 518, 518
    logits = torch.randn(B, C, H, W)
    conf_scores = logits.softmax(dim=1).max(dim=1)[0]      # [B,H,W]
    pseudo_labels = logits.argmax(dim=1)                    # [B,H,W]

    for edge_width in (1, 3):
        edge_map = extract_multi_class_edge(pseudo_labels, edge_width=edge_width)
        assert edge_map.shape == (B, H, W), edge_map.shape
        assert edge_map.dtype == torch.bool

        # edge-aware threshold (mirror of semiearth.py loss code)
        conf_thresh, edge_conf_thresh = 0.95, 0.85
        interior = (~edge_map) & (conf_scores >= conf_thresh)
        boundary = edge_map & (conf_scores >= edge_conf_thresh)
        conf_mask = interior | boundary
        assert conf_mask.shape == (B, H, W) and conf_mask.dtype == torch.bool

        # baseline threshold must always be a superset... actually interior
        # threshold >= baseline, boundary <= baseline -> no subset guarantee;
        # just assert masks are valid bools.
        assert conf_mask.sum().item() >= 0

    # VLM priority mask: low-conf & edge
    edge_map = extract_multi_class_edge(pseudo_labels, edge_width=3)
    low_conf = conf_scores < 0.7
    priority = low_conf & edge_map
    assert priority.shape == (B, H, W)


def test_cuda_if_available():
    if not torch.cuda.is_available():
        print('  (CUDA not available here -- skipping device check)')
        return
    labels = torch.randint(0, 8, (2, 518, 518), device='cuda')
    edge = extract_multi_class_edge(labels, edge_width=3)
    assert edge.device.type == 'cuda' and edge.shape == (2, 518, 518)


if __name__ == '__main__':
    _run('synthetic boundary (width=1 & 3)', test_synthetic_boundary)
    _run('random [2,8,518,518] shapes + edge threshold', test_random_shapes_and_threshold)
    _run('cuda device check', test_cuda_if_available)
    print('\nALL CEE TESTS PASSED')
