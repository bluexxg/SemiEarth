"""CPU-capable correctness tests; no pretrained weights, Qwen or datasets needed.

Run: python -m unittest test_ctsa -v
"""
import copy
import importlib.util
import sys
import types
import unittest
import random
from unittest import mock
import numpy as np
import torch
from torch.nn import functional as F
from model.semseg.ctsa import CTSA
from model.semseg.dpt import DPT
from model.backbone.dinov2_layers.attention import Attention
from util.ctsa import prepare_ctsa, cutmix_tensor, auxiliary_loss, ctsa_weight
from util.cee import extract_multi_class_edge
from util.training_runtime import update_ema, load_segmentation_state, intersection_union
from util.utils import intersectionAndUnion

torch.set_num_threads(2)


class CTSATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_gradient_direction_and_zero_gate(self):
        module = CTSA(24, dim=16, heads=4, gate_statistics=True)
        student = torch.randn(2, 9, 24, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        gate = torch.ones(2, 9, 1)
        module(student, teacher, gate).square().mean().backward()
        self.assertIsNone(teacher.grad)
        for name in ('q', 'k', 'v', 'out'):
            self.assertGreater(getattr(module, name).weight.grad.abs().sum().item(), 0)
        module.zero_grad(set_to_none=True)
        out = module(student, teacher, torch.zeros_like(gate))
        torch.testing.assert_close(out, student, rtol=0, atol=0)
        out.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_no_cross_sample_attention(self):
        module = CTSA(24, dim=16, heads=4).eval()
        s, t = torch.randn(2, 9, 24), torch.randn(2, 9, 24)
        full = module(s, t)
        separate = torch.cat([module(s[i:i+1], t[i:i+1]) for i in range(2)])
        torch.testing.assert_close(full, separate)

    def test_cutmix_alignment_and_gate_modes(self):
        teacher = torch.stack([torch.ones(4, 3), torch.full((4, 3), 9.)])
        box = torch.zeros(2, 4, 4); box[:, :, 2:] = 1
        labels = torch.zeros(2, 4, 4, dtype=torch.long)
        raw = labels.clone(); raw[:, 0, 0] = 1
        conf = torch.full((2, 4, 4), .8)
        accepted = torch.ones_like(labels, dtype=torch.bool)
        mixed, gate, active = prepare_ctsa(teacher, raw, labels, conf, accepted, box, (2, 2))
        torch.testing.assert_close(mixed[0, :, 0], torch.tensor([1., 9., 1., 9.]))
        self.assertEqual(gate[0, 0, 0], 0)
        _, confidence_gate, _ = prepare_ctsa(teacher, raw, labels, conf, accepted, box,
                                            (2, 2), mode='confidence')
        self.assertGreater(confidence_gate[0, 0, 0], 0)
        box[:, :, 1:] = 1  # first patch column straddles the seam
        _, gate, _ = prepare_ctsa(teacher, labels, labels, conf, accepted, box, (2, 2), mode='none')
        self.assertEqual(gate[0, 0, 0], 0)
        self.assertEqual(gate[0, 2, 0], 0)

    def test_cee_acceptance_is_used_in_auxiliary_loss(self):
        labels = torch.zeros(2, 8, 8, dtype=torch.long); labels[:, :, 4:] = 1
        edges = extract_multi_class_edge(labels, edge_width=1)
        confidence = torch.full_like(labels, .8, dtype=torch.float)
        accepted = ((~edges) & (confidence >= .95)) | (edges & (confidence >= .75))
        self.assertTrue(torch.equal(accepted, edges))
        logits = torch.randn(2, 2, 8, 8, requires_grad=True)
        ignore = torch.zeros_like(labels)
        expected = (F.cross_entropy(logits, labels, reduction='none') * edges).sum() / labels.numel()
        result = auxiliary_loss(logits, labels, accepted, ignore, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result, expected)
        zero = auxiliary_loss(logits, labels, accepted, torch.full_like(ignore, 255),
                              torch.zeros(2, dtype=torch.bool))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sdpa_matches_manual_attention(self):
        layer = Attention(24, num_heads=4, qkv_bias=True).eval()
        x = torch.randn(2, 13, 24)
        q, k, v = layer.qkv(x).reshape(2, 13, 3, 4, 6).permute(2, 0, 3, 1, 4)
        expected = (((q * layer.scale) @ k.transpose(-1, -2)).softmax(-1) @ v)
        expected = layer.proj(expected.transpose(1, 2).reshape(2, 13, 24))
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)

    def test_gpu_metric_formula_matches_original_numpy(self):
        pred = torch.randint(0, 8, (2, 21, 19))
        target = torch.randint(0, 8, pred.shape); target[:, :2] = 255
        original_i, original_u, _ = intersectionAndUnion(pred.numpy(), target.numpy(), 8, 255)
        actual_i, actual_u = intersection_union(pred, target, 8)
        np.testing.assert_array_equal(actual_i.numpy(), original_i)
        np.testing.assert_array_equal(actual_u.numpy(), original_u)

    def test_real_dino_dpt_backward_checkpoint_and_ema(self):
        cfg = dict(use_ctsa=True, ctsa_dim=32, ctsa_heads=4)
        model = DPT('small', 3, features=16, out_channels=[16]*4, ctsa_config=cfg)
        teacher = copy.deepcopy(model); teacher.ctsa = None
        teacher.eval().requires_grad_(False)
        x = torch.randn(3, 3, 56, 56)
        with torch.no_grad():
            ordinary_before = model(x)
            _, feature = teacher(x[1:], return_last_feature=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.ctsa.k.weight.detach().clone()
        logits, aux = model(x, teacher_feature=feature, ctsa_gate=torch.ones(2, 16, 1), num_labeled=1)
        torch.testing.assert_close(logits, ordinary_before, atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(aux.shape), (2, 3, 56, 56))
        loss = logits.square().mean() + aux.square().mean()
        loss.backward(); opt.step()
        self.assertFalse(torch.equal(before, model.ctsa.k.weight))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        update_ema(model, teacher, 0)
        for name, value in teacher.named_parameters():
            torch.testing.assert_close(value, dict(model.named_parameters())[name])
        deployment = copy.deepcopy(teacher)
        load_segmentation_state(deployment, {'module.'+k:v for k,v in model.state_dict().items()})
        torch.testing.assert_close(deployment(x), model.eval()(x))
        # Original CEE checkpoint can initialize only the newly added CTSA parameters.
        load_segmentation_state(model, teacher.state_dict(), allow_missing_ctsa=True)

    def test_schedule(self):
        cfg = dict(ctsa_weight=.1, ctsa_warmup_ratio=.1, ctsa_ramp_ratio=.1)
        self.assertEqual(ctsa_weight(0, 100, cfg), 0)
        self.assertAlmostEqual(ctsa_weight(15, 100, cfg), .05)
        self.assertAlmostEqual(ctsa_weight(99, 100, cfg), .1)

    def test_single_strong_preserves_consumed_views(self):
        from PIL import Image
        from dataset.semi import SemiDataset
        image = Image.fromarray(np.random.default_rng(4).integers(0, 255, (56, 56, 3), dtype=np.uint8))
        legacy = SemiDataset.__new__(SemiDataset)
        legacy.name='loveda'; legacy.root=''; legacy.mode='train_u'
        legacy.size=28; legacy.ids=['image.png']; legacy.single_strong=False
        optimized = copy.copy(legacy); optimized.single_strong=True
        def seed():
            random.seed(7); np.random.seed(7); torch.manual_seed(7)
        with mock.patch('dataset.semi.Image.open', side_effect=lambda _: image.copy()):
            seed(); old = legacy[0]
            seed(); new = optimized[0]
        for expected, actual in zip([old[0], old[1], old[3], old[4]], new):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)



class VLMBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests exercise the REAL unmodified CEE policy and SAM-free
        # spatial construction, with only the external Qwen model substituted.
        stubs = {}
        if importlib.util.find_spec('transformers') is None:
            hf = types.ModuleType('transformers')
            hf.Qwen2_5_VLForConditionalGeneration = object
            hf.AutoProcessor = object
            stubs['transformers'] = hf
        if importlib.util.find_spec('qwen_vl_utils') is None:
            qwen = types.ModuleType('qwen_vl_utils')
            qwen.process_vision_info = lambda x: (None, None)
            stubs['qwen_vl_utils'] = qwen
        with mock.patch.dict(sys.modules, stubs):
            from model.semseg.vlm_pp import QwenVLPurifiedSemi
            from model.semseg.vlm_runtime import BatchedQwenVLPurifiedSemi
            cls.base, cls.batched = QwenVLPurifiedSemi, BatchedQwenVLPurifiedSemi

    def make_pair(self, use_cee=True, mismatch=True, minimum=0, cache=False):
        def setup(kind):
            obj = kind.__new__(kind)
            obj.vlm_pp_conf_threshold=.7; obj.current_iter=0
            obj.use_cee=use_cee; obj.use_cee_vlm=True; obj.cee_debug=False
            obj.min_edge_pixels=minimum; obj.edge_vlm_weight=1.5
            obj.use_vlm_on_mismatch=mismatch; obj.use_vlm_cache=cache
            obj.inference_interval=2; obj.cached_spatial_probs=None; obj.cached_shape=None
            obj.num_classes=3; obj.class_names=['road','building','tree']
            obj.use_sam=False; obj.bbox_confidence=.95; obj.grounding_prompt='Locate classes'
            obj.qwen_batch_size=2; obj._prefetched=None
            return obj
        base, batch = setup(self.base), setup(self.batched)
        def prediction(messages):
            pixel = messages[0]['content'][0]['image'].getpixel((0,0))[0]
            name = base.class_names[pixel % 3]
            return f'{name}: [0, 0, 8, 8]'
        base._run_qwen = prediction
        batch._run_batch = mock.Mock(side_effect=lambda conversations: [prediction(m) for m in conversations])
        # Fallback to original single-image method without loading a model.
        return base, batch, prediction

    def test_batched_matches_original_cee_masks_labels_and_confidence(self):
        for use_cee in (False, True):
            for mismatch in (False, True):
                base, batch, prediction = self.make_pair(use_cee, mismatch)
                images = torch.randn(4, 3, 8, 8)
                labels = torch.zeros(4, 8, 8, dtype=torch.long)
                conf = torch.full((4, 8, 8), .2)
                edges = torch.ones_like(labels, dtype=torch.bool); edges[1] = False
                with mock.patch.object(self.base, '_run_qwen', side_effect=prediction):
                    expected = base.get_qwen_purify(images, labels, conf, edges)
                    actual = batch.get_qwen_purify(images, labels, conf, edges)
                if mismatch:
                    for a,b in zip(actual, expected):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                else:
                    torch.testing.assert_close(actual, expected,rtol=0,atol=0)
                self.assertTrue(batch._run_batch.called)
                self.assertIsNone(batch._prefetched)

    def test_skip_and_cache_scheduling_preserved(self):
        base, batch, prediction = self.make_pair(minimum=1000)
        images = torch.randn(3,3,8,8); labels = torch.zeros(3,8,8,dtype=torch.long)
        conf = torch.full((3,8,8),.2); edge = torch.ones_like(labels,dtype=torch.bool)
        actual = batch.get_qwen_purify(images,labels,conf,edge)
        self.assertFalse(batch._run_batch.called)
        torch.testing.assert_close(actual[0],conf)
        base, batch, prediction = self.make_pair(cache=True)
        # Identical scheduling to the inherited implementation, including the
        # existing explicitly enabled cache; this optimization does not enable it.
        with mock.patch.object(self.base, '_run_qwen', side_effect=prediction):
            for _ in range(3):
                a=base.get_qwen_purify(images,labels,conf,edge)
                b=batch.get_qwen_purify(images,labels,conf,edge)
                for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(batch._run_batch.call_count,4)  # two chunks on calls 1 and 2


if __name__ == '__main__':
    unittest.main()
