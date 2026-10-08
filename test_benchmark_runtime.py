"""CPU checks for benchmark guards and CEE output comparisons."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from util.benchmark_runtime import add_arguments, prepare, accepted_mask, compare_outputs, summarize, Benchmark


class BenchmarkTests(unittest.TestCase):
    def args(self, folder):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(['--benchmark'])
        args.init_checkpoint = None
        args.save_path = str(folder / 'new-run')
        args.benchmark_checkpoint = str(folder / 'latest.pth')
        Path(args.benchmark_checkpoint).touch()
        return args

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            Path(args.save_path).mkdir()
            with self.assertRaisesRegex(ValueError, 'NEW'):
                prepare(args, {})

    def test_weights_only_init_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.init_checkpoint = args.benchmark_checkpoint
            with self.assertRaisesRegex(ValueError, 'preserve'):
                prepare(args, {})

    def test_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            with self.assertRaisesRegex(ValueError, 'cache'):
                prepare(args, {'use_ctsa': True, 'vlm_type': 'qwen_vl', 'use_vlm_cache': True})

    def test_normal_training_is_untouched(self):
        args = argparse.Namespace(benchmark=False)
        cfg = {'profile_steps': 5, 'qwen_batch_size': 2}
        prepare(args, cfg)
        self.assertEqual(cfg, {'profile_steps': 5, 'qwen_batch_size': 2})

    def test_thresholds_and_ignored_pixels(self):
        cfg = {'use_cee': True, 'use_edge_threshold': True, 'conf_thresh': .95, 'edge_conf_thresh': .75}
        conf = torch.tensor([[[.8, .8, .99]]])
        edge = torch.tensor([[[True, False, True]]])
        ignore = torch.tensor([[[0, 0, 255]]])
        self.assertEqual(accepted_mask(conf, edge, ignore, cfg).tolist(), [[[True, False, False]]])
        trial = conf.clone()
        trial[0, 0, 0] = .7
        labels = torch.zeros_like(ignore)
        changed_labels = labels.clone()
        changed_labels[0, 0, 2] = 7  # ignored pixel must not count
        result = compare_outputs((conf, labels), (trial, changed_labels), edge, ignore, cfg)
        self.assertEqual(result['accepted_changed_pixels'], 1)
        self.assertEqual(result['label_changed_pixels'], 0)
        self.assertFalse(result['within_tolerance'])

    def test_nonfinite_output_fails(self):
        conf = torch.tensor([[[float('nan')]]])
        labels = torch.zeros((1, 1, 1), dtype=torch.long)
        result = compare_outputs((conf, labels), (conf, labels), None, labels, {'conf_thresh': .95})
        self.assertFalse(result['within_tolerance'])

    def test_summary_uses_all_samples(self):
        self.assertEqual(summarize([3., 1., 2.])['mean_s'], 2.)
        with self.assertRaises(ValueError):
            summarize([])

    def test_inclusive_profiler_and_suspension(self):
        bench = Benchmark.__new__(Benchmark)
        bench.measured, bench.suspended = True, False
        bench.methods, bench.counts = {}, {}
        obj = argparse.Namespace(run=lambda x: x + 1)
        with patch('torch.cuda.synchronize'):
            bench.wrap(obj, 'run', 'operation')
            self.assertEqual(obj.run(2), 3)
            bench.suspended = True
            self.assertEqual(obj.run(4), 5)
        self.assertEqual(bench.counts['operation'], 1)


if __name__ == '__main__':
    unittest.main()
