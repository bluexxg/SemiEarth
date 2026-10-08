#!/usr/bin/env python3
"""Three-step, read-only grounding diagnostics using the existing benchmark path."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def install_hooks(cls, record):
    original_parse = cls._parse_output
    original_purify = cls.get_qwen_purify
    original_sam = cls._load_sam_model

    def parse(self, text):
        result = original_parse(self, text)
        record({'kind': 'qwen_output', 'classes': self.class_names,
                'prompt': self.grounding_prompt, 'raw_text': text,
                'parsed_detections': result,
                'box_count': sum(len(v) for v in result.values())})
        return result

    def load_sam(self):
        result = original_sam(self)
        for name in ('set_image', 'predict'):
            original = getattr(self.sam_predictor, name)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                value = _original(*args, **kwargs)
                record({'kind': 'sam_call', 'method': _name})
                return value
            setattr(self.sam_predictor, name, wrapped)
        return result

    def purify(self, images, pseudo_labels, conf_scores, edge_map=None):
        before_labels = pseudo_labels.clone()
        before_conf = conf_scores.clone()
        result = original_purify(self, images, pseudo_labels, conf_scores, edge_map)
        after_conf, after_labels = result if isinstance(result, tuple) else (result, pseudo_labels)
        record({'kind': 'purification', 'use_sam': self.use_sam,
                'labels_changed': int((after_labels != before_labels).sum().item()),
                'confidence_changed': int((after_conf != before_conf).sum().item()),
                'confidence_max_abs': float((after_conf-before_conf).abs().max().item())})
        return result

    cls._parse_output = parse
    cls._load_sam_model = load_sam
    cls.get_qwen_purify = purify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['loveda', 'potsdam'])
    parser.add_argument('split')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=29511)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--qwen-batch', type=int, default=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.steps < 1 or args.qwen_batch < 1:
        parser.error('Invalid port, steps or qwen-batch')
    os.chdir(ROOT)
    if not Path(args.checkpoint).is_file() or Path(args.checkpoint).stat().st_size == 0:
        parser.error('Checkpoint is missing or empty')
    if not args.worker:
        output = ROOT/'exp'/'vlm_diagnostics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=static',
               '--nnodes=1', '--nproc_per_node=1', '--node_rank=0',
               '--master_addr=127.0.0.1', f'--master_port={args.port}',
               str(Path(__file__).resolve()), args.dataset, args.split,
               '--checkpoint', str(Path(args.checkpoint).resolve()), '--worker',
               '--output', str(output), '--steps', str(args.steps),
               '--qwen-batch', str(args.qwen_batch)]
        print(f'Diagnostic output: {output}', flush=True)
        with (output/'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace',
                                       env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONHASHSEED='20260929'))
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        print(f'Please share this directory: {output}', flush=True)
        raise SystemExit(code)

    sys.path.insert(0, str(ROOT))
    output = Path(args.output)
    def record(data):
        with (output/'vlm_diagnostic.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
        if data['kind'] == 'qwen_output':
            print(f'[VLM diagnostic] parsed boxes={data["box_count"]}', flush=True)

    from model.semseg.vlm_pp import QwenVLPurifiedSemi
    install_hooks(QwenVLPurifiedSemi, record)
    from transformers import Qwen2_5_VLForConditionalGeneration
    original_generate = Qwen2_5_VLForConditionalGeneration.generate
    def generate(self, *a, **kw):
        result = original_generate(self, *a, **kw)
        ids = kw.get('input_ids')
        if ids is not None:
            tokens = result[:, ids.shape[1]:].detach().cpu().tolist()
            eos = self.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            lengths = []
            for row in tokens:
                position = next((i for i, v in enumerate(row) if v in eos), None)
                lengths.append({'tokens_through_eos': position+1 if position is not None else len(row),
                                'eos_seen': position is not None,
                                'limit_reached_without_eos': position is None and len(row) >= kw.get('max_new_tokens',256)})
            record({'kind': 'generation', 'max_new_tokens': kw.get('max_new_tokens'), 'sequences': lengths})
        return result
    Qwen2_5_VLForConditionalGeneration.generate = generate
    sys.argv = ['semiearth.py', '--config', f'configs/{args.dataset}.yaml',
                '--labeled-id-path', f'splits/{args.dataset}/{args.split}/labeled.txt',
                '--unlabeled-id-path', f'splits/{args.dataset}/{args.split}/unlabeled.txt',
                '--save-path', str(output/'probe'), '--benchmark',
                '--benchmark-checkpoint', str(Path(args.checkpoint).resolve()),
                '--benchmark-steps', str(args.steps), '--benchmark-warmup', '0',
                '--benchmark-qwen-batch', str(args.qwen_batch), '--benchmark-detail',
                '--benchmark-seed', '20260929']
    runpy.run_path(str(ROOT/'semiearth.py'), run_name='__main__')


if __name__ == '__main__':
    main()
