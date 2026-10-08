#!/usr/bin/env python3
"""Run isolated Qwen batch 2/4 throughput trials and one detailed paired probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def launcher_command(port):
    # Static loopback avoids hostname discovery (numeric hostnames can resolve incorrectly).
    return [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=static',
            '--nnodes=1', '--nproc_per_node=1', '--node_rank=0',
            '--master_addr=127.0.0.1', f'--master_port={port}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['loveda', 'potsdam'])
    parser.add_argument('split')
    parser.add_argument('--checkpoint', required=True, help='Existing latest.pth containing model/EMA/optimizer/epoch')
    parser.add_argument('--config', help='Defaults to configs/<dataset>.yaml; never modified')
    parser.add_argument('--output', help='NEW output directory; default exp/benchmarks/<timestamp>')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--detail-steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--gpu', help='Optional CUDA_VISIBLE_DEVICES override, e.g. 0')
    parser.add_argument('--port', type=int, default=29511, help='Local rendezvous port (default: 29511)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.steps < 1 or args.detail_steps < 1 or args.warmup < 0:
        parser.error('steps/detail-steps must be positive; warmup must be nonnegative')
    os.chdir(ROOT)
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config or f'configs/{args.dataset}.yaml').resolve()
    labeled = ROOT / 'splits' / args.dataset / args.split / 'labeled.txt'
    unlabeled = labeled.with_name('unlabeled.txt')
    for p in (checkpoint, config, labeled, unlabeled):
        if not p.is_file():
            parser.error(f'Missing input: {p}')
    import yaml
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    if cfg['dataset'] != args.dataset:
        parser.error('Dataset argument and config disagree')
    output = Path(args.output or ('exp/benchmarks/' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [ROOT / 'util/grounding_parser.py', ROOT / 'scripts/benchmark_ctsa.py', ROOT / 'semiearth.py', ROOT / 'util/benchmark_runtime.py',
                  ROOT / 'util/cee.py', ROOT / 'util/ctsa.py',
                  ROOT / 'model/semseg/vlm_pp.py', ROOT / 'model/semseg/vlm_runtime.py',
                  ROOT / 'model/semseg/dpt.py', ROOT / 'dataset/semi.py', config, labeled, unlabeled]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in code_paths}
    checkpoint_hash = digest(checkpoint)
    manifest = {'python': sys.executable, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': checkpoint_hash, 'input_sha256': hashes,
                'arguments': vars(args), 'variants': []}
    env = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONUNBUFFERED='1')
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    manifest['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '(inherited default)')
    summary_path = output / 'benchmark_summary.json'
    try:
        trials = [('batch2', 2, args.steps, False), ('batch4', 4, args.steps, False),
                  ('detail_compare', 2, args.detail_steps, True)]
        for name, qbatch, steps, detailed in trials:
            if digest(checkpoint) != checkpoint_hash:
                raise RuntimeError('Checkpoint changed. Stop concurrent training and use a stable checkpoint.')
            for p in code_paths:
                key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                if digest(p) != hashes[key]:
                    raise RuntimeError(f'Input changed between trials: {p}')
            cmd = launcher_command(args.port) + ['semiearth.py',
                   '--config', str(config), '--labeled-id-path', str(labeled),
                   '--unlabeled-id-path', str(unlabeled), '--save-path', str(output / name),
                   '--benchmark', '--benchmark-checkpoint', str(checkpoint),
                   '--benchmark-steps', str(steps), '--benchmark-warmup', str(args.warmup),
                   '--benchmark-qwen-batch', str(qbatch), '--benchmark-seed', str(args.seed)]
            if detailed:
                cmd += ['--benchmark-detail', '--benchmark-compare-batch', '4']
            print(f'\nStarting {name}: warmup={args.warmup}, measured={steps}; logs: {output / (name + ".log")}', flush=True)
            log_path = output / (name + '.log')
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                raise RuntimeError(f'{name} failed (exit {code}); see {log_path}')
            if digest(checkpoint) != checkpoint_hash:
                raise RuntimeError('Checkpoint changed during trial; comparison invalid.')
            result = json.loads((output / name / 'summary.json').read_text(encoding='utf-8'))
            result['name'] = name
            result['oom_serial_fallback'] = 'Qwen batch OOM' in log_path.read_text(encoding='utf-8')
            manifest['variants'].append(result)
            summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        base, trial, detail = manifest['variants']
        speedup = base['timing']['mean_s'] / trial['timing']['mean_s']
        checks = detail['comparisons']
        meaningful = any(json.loads(line)['calls_and_tokens'].get('qwen_generate', 0) > 0
                         for line in (output / 'detail_compare' / 'steps.jsonl').read_text().splitlines())
        manifest.update(status='complete', speedup=speedup,
                        measured_time_halved=speedup >= 2,
                        sampled_output_checks_pass=bool(checks) and meaningful and
                        all(x['within_tolerance'] for x in checks),
                        purification_observed=meaningful,
                        note='Short screening only. Validate longer timing and mIoU before changing production settings.')
        report = ['# SemiEarth short benchmark', '', '| Trial | Mean seconds/step | Median | P90 | Peak allocated MiB |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for item in (base, trial):
            t = item['timing']
            report.append(f'| {item["name"]} | {t["mean_s"]:.3f} | {t["median_s"]:.3f} | {t["p90_s"]:.3f} | {item["peak_allocated_mib"]:.0f} |')
        report += ['', f'Batch 4 speedup: {speedup:.3f}x',
                   f'Sampled output checks pass: {manifest["sampled_output_checks_pass"]}',
                   f'Qwen activity observed in detailed probe: {meaningful}',
                   f'Serial fallback observed: {any(x["oom_serial_fallback"] for x in manifest["variants"])}',
                   '', 'Detailed stage records: detail_compare/steps.jsonl. Nested method times overlap;',
                   'do not sum them. Detailed run is not a throughput result.',
                   'Output checks compare identical teacher inputs, before CutMix; they do not establish mIoU equivalence.',
                   'No model checkpoint was written. Production configuration was not modified.']
        (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nResults: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
