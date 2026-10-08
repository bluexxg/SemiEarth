"""Read-only asset and CUDA checks before starting a long experiment."""
import argparse
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset',choices=['loveda','potsdam'])
p.add_argument('split')
a=p.parse_args()
import yaml
cfg=yaml.safe_load(Path(f'configs/{a.dataset}.yaml').read_text(encoding='utf-8'))
errors=[]
def check(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:
        errors.append(str(path))
check(f'pretrained/{cfg["backbone"]}.pth')
check(cfg.get('sam_checkpoint','pretrained/sam_vit_b_01ec64.pth'))
q=Path(cfg['qwen_model_name'])
check(q/'config.json')
if not q.is_dir() or not any(x.stat().st_size for x in q.glob('*.safetensors')):
    errors.append(str(q)+' (missing nonempty safetensors)')
for split in [f'{a.split}/labeled.txt',f'{a.split}/unlabeled.txt','val.txt']:
    path=Path('splits')/a.dataset/split
    check(path)
    if path.is_file():
        lines=path.read_text().splitlines()
        for line in lines:
            if not line.strip():continue
            fields=line.split()
            if not split.endswith('unlabeled.txt') and len(fields)<2:
                errors.append(str(path)+' (missing label path)')
            for item in fields[:1 if split.endswith('unlabeled.txt') else 2]:
                check(Path(cfg['data_root'])/item)
print('Missing/empty asset count:',len(errors))
for e in errors[:20]:print('MISSING:',e)
import torch
print('Python:',sys.executable,'Torch:',torch.__version__,'CUDA:',torch.version.cuda)
if not torch.cuda.is_available():errors.append('CUDA unavailable')
else:print('GPU:',torch.cuda.get_device_name())
print('PASS' if not errors else 'FAIL: fix assets/environment before training')
print('This checks presence, not checkpoint loadability or model accuracy.')
raise SystemExit(bool(errors))
