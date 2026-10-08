#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset=${1:-loveda}
split=${2:-5_100}
gpus=${3:-1}
port=${4:-29501}
python_bin=${PYTHON_BIN:-python}
config=${CONFIG:-configs/${dataset}.yaml}
save_path=${SAVE_PATH:-exp/${dataset}/semiearth_cee_ctsa/dinov2_small/${split}}
extra=()
if [[ -n "${5:-}" ]]; then
    [[ -s "$5" ]] || { echo "Initialization checkpoint missing or empty: $5" >&2; exit 1; }
    [[ ! -e "$save_path/latest.pth" ]] || { echo "Use a NEW SAVE_PATH for initialization." >&2; exit 1; }
    extra+=(--init-checkpoint "$5")
fi
if [[ -e "$save_path/latest.pth" && ! -s "$save_path/latest.pth" ]]; then
    echo "Empty latest.pth: choose a new SAVE_PATH or restore a valid checkpoint." >&2
    exit 1
fi
mkdir -p "$save_path"
"$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node="$gpus"     --master_addr=127.0.0.1 --master_port="$port"     semiearth.py --config "$config"     --labeled-id-path "splits/${dataset}/${split}/labeled.txt"     --unlabeled-id-path "splits/${dataset}/${split}/unlabeled.txt"     --save-path "$save_path" --port "$port" "${extra[@]}" 2>&1 | tee -a "$save_path/out.log"
