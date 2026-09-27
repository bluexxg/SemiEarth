#!/usr/bin/env bash
set -euo pipefail
# bash scripts/train_ctsa.sh loveda 5_100 1 29501 [optional CEE best.pth]
dataset=${1:-loveda}
split=${2:-5_100}
gpus=${3:-1}
port=${4:-29501}
save_path="exp/${dataset}/semiearth_cee_ctsa/dinov2_small/${split}"
mkdir -p "$save_path"
extra=()
if [[ -n "${5:-}" ]]; then
    extra+=(--init-checkpoint "$5")
fi
torchrun --nnodes=1 --nproc_per_node="$gpus" --master_addr=127.0.0.1 --master_port="$port" \
    semiearth.py --config "configs/${dataset}.yaml" \
    --labeled-id-path "splits/${dataset}/${split}/labeled.txt" \
    --unlabeled-id-path "splits/${dataset}/${split}/unlabeled.txt" \
    --save-path "$save_path" --port "$port" "${extra[@]}" 2>&1 | tee -a "$save_path/out.log"
