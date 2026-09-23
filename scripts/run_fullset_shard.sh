#!/bin/bash
# PREREG_fullset.md: a full-set thinking-on arm, one shard.
#   run_fullset_shard.sh <bench> <tag> <gpu> <shard i/n>
# Every shard of one arm runs on the same model of accelerator; the arm is never
# split across two. Shards own items[i::n] sliced before the already-done
# filter, so a restarted shard keeps its items.
set -u
BENCH=$1; TAG=$2; GPU=$3; SHARD=$4
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$VENV/bin/python" scripts/taichu_eval.py --bench "$BENCH" --thinking on \
     --max_new 8192 --shard "$SHARD" --tag "$TAG"
