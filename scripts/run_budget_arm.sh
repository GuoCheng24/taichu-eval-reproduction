#!/bin/bash
# PREREG_budget.md, arms T-A and T-B: the same 100-item MathVista subset, thinking
# on, greedy, on an idle L40. The arms differ only in --max_new.
#   run_budget_arm.sh <tag> <gpu> <max_new>
set -u
TAG=$1; GPU=$2; MAXNEW=$3
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$VENV/bin/python" scripts/taichu_eval.py --bench mathvista --thinking on \
     --subset 100 --max_new "$MAXNEW" --tag "$TAG"
