#!/usr/bin/env bash
# M5-EMB: rebuild stage B + M3 with top-5 embedding neighbours (variant X). Needs work/emb/*_knn.parquet.
# FROM=n resumes at step n (1-based).
set -euo pipefail
export BER_VARIANT=X
PY=${PY:-python}
S=experiments/scripts
FROM=${FROM:-1}
N=0
step() { N=$((N + 1)); [ "$N" -lt "$FROM" ] && return 0; echo "=== $(date +%H:%M:%S) [$N] $*"; "$@"; }
step $PY $S/build_expanded.py --role val
step $PY $S/build_expanded.py --role tune
step $PY $S/build_expanded.py --role train
step $PY $S/train_matcher.py --name M1bEX --blocking CEX
step $PY $S/stage1_scores.py --blocking CEX --model M1bEX
step $PY $S/sibling_shards.py --blocking CEX
step $PY $S/train_matcher.py --name M2EX --blocking CEX --extra S_ --min-p1 0.01
step $PY $S/seed_scores.py --blocking CEX --model M2EX --prefix s2_
step $PY $S/score_stack.py --role rest --batch 60000
step $PY $S/comp_features.py
step $PY $S/comp_dropout.py --drop 0.19
step $PY $S/train_matcher.py --name M3X --blocking CEX --extra S_ QD_ --cascade-prefix s2_ --min-p1 0.01
step $PY $S/eval_m3_heldout.py --role rest --phase A
step $PY $S/eval_m3_heldout.py --role rest --phase B
echo "=== $(date +%H:%M:%S) DONE"
