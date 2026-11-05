#!/bin/zsh
# Step 1 (2026-11-05): h-noise sweep on S4D small, multi-step stage on (K=32), seed 1.
# Same recipe as phase 2 otherwise. One run per noise level -> picks the level for step 2.
cd "/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798" || exit 1
for HN in 0.02 0.05 0.1; do
  echo "=== h_noise $HN  $(date '+%H:%M')"
  ~/.venvs/magnet-ssm/bin/python -u -m src.experiments.final_eval \
    --models ssm_s --seeds 1 --tag "_hn${HN}_ms32" \
    --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
    --rollout-val --rollout-val-every 5 --h-noise $HN \
    --ms-horizon 32 --ms-epochs 12 --ms-chunks 30000 --ms-warmup-epochs 4 \
    --device mps --out results/sweep_noise --runs ~/magnet-data/runs/sweep_noise || exit 1
done
echo "sweep done $(date '+%H:%M')"
