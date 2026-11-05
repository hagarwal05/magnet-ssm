#!/bin/zsh
# Step 2 (2026-11-05): 3 seeds x {Dual-LSTM 771, S4D 205, S4D 1001} on the official A-E
# protocol, with the setting chosen by the noise sweep (results/sweep_noise): h_noise 0.1
# plus the multi-step stage (K=32). Same split (split-seed 1) for every run.
cd "/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798" || exit 1
echo "start $(date '+%H:%M')"
~/.venvs/magnet-ssm/bin/python -u -m src.experiments.final_eval \
  --models ssm_s official_lstm ssm_m --seeds 1 2 3 --tag "_hn0.1_ms32" \
  --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
  --rollout-val --rollout-val-every 5 --h-noise 0.1 \
  --ms-horizon 32 --ms-epochs 12 --ms-chunks 30000 --ms-warmup-epochs 4 \
  --device mps --out results/seeds3 --runs ~/magnet-data/runs/seeds3
echo "done $(date '+%H:%M')"
