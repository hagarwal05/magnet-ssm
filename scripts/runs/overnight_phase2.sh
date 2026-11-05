#!/bin/zsh
# Phase 2 (queued 2026-11-05 ~01:40): identical to phase 1 except --h-noise 0.05
# (Gaussian noise, std 0.05 in normalized units, on the H history during training) to test
# whether it reduces autoregressive drift. Starts when phase 1 (pid 44354) exits.
while kill -0 44354 2>/dev/null; do sleep 60; done
cd "/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798" || exit 1
echo "phase 2 start $(date '+%H:%M')"
exec ~/.venvs/magnet-ssm/bin/python -u -m src.experiments.final_eval \
  --models official_lstm ssm_s ssm_m \
  --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
  --rollout-val --rollout-val-every 5 --h-noise 0.05 \
  --device mps --out results/final_AtoE_hnoise --runs ~/magnet-data/runs/final_AtoE_hnoise
