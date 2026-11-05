#!/bin/zsh
# Overnight run 2026-11-05: official A-E protocol, 3 models x 5 materials, MPS.
cd "/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798" || exit 1
exec ~/.venvs/magnet-ssm/bin/python -u -m src.experiments.final_eval \
  --models official_lstm ssm_s ssm_m \
  --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
  --rollout-val --rollout-val-every 5 \
  --device mps --out results/final_AtoE
