# Reference numbers

`leaderboard_mc2.csv` is transcribed verbatim from the "[News #13] Preliminary ranking of
32 teams" table in the upstream `README.md` (minjiechen/magnetchallenge-2 @ ab4913c).

Columns:

- `seq_err_p95_pct` is "Avg. 95% Seq. Error [%]": the 95th-percentile sequence relative
  error, averaged over the five final-test materials A to E. Computed by the organizers
  with the grading definition RMS(H_pred - H_meas) / RMS(H_meas).
- `energy_err_p95_pct` is "Avg. 95% Energy Error [%]", the grading definition, with the
  predicted-segment energy error normalized by the full-excitation core loss.
- `avg_params` is "Avg. Parameters" per material.

These numbers come from materials A to E with the organizers' scoring. They are directly
comparable only to our results on the same final-test files scored with the same
definitions (`seq_rms_p95`, `energy_rel_p95` with `energy_rel_denominator == "loss"`).
Results on the 10 known materials or on our own held-out split are a different test set.
Plots built from this table label it that way. See `docs/reference_audit.md`.
