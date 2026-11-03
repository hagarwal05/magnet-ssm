# Data

Download from the MagNet Challenge 2 Dropbox and place it here. Loaders search recursively
and are case-insensitive, so the original folder structure can be kept.

## Committing data (Git LFS)

Everything under `data/` except this README goes through Git LFS (`.gitattributes`), and
`data/cache/` is ignored. On the machine that has the data:

```bash
git lfs install                    # once per machine, after installing git-lfs
du -sh data/*                      # check the size first (limits below)
git add data/raw/3C90 data/raw/N87 # add only what the experiments use
git lfs ls-files                   # every data file must be listed here before you push
git commit -m "Add raw data for 3C90, N87"
git push
```

If `git lfs ls-files` doesn't list a data file, git-lfs wasn't active and the file was
committed as a plain blob. Unstage it (`git reset`), run `git lfs install`, and add it again.

GitHub limits (free plan): plain files are blocked over 100 MiB, LFS files over 2 GB, and the
account includes 10 GiB of LFS storage plus 10 GiB of LFS downloads per month. Every clone
that pulls the data counts against the download quota. The full dataset is likely
~20 GB as `.csv.gz` (estimate from ~103k sequences of ~12k steps), so commit the subset of
materials the experiments use rather than the whole Dropbox folder. To pull only some
data on a machine: `git lfs pull --include="data/raw/3C90/**"`.

Committed so far (commit `4c3060d`, ~250 MB in LFS):

- `Evaluation Tools/Ground Truth Data/Material {A..E} - <name>/`: final-test files for
  3C92, 3C95, FEC007, FEC014, T37 (padded H, true B/H/T, `Loss`)
- `Evaluation Tools/Evaluation_Read.m`: the organizers' scoring script
- `Result_Final.xlsx`: per-team, per-material scores (source of
  `results/reference/leaderboard_mc2_per_material.csv`)

Score a model on them:
`python -m src.evaluate --checkpoint <model.pt> --test-dir "data/Evaluation Tools/Ground Truth Data" --materials 3C92 3C95 FEC007 FEC014 T37`

Still needed for training: the raw training sequences (`raw/` below).

```
data/
  raw/            # "Training Data": <M>_<i>_B.csv, <M>_<i>_H.csv, <M>_<i>_T.csv, i = 1..7
                  #   10 known materials (3C90 ... N87) and the final A..E training sets
  pretest/        # MagNet_Challenge_2_Pretest: per-material test files (h5 or csv)
                  #   + any Normalization_Params.json that ships with the 771-param models
  final/          # "Final Evaluation" testing data for A..E:
                  #   <M>_Padded_H_seq.csv, <M>_True_B_seq.csv, <M>_True_T.csv
  cache/          # created automatically: float32 HDF5 copies of raw CSVs (faster loads)
```

Links (from PROJECT_CONTEXT.md):

- Main data: https://www.dropbox.com/scl/fo/o07x7fa6d5elev19pzd1w/AISuBEWf8LQVszFeoaTlqY4?rlkey=8ec7mqukfr9no61h193zb4qdb&dl=0
- All teams' results + final evaluation ground truth: https://www.dropbox.com/scl/fo/en5z0mef5cub7q574cknf/AA7moyzEJDOC398WwhvLzJs?rlkey=n9r8jqk6azh461tmxy1ixr3vm&dl=0

Small real data is already available without Dropbox:

- `third_party/magnetchallenge-2/tutorials/tutorial-1/3C90_Training.h5`: windowed 3C90 demo
- `third_party/magnetchallenge-2/tutorials/tutorial-3/3C90_Testing_{padded,true}.h5`: 105 test segments
- `third_party/magnetchallenge-2/tutorials/tutorial-2/data/N87_5_{B,H,T}.csv.gz`: 128 raw N87
  sequences, frequency bucket 5, 25 °C only (pass the folder as `--raw-dir`)
