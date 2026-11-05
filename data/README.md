# Data

Two kinds of data, handled differently:

| | what | where it lives | in git? |
|---|---|---|---|
| **Small reference data** | A–E ground truth, scoring script, leaderboard workbook, team submissions | `data/Evaluation Tools/`, `data/Result_Final.xlsx`, `data/Competition Result/` | yes, Git LFS (~3.2 GB) |
| **Training database** | MagNetX Database: raw training sequences, pretest sets, A–E training/testing inputs | downloaded per machine, packed into `data/packed/` | **no** (23 GB) |

## Training database: download once per machine, then pack

1. Download the **MagNetX Database** folder from the main MagNet Challenge 2 Dropbox
   (link below). It contains `Training Data/` (10 per-material zips), `MagNet_Challenge_2_Pretest/`
   and `MagNet_Challenge_2_Final_Evaluation/`. Put it anywhere, e.g. `~/magnet-data/`.
   It is git-ignored if you put it in the project root.
2. Pack it (reads the zips in place, ~10 min, ~5 GB output):

   ```bash
   python -m src.pack_data --src ~/magnet-data/"MagNetX Database" --out data/packed
   ```

   Every number is float32 in the source CSVs, so packing loses nothing; this was checked on
   every bucket of N87 against the CSVs. `data/packed/manifest.json` records sequence counts
   and a sha256 per file.
3. Use it: `--raw-dir data/packed` for `src.train` and the experiments. The pretest test sets
   are in `data/packed/pretest/`: `src.evaluate --test-dir data/packed/pretest --materials 3C90 ...`.

After packing, the original download (`MagNetX Database/` and `MagNetX Database.zip`) is only
needed to re-pack. The 24.8 GB `MagNetX Database.zip` duplicates the extracted folder
exactly (all 146 files, same sizes), so it can go.

On the GPU cluster: download from Dropbox there and run the same pack command.

### What is in the database (checked 2026-11-05)

| set | contents |
|---|---|
| `Training Data/<M>.zip` | 3C90 3C94 3E6 3F4 77 78 N27 N30 N87: 7 frequency buckets; N49: buckets 1–5. Each bucket: `<M>_<i>_{B,H,T}.csv`, rows = sequences at 16 MHz, lengths 32016 (50 kHz) to 2016 (800 kHz); T ∈ {25, 50, 70} °C. Plus `Test_Info.xlsx`. 33.5 GB of CSV unpacked. |
| `MagNet_Challenge_2_Pretest/<M>/` | `<M>_Testing_{Padded,True}.h5`: 6,300 segments per material (N49: 4,500), warm-ups 100 / 500 / 900, with `Loss`. No normalization files or model weights. |
| `MagNet_Challenge_2_Final_Evaluation/Training/Material {A..E}` | raw training CSVs for the five new materials. A: 105 sequences (7 buckets); B: buckets 1–5 only. |
| `MagNet_Challenge_2_Final_Evaluation/Testing/Material {A..E}` | test inputs (padded H, B, T). The ground truth is in `data/Evaluation Tools/`. |

Sequence counts per material are higher than the organizers' slide (e.g. 3C90: 17,269 vs
13,587), because the slide predates later uploads. Counts are recorded in the manifest.

## Small reference data (Git LFS)

Everything under `data/` except this README goes through Git LFS (`.gitattributes`);
`data/cache/` and `data/packed/` are ignored.

- `Evaluation Tools/Ground Truth Data/Material {A..E} - <name>/`: final-test files for
  3C92, 3C95, FEC007, FEC014, T37 (padded H, true B/H/T, `Loss`)
- `Evaluation Tools/Evaluation_Read.m`: the organizers' scoring script
- `Result_Final.xlsx`: per-team, per-material scores (source of
  `results/reference/leaderboard_mc2_per_material.csv`)
- `Competition Result/<Team>/Result/H_Material {A..E}.csv` (~2.9 GB): every team's submitted
  predictions. **Don't `git lfs pull` everything.** Pull a team when you need it:
  `git lfs pull --include="data/Competition Result/Hefei/Result/*"`. To skip these files on
  every pull on a machine: `git config lfs.fetchexclude "data/Competition Result/**"`.

Score a model on A–E:
`python -m src.evaluate --checkpoint <model.pt> --test-dir "data/Evaluation Tools/Ground Truth Data" --materials 3C92 3C95 FEC007 FEC014 T37`

GitHub limits (free plan): plain files are blocked over 100 MiB, LFS files over 2 GB, and the
account includes 10 GiB of LFS storage plus 10 GiB of LFS downloads per month.

## Links

- Main data (MagNetX Database): https://www.dropbox.com/scl/fo/o07x7fa6d5elev19pzd1w/AISuBEWf8LQVszFeoaTlqY4?rlkey=8ec7mqukfr9no61h193zb4qdb&dl=0
- All teams' results + final evaluation ground truth: https://www.dropbox.com/scl/fo/en5z0mef5cub7q574cknf/AA7moyzEJDOC398WwhvLzJs?rlkey=n9r8jqk6azh461tmxy1ixr3vm&dl=0

Small real data available without Dropbox (in the reference clone):

- `third_party/magnetchallenge-2/tutorials/tutorial-1/3C90_Training.h5`: windowed 3C90 demo
- `third_party/magnetchallenge-2/tutorials/tutorial-3/3C90_Testing_{padded,true}.h5`: 105 test segments
- `third_party/magnetchallenge-2/tutorials/tutorial-2/data/N87_5_{B,H,T}.csv.gz`: 128 raw N87
  sequences, frequency bucket 5, 25 °C only
