"""Pack the MagNetX Database (Dropbox download) into one compact HDF5 file per material.

The raw training data is ~33 GB of CSV text (~18 bytes per number) inside per-material zips.
The measurements are float32 at the source (a float32 round-trip changes them by ~5e-15, i.e.
CSV rounding only), so storing float32 loses nothing. With HDF5 shuffle+gzip each value takes
~3 bytes, so the whole database packs to roughly 5 GB.

Layout of the output folder (default data/packed/):

  train/<M>.h5      one per material: groups bucket_<i>/{B, H, T} (B, H: (n, L) float32,
                    NaN-padded rows if lengths differ; T: (n,) float32). Attribute
                    ``source`` names the zip or folder it came from.
  pretest/<M>_Testing_{Padded,True}.h5    copied as-is (already HDF5)
  final_test/<L>/   the A-E testing CSVs, copied as-is (the ground truth for A-E is in
                    data/Evaluation Tools/, committed separately)
  manifest.json     per file: sequences per bucket, shapes, sha256

Reads the training zips directly, without extracting them. Resumable: materials whose
output exists and matches the manifest are skipped.

  python -m src.pack_data --src "MagNetX Database" --out data/packed
  python -m src.pack_data --src ~/magnet-data/"MagNetX Database" --materials 3C90 N87
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

KNOWN = ("3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87")
FINAL = ("A", "B", "C", "D", "E")
FINAL_NAMES = {"A": "3C92", "B": "3C95", "C": "FEC007", "D": "FEC014", "E": "T37"}
H5_OPTS = dict(compression="gzip", compression_opts=4, shuffle=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(fobj) -> np.ndarray:
    return pd.read_csv(fobj, header=None, dtype=np.float64, engine="c").to_numpy()


def _check_float32(x: np.ndarray, label: str) -> None:
    finite = np.isfinite(x)
    if not finite.any():
        return
    v = x[finite]
    rel = np.max(np.abs(v.astype(np.float32).astype(np.float64) - v) / np.maximum(np.abs(v), 1e-30))
    if rel > 1e-6:
        raise ValueError(f"{label}: values are not float32-representable (max rel err {rel:.2e}); "
                         "refusing to pack lossily")


def _write_bucket(f: h5py.File, i: int, B, H, T, label: str) -> dict:
    if B.shape != H.shape or len(T) != len(B):
        raise ValueError(f"{label} bucket {i}: shapes B{B.shape} H{H.shape} T{T.shape}")
    _check_float32(B, f"{label} bucket {i} B")
    _check_float32(H, f"{label} bucket {i} H")
    g = f.create_group(f"bucket_{i}")
    rows = min(256, len(B)) or 1
    g.create_dataset("B", data=B.astype(np.float32), chunks=(rows, B.shape[1]), **H5_OPTS)
    g.create_dataset("H", data=H.astype(np.float32), chunks=(rows, H.shape[1]), **H5_OPTS)
    g.create_dataset("T", data=T.astype(np.float32))
    nan = np.isnan(B) | np.isnan(H)
    return {"n": int(len(B)), "length": int(B.shape[1]),
            "nan_padded_rows": int(nan.any(axis=1).sum()),
            "temps": sorted(float(t) for t in np.unique(T))}


def pack_material(read, material: str, source: str, out: Path) -> dict:
    """``read(i, k)`` returns the CSV array for bucket i and k in B/H/T, or None if absent."""
    tmp = out.with_suffix(".h5.partial")
    buckets = {}
    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        f.attrs["material"] = material
        f.attrs["source"] = source
        f.attrs["dtype_note"] = "float32; source CSVs are float32-representable (checked)"
        for i in range(1, 8):
            arrays = {k: read(i, k) for k in "BHT"}
            if all(a is None for a in arrays.values()):
                continue
            if any(a is None for a in arrays.values()):
                raise FileNotFoundError(f"{material} bucket {i}: missing {[k for k, a in arrays.items() if a is None]}")
            buckets[i] = _write_bucket(f, i, arrays["B"], arrays["H"], arrays["T"].reshape(-1), material)
            print(f"    bucket {i}: {buckets[i]['n']:5d} x {buckets[i]['length']}", flush=True)
    tmp.rename(out)
    return {"file": str(out.name), "source": source, "buckets": buckets,
            "n_sequences": sum(b["n"] for b in buckets.values()),
            "bytes": out.stat().st_size, "sha256": sha256(out), "seconds": round(time.time() - t0, 1)}


def zip_reader(zip_path: Path, material: str):
    z = zipfile.ZipFile(zip_path)
    names = {n.lower(): n for n in z.namelist()}

    def read(i, k):
        for cand in (f"{material}/{material}_{i}_{k}.csv", f"{material}_{i}_{k}.csv"):
            if cand.lower() in names:
                with z.open(names[cand.lower()]) as fobj:
                    return _read_csv(fobj)
        return None

    return read


def folder_reader(folder: Path, prefix: str):
    def read(i, k):
        p = folder / f"{prefix}_{i}_{k}.csv"
        return _read_csv(p) if p.exists() else None

    return read


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help='the "MagNetX Database" folder')
    ap.add_argument("--out", type=Path, default=Path("data/packed"))
    ap.add_argument("--materials", nargs="*", default=None,
                    help=f"subset of {list(KNOWN) + list(FINAL)} (default: all)")
    ap.add_argument("--skip-pretest", action="store_true")
    ap.add_argument("--skip-final-test", action="store_true")
    args = ap.parse_args(argv)

    src, out = args.src.expanduser(), args.out
    want = set(args.materials) if args.materials else set(KNOWN) | set(FINAL)
    unknown = want - set(KNOWN) - set(FINAL)
    if unknown:
        ap.error(f"unknown materials {sorted(unknown)}")
    (out / "train").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"train": {}, "pretest": {}, "final_test": {}}

    def save():
        man_path.write_text(json.dumps(manifest, indent=2))

    def done(key, m, path):
        e = manifest[key].get(m)
        return e is not None and path.exists() and path.stat().st_size == e.get("bytes")

    # training data: 10 known materials (zips) and A-E (CSV folders)
    for m in [x for x in KNOWN if x in want]:
        dst = out / "train" / f"{m}.h5"
        if done("train", m, dst):
            print(f"[skip] {m} already packed")
            continue
        zp = src / "Training Data" / f"{m}.zip"
        if not zp.exists():
            print(f"[miss] {zp} not found")
            continue
        print(f"[pack] {m} from {zp.name}", flush=True)
        manifest["train"][m] = pack_material(zip_reader(zp, m), m, f"Training Data/{m}.zip", dst)
        save()
    for L in [x for x in FINAL if x in want]:
        dst = out / "train" / f"{L}.h5"
        if done("train", L, dst):
            print(f"[skip] {L} already packed")
            continue
        folder = src / "MagNet_Challenge_2_Final_Evaluation" / "Training" / f"Material {L}"
        if not folder.exists():
            print(f"[miss] {folder} not found")
            continue
        print(f"[pack] {L} ({FINAL_NAMES[L]}) from {folder.name}/", flush=True)
        e = pack_material(folder_reader(folder, L), L,
                          f"MagNet_Challenge_2_Final_Evaluation/Training/Material {L}", dst)
        e["material_name"] = FINAL_NAMES[L]
        manifest["train"][L] = e
        save()

    # pretest test sets (already HDF5): copy
    if not args.skip_pretest:
        (out / "pretest").mkdir(exist_ok=True)
        for m in [x for x in KNOWN if x in want]:
            for kind in ("Padded", "True"):
                s = src / "MagNet_Challenge_2_Pretest" / m / f"{m}_Testing_{kind}.h5"
                d = out / "pretest" / s.name
                key = s.name
                if s.exists() and not done("pretest", key, d):
                    shutil.copy2(s, d)
                    manifest["pretest"][key] = {"bytes": d.stat().st_size, "sha256": sha256(d)}
                    save()
        print(f"[copy] pretest: {len(manifest['pretest'])} files")

    # final-test inputs (CSV): copy
    if not args.skip_final_test:
        for L in [x for x in FINAL if x in want]:
            sdir = src / "MagNet_Challenge_2_Final_Evaluation" / "Testing" / f"Material {L}"
            if not sdir.exists():
                continue
            ddir = out / "final_test" / L
            ddir.mkdir(parents=True, exist_ok=True)
            for s in sorted(sdir.glob("*.csv")):
                key = f"{L}/{s.name}"
                d = ddir / s.name
                if not done("final_test", key, d):
                    shutil.copy2(s, d)
                    manifest["final_test"][key] = {"bytes": d.stat().st_size, "sha256": sha256(d)}
                    save()
        print(f"[copy] final_test: {len(manifest['final_test'])} files")

    total = sum(e["bytes"] for sec in manifest.values() for e in sec.values())
    print(f"wrote {man_path}  ({total/1e9:.2f} GB under {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
