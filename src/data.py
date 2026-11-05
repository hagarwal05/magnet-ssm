"""Data layer: HDF5 / CSV loading, windowing, normalization and train/val splits.

Three on-disk formats are supported:

* Windowed training HDF5 (``<M>_Training.h5``, tutorial-1 layout): groups ``B_seq_f_i``,
  ``H_seq_f_i`` (n, 80), ``B_scal_i``, ``H_scal_i``, ``T_i`` (n, 1) for buckets i = 1..7.
* Raw sequences (the Dropbox "Training Data"): ``<M>_<i>_B.csv``, ``<M>_<i>_H.csv``,
  ``<M>_<i>_T.csv`` per frequency bucket i; rows are sequences, columns are 16 MHz steps.
* Test sets of 1000-step segments, as HDF5 pairs (``B_seq``, ``H_seq``, ``T``, optional
  ``Loss``) or as the CSVs of the final evaluation (``<M>_Padded_H_seq.csv``,
  ``<M>_True_B_seq.csv``, ``<M>_True_T.csv``, optional ``<M>_True_H_seq.csv``).
  The padded H holds the true warm-up prefix followed by NaN. The warm-up length is read
  per sequence from the NaN position, never assumed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch

from .normalize import NormParams

WINDOW = 80
SEGMENT_LENGTH = 1000
DEFAULT_WARMUPS = (100, 300, 500, 700, 900)


# =========================================================================================
# Windowed (teacher-forcing) samples
# =========================================================================================

@dataclass
class WindowSet:
    """Next-step samples: history windows -> next H. ``group`` identifies the source
    sequence (or -1 when unknown), so splits can keep a sequence on one side."""

    seq_B: np.ndarray   # (N, W)
    seq_H: np.ndarray   # (N, W)
    B_next: np.ndarray  # (N,)
    H_next: np.ndarray  # (N,)
    T: np.ndarray       # (N,)
    group: np.ndarray   # (N,) int64
    source: np.ndarray  # (N,) material name per sample

    def __len__(self) -> int:
        return len(self.H_next)

    def subset(self, idx) -> "WindowSet":
        return WindowSet(self.seq_B[idx], self.seq_H[idx], self.B_next[idx], self.H_next[idx],
                         self.T[idx], self.group[idx], self.source[idx])

    @staticmethod
    def concat(sets: Sequence["WindowSet"]) -> "WindowSet":
        # Re-number groups so ids from different files never collide.
        groups, offset = [], 0
        for s in sets:
            g = s.group.copy()
            known = g >= 0
            g[known] += offset
            offset = (g.max() + 1) if known.any() else offset
            groups.append(g)
        return WindowSet(
            np.concatenate([s.seq_B for s in sets]),
            np.concatenate([s.seq_H for s in sets]),
            np.concatenate([s.B_next for s in sets]),
            np.concatenate([s.H_next for s in sets]),
            np.concatenate([s.T for s in sets]),
            np.concatenate(groups),
            np.concatenate([s.source for s in sets]),
        )

    def fit_norm(self, tie: bool = True) -> NormParams:
        """Fit normalization on these samples. ``tie=True`` (our default) uses one set of B
        statistics for the window and the next-B input, and one set of H statistics for the
        window and the target, so predictions can be fed back without a change of units."""
        p = NormParams.fit(self.seq_B, self.seq_H, self.B_next, self.H_next, self.T)
        if tie:
            d = p.to_dict()
            d.update(mean_Scal=d["mean_B"], std_Scal=d["std_B"], mean_out=d["mean_H"], std_out=d["std_H"])
            p = NormParams.from_dict(d)
        return p

    def to_tensors(self, norm: NormParams, device="cpu") -> tuple[torch.Tensor, ...]:
        """Normalized float32 tensors (seq_B[N,W,1], seq_H[N,W,1], B_next[N,1], T[N,1], H_next[N,1])."""

        def t(x, shape):
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape).to(device)

        n, w = self.seq_B.shape
        return (
            t(norm.norm_B(self.seq_B), (n, w, 1)),
            t(norm.norm_H(self.seq_H), (n, w, 1)),
            t(norm.norm_Scal(self.B_next), (n, 1)),
            t(norm.norm_T(self.T), (n, 1)),
            t(norm.norm_out(self.H_next), (n, 1)),
        )


def _bucket_ids(keys: Iterable[str], prefix: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    return sorted(int(m.group(1)) for k in keys if (m := pat.match(k)))


def load_training_h5(path: str | Path, buckets: Optional[Sequence[int]] = None,
                     material: Optional[str] = None) -> WindowSet:
    """Read a tutorial-1 style windowed training file and concatenate its buckets (1..7)."""
    path = Path(path)
    material = material or path.name.split("_")[0]
    parts = []
    with h5py.File(path, "r") as f:
        ids = list(buckets) if buckets is not None else _bucket_ids(f.keys(), "B_seq_f_")
        if not ids:
            raise ValueError(f"{path}: no B_seq_f_<i> groups found (keys: {sorted(f.keys())[:10]}...)")
        for i in ids:
            parts.append((
                f[f"B_seq_f_{i}"][:], f[f"H_seq_f_{i}"][:],
                f[f"B_scal_{i}"][:].reshape(-1), f[f"H_scal_{i}"][:].reshape(-1),
                f[f"T_{i}"][:].reshape(-1),
            ))
    seq_B, seq_H, B_next, H_next, T = (np.concatenate(x).astype(np.float64) for x in zip(*parts))
    if seq_B.ndim != 2 or seq_B.shape != seq_H.shape or len(B_next) != len(seq_B):
        raise ValueError(f"{path}: inconsistent shapes {seq_B.shape}, {seq_H.shape}, {B_next.shape}")
    n = len(B_next)
    return WindowSet(seq_B, seq_H, B_next, H_next, T,
                     group=np.full(n, -1, dtype=np.int64), source=np.full(n, material))


def split_indices(n: int, fractions: Sequence[float], seed: int = 1,
                  groups: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Split ``range(n)`` by ``fractions`` (summing to 1).

    Without groups (or when every group id is -1) this is a plain random split, the same
    as the upstream ``random_split(0.8, 0.1, 0.1)``. With groups, whole groups (source
    sequences) are assigned to one side, so overlapping windows never straddle the split.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    if groups is None or np.all(groups < 0):
        perm = rng.permutation(n)
        cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
        return np.split(perm, cuts)
    uniq = rng.permutation(np.unique(groups))
    cuts = np.cumsum([int(round(f * len(uniq))) for f in fractions[:-1]])
    parts = np.split(uniq, cuts)
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def iterate_batches(tensors: Sequence[torch.Tensor], batch_size: int, shuffle: bool,
                    generator: Optional[torch.Generator] = None):
    """Yield aligned mini-batches from tensors that already live on the target device."""
    n = tensors[0].shape[0]
    if shuffle:
        order = torch.randperm(n, generator=generator).to(tensors[0].device)
    else:
        order = torch.arange(n, device=tensors[0].device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield tuple(t[idx] for t in tensors)


# =========================================================================================
# Raw sequences (Dropbox training CSVs)
# =========================================================================================

@dataclass
class RawBucket:
    """All sequences of one material at one frequency bucket. Rows may be NaN-padded."""

    material: str
    bucket: int
    B: np.ndarray  # (n, L)
    H: np.ndarray  # (n, L)
    T: np.ndarray  # (n,)
    seq_id: np.ndarray = field(default=None)  # (n,) globally unique ids, assigned on load

    def rows(self, idx) -> "RawBucket":
        return RawBucket(self.material, self.bucket, self.B[idx], self.H[idx], self.T[idx], self.seq_id[idx])

    def valid_lengths(self) -> np.ndarray:
        bad = np.isnan(self.B) | np.isnan(self.H)
        return np.where(bad.any(axis=1), bad.argmax(axis=1), self.B.shape[1])


def _find_file(root: Path, filename: str) -> Optional[Path]:
    """Case-insensitive search for ``filename`` (or ``filename + '.gz'``) under root."""
    wanted = {filename.lower(), filename.lower() + ".gz"}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower() in wanted:
                return Path(dirpath) / name
    return None


def _read_matrix(path: Path) -> np.ndarray:
    import pandas as pd

    return pd.read_csv(path, header=None, dtype=np.float64).to_numpy()


_SEQ_COUNTER = [0]


def _cached_bucket(files: dict, cache: Optional[Path], material: str, i: int):
    """Read one bucket's CSVs, through an HDF5 cache when ``cache`` is set. The cache stores
    float32 (the measurements carry far fewer significant digits) and is ~10x faster to load."""
    if cache is not None:
        p = cache / f"{material}_{i}.h5"
        if p.exists():
            with h5py.File(p, "r") as f:
                return f["B"][:].astype(np.float64), f["H"][:].astype(np.float64), f["T"][:].astype(np.float64)
    B, H = _read_matrix(files["B"]), _read_matrix(files["H"])
    T = _read_matrix(files["T"]).reshape(-1)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache / f"{material}_{i}.h5", "w") as f:
            f.create_dataset("B", data=B.astype(np.float32), compression="lzf")
            f.create_dataset("H", data=H.astype(np.float32), compression="lzf")
            f.create_dataset("T", data=T.astype(np.float32))
    return B, H, T


def load_packed_material(path: str | Path, buckets: Optional[Sequence[int]] = None) -> list[RawBucket]:
    """Load one material packed by ``src.pack_data`` (``train/<M>.h5``)."""
    path = Path(path)
    out = []
    with h5py.File(path, "r") as f:
        material = str(f.attrs.get("material", path.stem))
        ids = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("bucket_"))
        for i in (buckets or ids):
            g = f[f"bucket_{i}"]
            B, H, T = g["B"][:].astype(np.float64), g["H"][:].astype(np.float64), g["T"][:].astype(np.float64)
            seq = np.arange(_SEQ_COUNTER[0], _SEQ_COUNTER[0] + len(B))
            _SEQ_COUNTER[0] += len(B)
            out.append(RawBucket(material, i, B, H, T, seq))
    return out


def load_raw_material(root: str | Path, material: str, buckets: Optional[Sequence[int]] = None,
                      cache: Optional[str | Path] = None) -> list[RawBucket]:
    """Load one material's raw training sequences from ``root``, which is either a packed
    folder (``<root>/train/<M>.h5`` or ``<root>/<M>.h5`` from ``src.pack_data``) or a folder
    of ``<M>_<i>_{B,H,T}.csv`` (optionally ``.csv.gz``), searched recursively.
    ``cache``: folder for the CSV -> HDF5 cache (e.g. data/cache); unused for packed data."""
    root = Path(root)
    for packed in (root / "train" / f"{material}.h5", root / f"{material}.h5"):
        if packed.exists():
            return load_packed_material(packed, buckets)
    cache = Path(cache) if cache is not None else None
    out = []
    for i in (buckets or range(1, 8)):
        files = {k: _find_file(root, f"{material}_{i}_{k}.csv") for k in ("B", "H", "T")}
        cached = cache is not None and (cache / f"{material}_{i}.h5").exists()
        if any(v is None for v in files.values()) and not cached:
            if buckets is not None:
                raise FileNotFoundError(f"{material} bucket {i}: missing {[k for k, v in files.items() if v is None]} under {root}")
            continue
        B, H, T = _cached_bucket(files, cache, material, i)
        if B.shape != H.shape or len(T) != len(B):
            raise ValueError(f"{material} bucket {i}: shapes B{B.shape} H{H.shape} T{T.shape}")
        ids = np.arange(_SEQ_COUNTER[0], _SEQ_COUNTER[0] + len(B))
        _SEQ_COUNTER[0] += len(B)
        out.append(RawBucket(material, i, B, H, T, ids))
    if not out:
        raise FileNotFoundError(f"no raw CSVs for material {material!r} under {root}")
    return out


def split_raw(buckets: Sequence[RawBucket], fractions: Sequence[float], seed: int = 1) -> list[list[RawBucket]]:
    """Sequence-level split of raw buckets, stratified by bucket (frequency)."""
    rng = np.random.default_rng(seed)
    parts: list[list[RawBucket]] = [[] for _ in fractions]
    for b in buckets:
        perm = rng.permutation(len(b.T))
        cuts = np.cumsum([int(round(f * len(perm))) for f in fractions[:-1]])
        for k, idx in enumerate(np.split(perm, cuts)):
            if len(idx):
                parts[k].append(b.rows(np.sort(idx)))
    return parts


@dataclass
class ChunkSet:
    """Multi-step training samples: ``window + horizon`` consecutive steps of B and H.

    The model starts from the true H over the first ``window`` steps and predicts the next
    ``horizon`` steps autoregressively, feeding its own predictions back (``src.train``)."""

    B: np.ndarray       # (N, window + horizon)
    H: np.ndarray       # (N, window + horizon)
    T: np.ndarray       # (N,)
    group: np.ndarray   # (N,) source sequence id
    window: int
    horizon: int

    def __len__(self) -> int:
        return len(self.T)

    def to_tensors(self, norm: NormParams, device="cpu") -> tuple[torch.Tensor, ...]:
        """(B_win[N,W+K] normalized as B, B_next[N,W+K] normalized as Scal, H[N,W+K] as H,
        T[N,1]). Callers slice the time axis."""
        def t(x):
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        return (t(norm.norm_B(self.B)), t(norm.norm_Scal(self.B)), t(norm.norm_H(self.H)),
                t(norm.norm_T(self.T)).reshape(-1, 1))


def chunks_from_raw(buckets: Sequence[RawBucket], horizon: int, window: int = WINDOW,
                    per_seq: int = 8, seed: int = 1) -> ChunkSet:
    """Sample ``per_seq`` chunks of ``window + horizon`` steps uniformly from each sequence."""
    rng = np.random.default_rng(seed)
    L = window + horizon
    Bs, Hs, Ts, Gs = [], [], [], []
    offs = np.arange(L)[None, :]
    for b in buckets:
        lengths = b.valid_lengths()
        for r in range(len(b.T)):
            n = int(lengths[r])
            if n < L:
                continue
            starts = rng.integers(0, n - L + 1, size=per_seq)
            idx = starts[:, None] + offs
            Bs.append(b.B[r, idx].astype(np.float32))
            Hs.append(b.H[r, idx].astype(np.float32))
            Ts.append(np.full(per_seq, b.T[r], dtype=np.float32))
            Gs.append(np.full(per_seq, b.seq_id[r], dtype=np.int64))
    if not Bs:
        raise ValueError(f"no sequence has {L} valid steps")
    return ChunkSet(np.concatenate(Bs), np.concatenate(Hs), np.concatenate(Ts), np.concatenate(Gs),
                    window, horizon)


def windows_from_raw(buckets: Sequence[RawBucket], window: int = WINDOW, per_seq: int = 64,
                     seed: int = 1) -> WindowSet:
    """Sample ``per_seq`` next-step windows uniformly from each sequence (targets at index
    >= window). Stored as float32: a full material is ~10k sequences, so 64 windows per
    sequence is already ~0.4 GB."""
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("seq_B", "seq_H", "B_next", "H_next", "T", "group", "source")}
    offs = np.arange(-window, 0)[None, :]
    for b in buckets:
        lengths = b.valid_lengths()
        for r in range(len(b.T)):
            L = int(lengths[r])
            if L <= window:
                continue
            k = min(per_seq, L - window)
            m = np.sort(rng.choice(np.arange(window, L), size=k, replace=False))
            idx = m[:, None] + offs
            parts["seq_B"].append(b.B[r, idx].astype(np.float32))
            parts["seq_H"].append(b.H[r, idx].astype(np.float32))
            parts["B_next"].append(b.B[r, m].astype(np.float32))
            parts["H_next"].append(b.H[r, m].astype(np.float32))
            parts["T"].append(np.full(k, b.T[r], dtype=np.float32))
            parts["group"].append(np.full(k, b.seq_id[r], dtype=np.int64))
            parts["source"].append(np.full(k, b.material))
    if not parts["T"]:
        raise ValueError("no sequence is longer than the window")
    return WindowSet(**{k: np.concatenate(v) for k, v in parts.items()})


# =========================================================================================
# Test sets (1000-step segments with a variable warm-up)
# =========================================================================================

def warmup_lengths(H_padded: np.ndarray) -> np.ndarray:
    """Per-sequence warm-up = index of the first NaN (full length when there is none)."""
    nan = np.isnan(H_padded)
    return np.where(nan.any(axis=1), nan.argmax(axis=1), H_padded.shape[1]).astype(np.int64)


@dataclass
class TestSet:
    __test__ = False                    # not a pytest test class

    B: np.ndarray                       # (N, L) full excitation
    H_padded: np.ndarray                # (N, L) true warm-up, NaN afterwards
    T: np.ndarray                       # (N,)
    H_true: Optional[np.ndarray] = None  # (N, L) full measured H, for scoring
    loss: Optional[np.ndarray] = None    # (N,) full-excitation core loss, when provided
    name: str = ""

    def __post_init__(self):
        self.B = np.asarray(self.B, dtype=np.float64)
        self.H_padded = np.asarray(self.H_padded, dtype=np.float64)
        self.T = np.asarray(self.T, dtype=np.float64).reshape(-1)
        if self.H_true is not None:
            self.H_true = np.asarray(self.H_true, dtype=np.float64)
            if self.H_true.shape != self.B.shape:
                raise ValueError(f"H_true {self.H_true.shape} vs B {self.B.shape}")
            w = self.warmups
            prefix = np.arange(self.B.shape[1])[None, :] < w[:, None]
            bad = (prefix & ~np.isclose(self.H_padded, self.H_true, rtol=1e-6, atol=1e-6)).any(axis=1)
            # A one-step warm-up is a provided initial condition, not a measurement: Material C
            # (FEC007) gives H(1) = 0 A/m while the measured H(1) is about +/-0.2 A/m.
            if np.any(bad & (w > 1)):
                raise ValueError("warm-up prefix of the padded H differs from the true H; "
                                 "are the padded/true files from the same set?")
        if self.loss is not None:
            self.loss = np.asarray(self.loss, dtype=np.float64).reshape(-1)
        if not (self.B.shape == self.H_padded.shape and len(self.T) == len(self.B)):
            raise ValueError(f"shapes B{self.B.shape} H{self.H_padded.shape} T{self.T.shape}")

    @property
    def warmups(self) -> np.ndarray:
        return warmup_lengths(self.H_padded)

    def __len__(self) -> int:
        return len(self.B)


def load_test_h5(padded: str | Path, true: Optional[str | Path] = None, name: str = "") -> TestSet:
    with h5py.File(padded, "r") as f:
        B, H, T = f["B_seq"][:], f["H_seq"][:], f["T"][:]
        loss = f["Loss"][:] if "Loss" in f else None
    H_true = None
    if true is not None:
        with h5py.File(true, "r") as f:
            H_true = f["H_seq"][:]
            if "B_seq" in f and not np.allclose(f["B_seq"][:], B):
                raise ValueError(f"B_seq differs between {padded} and {true}")
            if loss is None and "Loss" in f:
                loss = f["Loss"][:]
    return TestSet(B, H, T, H_true, loss, name=name or Path(padded).name.split("_")[0])


def load_test_csv(root: str | Path, material: str) -> TestSet:
    """CSV test layout. Accepts the final-test names (``A_True_B_seq.csv``) and the pretest
    names with a ``_Testing`` infix (``3C90_Testing_True_B_seq.csv``)."""
    root = Path(root)

    def find(suffix: str) -> Optional[Path]:
        for infix in ("_", "_Testing_"):
            p = _find_file(root, f"{material}{infix}{suffix}")
            if p is not None:
                return p
        return None

    def need(suffix: str) -> Path:
        p = find(suffix)
        if p is None:
            raise FileNotFoundError(f"{material}_[Testing_]{suffix} not found under {root}")
        return p

    B = _read_matrix(need("True_B_seq.csv"))
    H = _read_matrix(need("Padded_H_seq.csv"))
    T = _read_matrix(need("True_T.csv"))
    p_true, p_loss = find("True_H_seq.csv"), find("True_Loss.csv")
    return TestSet(B, H, T,
                   _read_matrix(p_true) if p_true else None,
                   _read_matrix(p_loss) if p_loss else None, name=material)


def find_test_set(root: str | Path, material: str) -> TestSet:
    """Locate a material's test set under ``root``: an HDF5 padded/true pair (any case,
    e.g. ``3C90_Testing_padded.h5`` or ``N87_Testing_Padded.h5``) or the CSV layout."""
    root = Path(root)
    padded = _find_file(root, f"{material}_Testing_padded.h5")
    if padded is not None:
        return load_test_h5(padded, _find_file(root, f"{material}_Testing_true.h5"), name=material)
    return load_test_csv(root, material)


def make_test_segments(buckets: Sequence[RawBucket], seg_len: int = SEGMENT_LENGTH,
                       starts_per_seq: int = 1, warmups: Sequence[int] = DEFAULT_WARMUPS,
                       seed: int = 1, name: str = "") -> TestSet:
    """Build an official-style test set from held-out raw sequences: random 1000-step
    segments, each repeated once per warm-up length (outer loop over warm-ups, the
    ordering of ``Test/Test_Save_Template.m``)."""
    rng = np.random.default_rng(seed)
    Bs, Hs, Ts = [], [], []
    for b in buckets:
        lengths = b.valid_lengths()
        for r in range(len(b.T)):
            L = int(lengths[r])
            if L < seg_len:
                continue
            for s in rng.integers(0, L - seg_len + 1, size=starts_per_seq):
                Bs.append(b.B[r, s:s + seg_len])
                Hs.append(b.H[r, s:s + seg_len])
                Ts.append(b.T[r])
    if not Bs:
        raise ValueError(f"no sequence has {seg_len} valid steps")
    B, H, T = np.stack(Bs), np.stack(Hs), np.asarray(Ts)
    cols = np.arange(seg_len)[None, :]
    out_B, out_Hp, out_H, out_T = [], [], [], []
    for w in warmups:
        if not 0 < w < seg_len:
            raise ValueError(f"warm-up {w} outside (0, {seg_len})")
        out_B.append(B)
        out_H.append(H)
        out_Hp.append(np.where(cols < w, H, np.nan))
        out_T.append(T)
    return TestSet(np.concatenate(out_B), np.concatenate(out_Hp), np.concatenate(out_T),
                   np.concatenate(out_H), None, name=name)
