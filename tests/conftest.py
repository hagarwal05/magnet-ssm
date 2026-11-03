import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REF = ROOT / "third_party" / "magnetchallenge-2"


def ref_path(rel: str) -> Path:
    p = REF / rel
    if not p.exists():
        pytest.skip(f"reference file {rel} missing; run scripts/fetch_reference.sh")
    return p
