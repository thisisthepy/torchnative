"""#91: every Metal argsort arm in the candle fork names the kernel of its own dtype.

sort is on MPS_HOST_READBACK_OPS, so no mps run reaches this code yet; the
source is checked instead. An i8 arm naming an i16 kernel reads 2-byte
elements from a 1-byte buffer.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SORT_RS = ROOT / "torchnative/rust/vendor/candle-core/src/sort.rs"


def test_argsort_kernel_matches_dtype():
    arms = re.findall(r'DType::(\w+)\s*=>\s*"asort_(?:asc|desc)_(\w+)"', SORT_RS.read_text())
    assert len(arms) >= 16, arms
    bad = [(d, k) for d, k in arms if d.lower() != k]
    assert not bad, bad


if __name__ == "__main__":
    test_argsort_kernel_matches_dtype()
    print("ok test_argsort_kernel_matches_dtype")
