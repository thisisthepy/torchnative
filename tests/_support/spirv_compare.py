"""Recompile each shader and compare bytes with the checked-in SPIR-V (issue #46).

The old guard compared file mtimes, which git does not store: a fresh checkout
can be red with correct bytes and green with wrong ones. This compiles every
`.comp` with the command `shaders/compile.sh` uses and compares the output with
the checked-in `.spv`. Without `glslc` on PATH it returns a reason to skip by
name; it never reads an mtime.

The command is `glslc -fshader-stage=compute -O <src> -o <out>`. No `-g`, so no
source text is embedded and a comment-only edit leaves the bytes unchanged.

Header word 2 (bytes 8..12) is the generator id, and it carries the glslang
version: Ubuntu 24.04's shaderc 2023.8 writes 11 where the NDK 27 glslc that
produced the checked-in files wrote 10. Measured over all 28 shaders: that word
is the only difference (`spirv-dis` output identical below the header), so it
is masked and every other byte is compared.
"""

import pathlib
import shutil
import subprocess
import tempfile


def repo_root():
    return next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())


def shaders_dir():
    return repo_root() / "torchnative" / "rust" / "torch_c" / "shaders"


def _without_generator(words):
    return words[:8] + words[12:]


def compare(shaders):
    """Return None if every `.spv` equals a fresh compile, else a skip reason.

    Raises AssertionError on a missing, malformed or differing `.spv`.
    """
    sources = sorted(shaders.glob("*.comp"))
    assert sources, f"no .comp under {shaders}"
    for source in sources:
        words = source.with_suffix(".spv")
        assert words.is_file(), f"{words.name} is missing -- run shaders/compile.sh"
        # SPIR-V's magic number, little-endian.
        assert words.read_bytes()[:4] == b"\x03\x02\x23\x07", words.name
    glslc = shutil.which("glslc")
    if glslc is None:
        return "glslc is not on PATH, so the checked-in SPIR-V was not recompiled"
    scratch = repo_root() / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as out_dir:
        for source in sources:
            out = pathlib.Path(out_dir) / source.with_suffix(".spv").name
            r = subprocess.run(
                [glslc, "-fshader-stage=compute", "-O", str(source), "-o", str(out)],
                capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, f"glslc failed on {source.name}: {r.stderr}"
            assert _without_generator(out.read_bytes()) == _without_generator(
                    source.with_suffix(".spv").read_bytes()), (
                f"{source.with_suffix('.spv').name} differs from a fresh compile of "
                f"{source.name}: run shaders/compile.sh")
    return None
