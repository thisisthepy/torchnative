"""exir's `_serialize` package, as torchnative vendors it (issue #423, R5-V).

Upstream's `__init__.py` re-exports three names from `_program.py`; this one
does the same from torchnative's `_program.py`, whose serializer is the Rust one.
"""

from torchnative._vendor.executorch.exir._serialize._program import (
    deserialize_pte_binary as _deserialize_pte_binary,
    PTEFile as _PTEFile,
    serialize_pte_binary as _serialize_pte_binary,
)

# Internal APIs that should not be used outside of exir.
__all__ = [
    "_deserialize_pte_binary",
    "_serialize_pte_binary",
    "_PTEFile",
]
