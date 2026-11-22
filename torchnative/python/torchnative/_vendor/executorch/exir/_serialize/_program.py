"""exir's `_serialize/_program.py`, as torchnative vendors it (issue #423, R5-V).

`PTEFile` is upstream's dataclass, copied. `serialize_pte_binary` keeps upstream's
signature and returns upstream's `Cord`; the bytes come from
`torch._C._et_serialize_pte` (R5-S1), which equals upstream's serializer byte for
byte (tests/export/test_et_serialize.py). Reading a program back
(`deserialize_pte_binary`) is not vendored and refuses by name.
"""

from dataclasses import dataclass
from typing import List, Optional

from torchnative._vendor.executorch.exir._serialize._cord import Cord
from torchnative._vendor.executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from torchnative._vendor.executorch.exir.schema import Buffer, Program


@dataclass
class PTEFile:
    """
    Wraps together the data required to serialize into a PTE file.
    """

    program: Program
    # TODO(lfq): add constant data (currently restored in the program)
    # TODO(lfq): update this to List[bytes]
    mutable_data: Optional[List[Buffer]] = None
    named_data: Optional[NamedDataStoreOutput] = None


def serialize_pte_binary(
    pte_file: PTEFile,
    *,
    extract_delegate_segments: bool = False,
    segment_alignment: int = 128,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> Cord:
    from torchnative.export import et_serialize

    return Cord(
        et_serialize.serialize_pte_bytes(
            pte_file,
            extract_delegate_segments=extract_delegate_segments,
            segment_alignment=segment_alignment,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
    )


def deserialize_pte_binary(program_data):
    raise NotImplementedError(
        "deserialize_pte_binary: reading a program back is not part of torchnative's "
        "vendored exir (issue #423, R5-V)")
