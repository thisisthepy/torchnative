"""exir's `_serialize/_serialize.py`, as torchnative vendors it (issue #423, R5-V).

`serialize_for_executorch` keeps upstream's signature and its program half,
transcribed: the PTE file is the program, its mutable data and the named data
destined for it. The other half, external weights written to `.ptd` files by the
flat-tensor serializer, is not vendored: a program that has any refuses by name.
`FlatTensorConfig` and `FlatTensorSerializer` stand in for the names
`exir/program/_program.py` constructs; the serializer refuses if it is used.
"""

from typing import Dict, Optional, Tuple

from torchnative._vendor.executorch.exir._serialize._cord import Cord
from torchnative._vendor.executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from torchnative._vendor.executorch.exir._serialize._program import PTEFile, serialize_pte_binary
from torchnative._vendor.executorch.exir._serialize.data_serializer import DataSerializer

_PTD = ("external weights (.ptd files, the flat-tensor serializer) are not part of "
        "torchnative's vendored exir (issue #423, R5-V)")


class FlatTensorConfig:
    def __init__(self, segment_alignment: int = 128, *args, **kwargs) -> None:
        self.segment_alignment = segment_alignment


class FlatTensorSerializer(DataSerializer):
    def __init__(self, config: Optional[FlatTensorConfig] = None) -> None:
        self.config = config

    def serialize(self, data):
        raise NotImplementedError("FlatTensorSerializer.serialize: " + _PTD)

    def deserialize(self, blob):
        raise NotImplementedError("FlatTensorSerializer.deserialize: " + _PTD)


def serialize_for_executorch(
    emitter_output,
    config,
    data_serializer: DataSerializer,
    named_data_store: Optional[NamedDataStoreOutput] = None,
) -> Tuple[Cord, Dict[str, Cord]]:
    """Serialize the output from Emitter into ExecuTorch artifacts; PTE and PTD files."""

    # Serialize PTE file.
    pte_named_data = None
    if (
        named_data_store is not None
        and len(named_data_store.buffers) > 0
        and len(named_data_store.pte_data) > 0
    ):
        # Create a separate NamedDataStoreOutput with only pte_data; exclude
        # external_data, which shouldn't be serialized with the PTE file.
        if len(named_data_store.external_data) == 0:
            pte_named_data = named_data_store
        else:
            pte_named_data = NamedDataStoreOutput(
                buffers=named_data_store.buffers,
                pte_data=named_data_store.pte_data,
                external_data={},
            )
    pte: Cord = serialize_pte_binary(
        pte_file=PTEFile(
            program=emitter_output.program,
            mutable_data=emitter_output.mutable_data,
            named_data=pte_named_data,
        ),
        extract_delegate_segments=config.extract_delegate_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
    )

    # Early exit if no external weights.
    if len(emitter_output.external_constant_map) == 0 and (
        named_data_store is None or len(named_data_store.external_data) == 0
    ):
        return pte, {}

    raise NotImplementedError("serialize_for_executorch: " + _PTD)
