"""What the vendored exir needs from torchnative in place of its third-party imports.

Written by scripts/vendor/vendor_exir.py (issue #423, R5-V). Each name stands in
for one import of a package torchnative does not depend on, or for a step
torchnative does not run; the PATCHES list of the script says which import.
"""

import torch


def tabulate(rows, headers=(), tablefmt=None, **_kwargs):
    """`tabulate.tabulate`'s plain layout, for exir's delegation report text.

    Columns are left-aligned and separated by two spaces, with a dashed rule
    under the headers. The text is not byte-equal to tabulate's own formats;
    it only feeds a human-readable report."""
    table = [[str(c) for c in headers]] if headers else []
    table += [[str(c) for c in row] for row in rows]
    if not table:
        return ""
    width = [max(len(r[i]) if i < len(r) else 0 for r in table) for i in range(max(len(r) for r in table))]
    lines = ["  ".join(c.ljust(width[i]) for i, c in enumerate(r)).rstrip() for r in table]
    if headers:
        lines.insert(1, "  ".join("-" * w for w in width))
    return "\n".join(lines)


def _is_conv_node(n):
    """torchao 0.18.0 `torchao/quantization/pt2e/utils.py::_is_conv_node`, copied."""
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv3d.padding,
    ]


def _is_conv_transpose_node(n):
    """torchao 0.18.0 `torchao/quantization/pt2e/utils.py::_is_conv_transpose_node`, copied."""
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose2d.input,
    ]


class numpy_isclose_only:  # noqa: N801 - stands in for the module name `np`
    """`numpy.isclose` for two real scalars, numpy's formula and defaults:
    `abs(a - b) <= atol + rtol * abs(b)`, NaN never close (equal_nan=False),
    equal infinities close."""

    @staticmethod
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        a, b = float(a), float(b)
        if a != a or b != b:
            return equal_nan and a != a and b != b
        if a in (float("inf"), float("-inf")) or b in (float("inf"), float("-inf")):
            return a == b
        return abs(a - b) <= atol + rtol * abs(b)


def flatc_refused(*_args, **_kwargs):
    raise NotImplementedError(
        "flatc: torchnative never starts flatc; the XNNPACK payload is serialised by "
        "torch._C._et_serialize_xnnpack (issue #423, R5-S1)")


def serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes):
    """exir's `serialize_xnnpack_binary`, computed by torchnative's Rust serializer."""
    from torchnative.export import et_serialize

    return et_serialize.serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes)


def etrecord_refused():
    raise NotImplementedError(
        "ETRecord: executorch.devtools is not part of torchnative's vendored exir (issue #423, R5-V)")
