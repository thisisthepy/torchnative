"""Test-time learning methods.

TTL contains TTA contains TTT; the nesting is not flattened into sibling
modules. Each method declares its own differentiation requirement rather than
living in a directory named for one, because normalization calibration sits on
both sides of that line -- recomputing statistics needs no backward pass,
updating affine parameters by a loss does. See DESIGN.md §3.
"""
