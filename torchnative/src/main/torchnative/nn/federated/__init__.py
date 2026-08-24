"""Federated learning.

A layer above the single-device adaptation methods, not a sibling of them: the
local step is the same mechanism, with aggregation, transport and privacy on
top. Its dependencies stay behind the ``federated`` extra so that using
adaptation alone does not pull in the aggregation stack. See DESIGN.md §3.
"""
