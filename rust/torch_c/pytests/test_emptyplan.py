""""CoreML told us nothing" is not "CoreML told us CPU", and must not be silent.

`compute_plan` asks `MLComputePlan` which unit runs each operation. Every
caller of it -- `_say_what_ran`, and `_compile_model`'s `to()`-time guard --
used to begin by dropping the operations CoreML returned *no* usage for and
then returning early when nothing was left. So a model whose plan CoreML
declines to produce got **no warning at all**: the same silence that a FULL
offload deliberately gets, which is the exact outcome docs/graph/NPU2.md §1
exists to prevent.

That this is reachable is measured, not hypothetical. docs/graph/NPU2.md §9.1
records a single-`relu` and a single-`gelu` float16 program whose compiled
`.mlmodelc` gets `None` from
`get_compute_device_usage_for_mlprogram_operation` for **every** operation in
it -- including the `ios16.cast`s, which do get a usage in a `linear` program
compiled seconds earlier in the same process. Renaming the operation inside the
compiled bundle, changing nothing else, restores the usage; renaming it back
removes it again. So the artefact's identity, not the program it encodes, is
what decides -- and a library that reads that as "nothing to say" reports a
CPU-bound model as an offloaded one.

The rule this file fixes in place:

    every computing operation is on the report, and one whose unit CoreML
    would not name is on it as `"unknown"` rather than absent;
    unknown warns; unknown is not reported as the precision's fault;
    and a FULL offload is still silent, because a warning nobody can ignore
    is one nobody reads.

Pure: these are `_say_what_ran` and `computes` over rows built here, so they
run on a machine with no CoreML at all. The end-to-end half -- that a real
relu program reaches this path -- is in test_coremlops.py.
"""

import os
import sys
import warnings

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "torchnative", "src", "main"))

from torchnative.export import coreml as C


def _row(op, preferred, supported):
    return {"op": op, "preferred": preferred, "supported": list(supported)}


_KNOWN_NE = _row("ios16.conv", "NeuralEngine", ["CPU", "GPU", "NeuralEngine"])
_KNOWN_CPU = _row("ios16.relu", "CPU", ["CPU", "GPU", "NeuralEngine"])
_UNKNOWN = _row("ios16.relu", "unknown", [])
_CAST = _row("ios16.cast", "CPU", ["CPU", "GPU"])


def _said(rows, precision="float16", report=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_what_ran(report if report is not None else {},
                        precision, [1, 128, 32, 32], rows)
    return [str(w.message) for w in caught]


def test_a_plan_with_no_rows_at_all_warns_that_what_ran_is_unknown():
    """The failure this file is named for: zero rows used to return silently.

    `_say_what_ran` began `if not compute: return`, so a program CoreML
    produced no plan for was indistinguishable from a program that reached the
    Neural Engine completely. Those are opposite facts.
    """
    said = _said([])
    assert len(said) == 1, said
    assert "no compute plan at all" in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_an_unknown_operation_warns_even_when_other_operations_are_known():
    """A partially unknown plan is unknown, not a partial offload.

    If one operation's unit is unnamed, "everything reached the unit" cannot
    be said about this program, so the FULL-offload silence is not available
    to it -- even if every operation CoreML *did* name is on the unit.
    """
    said = _said([_CAST, _KNOWN_NE, _UNKNOWN])
    assert len(said) == 1, said
    assert "no usable compute plan" in said[0], said[0]
    assert "ios16.relu" in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_a_plan_of_nothing_but_boundary_casts_is_unknown_and_not_silence():
    """`computes()` drops the boundary casts, and dropping them all is not none.

    A program whose only operations are the float16 interface casts did no
    work CoreML will speak about, so "which unit ran this" is unanswered --
    the same sentence as an empty plan, reached a different way.
    """
    said = _said([_CAST, _CAST])
    assert len(said) == 1, said
    assert "unknown" in said[0], said[0]


def test_an_unknown_plan_is_not_blamed_on_the_precision():
    """The wrong sentence is worse than none: it sends the caller to float16.

    `_UNREACHABLE_PRECISION` says the Neural Engine is *not in the supported
    column at this precision*, which is a claim about what MLComputePlan
    returned. An unknown plan returned nothing, so that claim has no evidence
    behind it -- and a caller who acted on it would switch precision and get
    the same silence.
    """
    said = _said([_UNKNOWN])
    assert len(said) == 1, said
    assert "supported column" not in said[0], said[0]
    assert "precision=" not in said[0], said[0]


def test_a_full_offload_is_still_silent():
    """The constraint the existing design carries, unbroken.

    Every computing operation known and preferred on the unit is the one case
    with positive evidence that the offload worked, and it says nothing so
    that a warning stays worth reading. Widening "unknown warns" into
    "everything warns" would destroy that, so it is asserted here rather than
    assumed.
    """
    assert _said([_CAST, _KNOWN_NE, _CAST]) == []


def test_a_known_cpu_preferred_plan_still_gets_its_own_sentence():
    """Unknown must not swallow the case that already worked.

    CPU-preferred-but-NE-supported is a measured fact about the model's size
    (docs/graph/NPU2.md §2.1) and has its own wording. If the unknown branch
    caught this too, the distinction the whole file is about would be lost in
    the other direction.
    """
    said = _said([_CAST, _KNOWN_CPU, _CAST])
    assert len(said) == 1, said
    assert "is NOT what it preferred" in said[0], said[0]
    assert "unknown" not in said[0], said[0]


def test_the_unknown_sentence_is_said_once_per_model_and_not_per_shape():
    """Said through the report's own bookkeeping, like the other two.

    A leaf compiles per shape, so an unknown plan would otherwise be said at
    every batch a caller runs, and a warning repeated on every forward is one
    that gets filtered out.
    """
    report = {}
    first = _said([_UNKNOWN], report=report)
    second = _said([_UNKNOWN], report=report)
    assert len(first) == 1, first
    assert second == [], second


def test_unknown_rows_survive_the_boundary_cast_filter():
    """`computes()` drops casts by name, and an unknown row is not a cast.

    The filter is what stands between "this operation's unit is unnamed" and
    an empty list, so it is asserted directly: an unknown `relu` is still
    there afterwards.
    """
    kept = C.computes([_CAST, _UNKNOWN, _CAST])
    assert kept == [_UNKNOWN], kept


# -- a refusal is not a silence, and not an answer ---------------------------
#
# The two shapes this section separates were measured on 2026-09-17 in this
# worktree, on a quiet machine (load 1.4-2.6, no other agent running):
#
#   silence  MLComputePlan loads, lists the operations, and returns `None` for
#            every one of them. Under `ComputeUnit.ALL` the third float32
#            program compiled in a process went silent 12 times out of 12;
#            under `CPU_AND_NE` the same three programs in the same order
#            answered. Neither predecessor alone triggers it -- it takes the
#            accumulation -- which is docs/graph/NPU2.md 9.1's finding that
#            the compiled artefact's identity, not the program it encodes,
#            decides.
#
#   refusal  `MLComputePlan.load_from_path` raises outright:
#            `Failed to construct compute plan, internal failure.` Seen on the
#            211-leaf naive path 1 run in 4; the same leaf compiled alone
#            answered 12 times out of 12 under both settings.
#
# They are different conditions with one consequence, and the consequence is
# the trap: read either as "the answer was CPU" and a CPU-bound model is
# reported as an offloaded one. That conflation is how "the ANE rejects relu"
# came to be believed when the truth was "CoreML did not answer".


def _refused(report=None, detail="internal failure"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_refused(report if report is not None else {}, detail)
    return [str(w.message) for w in caught]


def test_a_refused_compute_plan_says_so_in_its_own_sentence():
    """`load_from_path` raising is not the same fact as an empty plan.

    An empty plan is CoreML answering "no device for any operation"; a refusal
    is CoreML declining to build the plan at all. The library has a sentence
    for the first (`_UNKNOWN_PLAN`) and this asserts it has a *different* one
    for the second, because sending a reader to 9.1's artefact-identity
    measurement for an error that never got as far as an artefact is sending
    them to the wrong place.
    """
    said = _refused()
    assert len(said) == 1, said
    assert "refused" in said[0].lower(), said[0]
    assert "internal failure" in said[0], said[0]


def test_a_refused_plan_is_never_reported_as_cpu_or_as_offloaded():
    """The conflation this whole file exists to prevent, at the refusal.

    "CoreML refused to answer" and "the answer was CPU" are different facts.
    A sentence that says the model ran on the CPU, or that it reached the
    unit, is making a claim about evidence that does not exist.
    """
    said = _refused()[0]
    assert "ran on the CPU" not in said, said
    assert "reached the Neural Engine" not in said, said
    # It must say the question is unanswered, not answer it.
    assert "unknown" in said.lower() or "not know" in said.lower(), said


def test_a_refused_plan_does_not_claim_the_precision_is_at_fault():
    """`_UNREACHABLE_PRECISION` blames a setting the caller can change.

    Saying it for a refusal sends the caller to change `precision=` for an
    error that has nothing to do with precision -- the same misdirection
    `test_an_unknown_plan_is_not_blamed_on_the_precision` fixes one shape up.
    """
    said = _refused()[0]
    assert "no compute_units setting can reach the unit" not in said, said
    assert "precision=" not in said, said


def test_the_refusal_sentence_is_said_once_per_model_and_not_per_shape():
    """A leaf compiles per shape, and 211 of them refuse together.

    The naive path's failure raised on `lm_head` while 210 other leaves were
    compiling; one sentence per shape per leaf is a wall of text a caller
    filters out, which is the same reason `_say_unknown` is once-per-model.
    """
    report = {}
    first = _refused(report=report)
    second = _refused(report=report)
    assert len(first) == 1, first
    assert second == [], second


def test_a_refusal_and_a_silence_do_not_share_one_sentence():
    """Told apart by the text, not only by the code path that produced them.

    If both produced the same string a reader could not tell which happened,
    and the two send them to different places: a silence to 9.1, a refusal to
    a CoreML internal error that no setting of ours provoked.
    """
    refusal = _refused()[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_unknown({}, [], False)
    silence = str(caught[0].message)
    assert refusal != silence, refusal
    assert "internal failure" not in silence, silence


def _at_to(plans, precision="float16", report=None):
    report = report if report is not None else {}
    report["plans"] = plans
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_at_to_time(report, precision)
    return [str(w.message) for w in caught]


def test_the_to_time_guard_sees_a_plan_that_came_back_with_nothing():
    """The second place the silence lived, and the one a test could not reach.

    `_compile_model`'s tail read `probe_rows` -- the *flattened, cast-filtered*
    rows of every plan. A plan CoreML returned nothing for contributes no rows
    to that list, so `if probe_rows and ...` skipped it and `to()` returned a
    model with a compiled leaf and not one word about it. It is checked over
    the plan entries instead, which is why this test hands it a plan whose
    `rows` is empty rather than handing it no plan.
    """
    said = _at_to([{"leaf": "linear", "rows": []}])
    assert len(said) == 1, said
    assert "no compute plan at all" in said[0], said[0]


def test_the_to_time_guard_does_not_blame_the_precision_for_an_unnamed_op():
    """`_UNREACHABLE_PRECISION` is a claim about a column CoreML filled in.

    An unknown row has an empty `supported` for the opposite reason -- CoreML
    filled in nothing -- and reporting it as "the unit is not in the supported
    column at this precision" is a false sentence with an action attached to
    it. The unknown branch is checked first so that sentence is unreachable
    from this state.
    """
    said = _at_to([{"leaf": "linear", "rows": [_CAST, _UNKNOWN]}])
    assert len(said) == 1, said
    assert "supported column" not in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_the_to_time_guard_still_says_the_precision_sentence_when_it_is_true():
    """Known rows, unit absent from a column CoreML did fill in: unchanged.

    This is the case that already worked -- a float32 program -- and the
    unknown branch must not have swallowed it.
    """
    known_no_ne = _row("ios16.linear", "CPU", ["CPU", "GPU"])
    said = _at_to([{"leaf": "linear", "rows": [known_no_ne]}], "float32")
    assert len(said) == 1, said
    assert "NOT in CoreML's supported column" in said[0], said[0]
    assert "float32" in said[0], said[0]


def test_the_to_time_guard_is_silent_for_a_leaf_that_reached_the_unit():
    """And silent when there is no plan at all, which is a deferred leaf.

    A conv is deferred because its shape is unknown at `to()` time, so it has
    no plan yet and `to()` has nothing to say; the sentence lands at the first
    forward instead. That is different from a plan that came back empty, and
    the two must not be merged -- which is what makes the `report["plans"]`
    test in the guard load-bearing rather than decorative.
    """
    assert _at_to([{"leaf": "linear", "rows": [_CAST, _KNOWN_NE, _CAST]}]) == []
    assert _at_to([]) == []


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
