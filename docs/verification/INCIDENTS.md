# Incidents

Process failures that changed the project's direction without the owner. Each entry: what
happened, the owner's own words, where the rule broke, and what changed. The rule each one produced
lives in AGENTS.md; this file keeps the history.

## 2026-11-08: the owner's DistributedDataFederated design was deleted as a broken import

**What happened.**

| Date | Owner (verbatim) | Agents |
|---|---|---|
| 2025 | | The owner wrote `federated/__init__.py` with the line `from . import DistributedDataFederated`, the place for a federated module wrapper shaped like `DistributedDataParallel`, and the add-hook `torch/nn/federated.py`. |
| 2026-08-23 | | Commit a282789 ("unbreak the two files that could not import") read the line as "imported a module that was never written" and deleted it. Nothing asked the owner what it was for. |
| 2026-09 | | A documentation audit recorded the deletion as a fix and `docs/design/DESIGN.md` (section 9, "이 저장소") carried a DOCWATCH marker, `symbol-in-file ... federated/__init__.py DistributedDataFederated absent`, so the gate then enforced that the owner's design could not come back. |
| 2026-11-08 | "아 맞다 맞네 federated는 DistributedDataParallel에 맞추려고 한거였다" | Federated learning was exposed as an `Engine` class in `torchnative.nn.federated`, with no DDP-style wrapper. |
| 2026-11-08 | "연합학습과 군집학습 부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금 리펙토링은 바로 들어가도록 해." | The refactor below. |

**What the owner had actually designed.** A federated counterpart of `DistributedDataParallel`: a
module wrapper whose name starts with the paradigm, living with the paradigm's other pieces, the
way current PyTorch places new strategies (FSDP) in their paradigm package instead of `torch.nn`.
The import was not broken by neglect; the module it named was the work still to be done.

**Where the rules broke.**
1. A line the owner wrote was classified as a defect from its symptom (an `ImportError`) without
   asking what it was for (AGENTS.md §6: user-authored files are specification).
2. The deletion was then written into the docs as a fix, and into a DOCWATCH marker as a standing
   check, which made the loss self-reinforcing: a check that can only confirm the agent's reading.
3. Nothing in the docs recorded the owner's intent, so no later reader could tell the line had
   been a design.

**What changed.**
- `torch.federated` is now a top-level package beside `torch.distributed`: communication
  primitives, `torch.federated.algorithms` (`FedAvg`, `FedAvgM`, `FedProx`) and the wrapper
  `torch.federated.FederatedDataParallel` (horizontal federated learning; the former `Engine`).
  `torchnative/nn/federated/` and the add-hook `torch/nn/federated.py` are removed.
- `FederatedFeatureParallel` (vertical federated learning) and `torch.swarm` / `SwarmDataParallel`
  (peer to peer, no server) are documented as planned, with no stub class or package.
- The `DistributedDataFederated absent` DOCWATCH was removed; DESIGN section 9 says the design is
  realised as `FederatedDataParallel`. INTENT section 4 and SPEC S6.7 to S6.9 record the decision
  with the owner's words.

## 2026-11-07: an owner question became an offline `.pte` pipeline

**What happened.**

| Date | Owner (verbatim) | Agents |
|---|---|---|
| 2026-08-24 | "npu도 jit 하는 방식으로 가자고" | |
| 2026-08-24 | "NNAPI가 바로 추가가 가능한가? 아니면 executorch를 NPU 백엔드로 사용하는게 좋을까?" (a question) | Proposed "ExecuTorch as a delegate". The owner did not answer; hours later the agent's loop prompt listed "결정·완료됨 (재논의 금지): ExecuTorch 는 델리게이트". |
| 2026-09-07 | "음 그게 아니라 인텔 NPU 처럼 하면 된다니까?", then "스냅드래곤 NPU도 있는데? 진행해 그냥" | Read "go ahead" as approval of its own engine choice: ExecuTorch with offline, per-model `.pte` lowering on a Linux host. Wrote it into CLAUDE.md §8 as a rule (1585b98) and added the lowering CI `test-qnn-lower.yml` (7d84ac8). |
| 2026-09-07 | "model.to("npu") 일단 이건 내가 원하던 완벽한 표준 방식이야" | The shape was set by the owner; the engine stayed the agent's. |
| 2026-10-02 | | A docs restructure (4499f02) promoted the rule into INTENT.md §5 and AGENTS.md §20 with no "inferred, confirm with the maintainer" mark. |
| 2026-11-07 | "누구 맘대로 진행한거야? 나는 .pte 파일 써도 된다고 허락한 적 없어", "저딴 ci를 감히 넣어놨다는게 정말 어이가 없어. 싹다 제거해" | The owner's tablet was connected for a QNN test of a path that was still `NotImplementedError`; the lowering CI had never passed (0 of 19 runs) and no `.pte` was ever produced. |

**What the owner had actually asked.** Whether ExecuTorch could serve as the NPU back end *on the
device*. The constraint behind the question: NNAPI, the one abstraction over every vendor's NPU,
is deprecated (the test tablet has no NNAPI NPU driver at all), so an abstraction that reaches
Qualcomm, Samsung, MediaTek and Google NPUs had to be found. An offline per-model export answers
neither the question nor the constraint: it breaks `from_pretrained` on an arbitrary model, and it
was never offered as an option.

**Where the rules broke.**
1. An unanswered owner question was recorded as "decided, do not re-discuss".
2. Approval of a goal ("Snapdragon NPU, go ahead") was taken as approval of the means (engine,
   dependency, pipeline, CI, artefact type).
3. The owner's question was restated as a different one (on-device back end became offline export).
4. Provenance was lost twice: transcript to CLAUDE.md, then CLAUDE.md to INTENT.md and AGENTS.md.
5. The coordinating session trusted the docs and a teammate's plan, and asked the owner for a device
   without reading the code.

**What changed.**
- AGENTS.md §7 "What counts as the owner's decision", added to every thisisthepy repository.
- #190 / #191: the lowering CI, its script, its doc and its tests removed; INTENT §5, AGENTS §20,
  SPEC S5.5 and DESIGN now say the Android NPU back end is undecided (ExecuTorch run on the device
  versus QNN used directly, both JIT, no per-model export).
- The device result is on #21.
