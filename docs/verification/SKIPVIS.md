# SKIPVIS — 건너뛴 줄이 `SKIP` 도 `ok` 도 아니라 사라지던 결함

**결론부터.** 게이트는 매 실행마다 `SKIP=0` 을 찍었습니다. **거짓이었습니다.** 실측 한 회차에서
24 개 테스트가 건너뛰어졌는데도 `SKIP=0` 으로 나왔습니다. 원인은 `suite_ledger.py` 의 `tally()` 가
`^SKIP\s` 로 시작하는 줄만 세는데, 스위트 29 개의 건너뛰기 헬퍼는 전부

```
   (skipped: TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch)
```

처럼 **앞에 공백 세 칸을 두고 `SKIP` 접두어 없이** 찍고 `return` 했다는 데 있습니다. 그 줄은
`ok` 컬럼에도 `SKIP` 컬럼에도 잡히지 않습니다 — **사라집니다.** 그 가드 뒤에서 테스트를 `pass` 로
비워도 로그는 똑같이 보입니다.

## 고친 모양

`tests/_support/_skip.py` 가 공유 등록 지점입니다. 건너뛰는 지점은

```python
if fixture is None:
    _skip.skip("no vendored shim installed")
    return
```

로 기록하고, 각 스위트의 `_main()`(또는 그에 상응하는 `__main__` 러너)은 `_skip.run_tests(items,
suite=...)` 를 통해 **`ok` 와 `SKIP` 이 배타적으로** 찍히도록 합니다. Vulkan 스위트(`test_vulkan4.py`,
`test_shim.py`)는 이미 `vulkan_coverage.py` 가 같은 모양으로 이 문제를 먼저 풀어 두었으므로 그 기존
등록 함수(`vulkan_coverage.vulkan_skip`)를 그대로 재사용했습니다 — 이 문서와 `_skip.py` 는 그것을
중복 구현하지 않습니다.

`tests/gate/test_skipvis.py` 가 회귀를 막는 스캐너입니다: 모든 `test_*.py` 의 AST 를
훑어 `print(...(skipped...)...)` 바로 뒤에 `return` 이 오는 모양(등록 호출 없이)을 찾으면 실패합니다.
`test_vulkan4.py` 와 `test_coremlops.py` 는 이 문서를 쓰는 시점에 다른 회차가 편집 중이어서 스캔에서
명시적으로 제외했습니다 — `test_skipvis.py` 의 `_EXEMPT` 를 보십시오.

## 무엇이 여전히 못 보는가

- **새 철자.** `(skipped` 문자열도 안 쓰고 등록 함수도 안 부르는 새로운 건너뛰기 표현이 생기면
  스캐너도 이 카운트도 못 잡습니다. 능력 기준이 아니라 문자열 기준입니다.
- **즉시 `return` 하지 않는 건너뛰기.** 루프를 `continue` 하거나 더 아래로 흘러가는 모양은 인접성
  검사 밖입니다.
- **제외된 두 파일.** 위 예외 목록이 비워지기 전까지는 이 카운트도 그 두 파일의 회귀는 보지
  못합니다.

<!-- DOCWATCH: count skip_lines_visible ge 1 -->

위 마커는 `tests/devices/npu/test_intelnpu.py` 를 이 호스트에서 직접 실행해 `SKIP ` 로 시작하는
줄을 셉니다. 그 스위트는 벤더링된 `_C` 빌드 없이도 돌고, macOS 에는 Intel NPU/OpenVINO 런타임이 없으므로
몇 개는 항상 건너뛰어집니다 — 그래서 `ge 1` 이 어떤 호스트에서도 참이어야 합니다. `eq` 가 아니라 `ge`
를 쓴 이유는 AGENTS.md 의 규정 그대로: 나중에 건너뛰기 지점이 늘어나는 것(예: 새 환경 변수 가드
추가)은 이 마커가 막을 일이 아니고, 이 마커가 막아야 하는 것은 **줄어들어 0 이 되는 것**, 즉 등록
호출이 다시 조용한 `print` 로 되돌아가는 회귀입니다.
