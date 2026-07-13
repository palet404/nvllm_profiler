# 실험 결과 설명 문서

nano-vLLM(Qwen3-0.6B, RTX 3070 Ti 8GB)의 세 가지 최적화 — Prefix Caching, Continuous
Batching, CUDA Graph — 를 실제 GPU 추론으로 실측한 기록이다. 시뮬레이션·더미 수치는
전혀 없으며, 모든 숫자는 `results/*.csv`의 원시 로그에서 그대로 뽑았다. 재현 커맨드는
각 절 끝에 적어뒀다.

## 목차

1. [방법론 — 왜 이렇게 쟀는가](#1-방법론)
2. [실험 A — Prefix Caching의 TTFT 효과](#2-실험-a--prefix-caching의-ttft-효과)
3. [실험 B — CUDA Graph의 효과](#3-실험-b--cuda-graph의-효과)
4. [실험 C — Continuous Batching의 효과](#4-실험-c--continuous-batching의-효과)
5. [실험 D — nano-vLLM 전체 스택 vs HF Transformers baseline](#5-실험-d--nano-vllm-전체-스택-vs-hf-transformers-baseline)
6. [실험 E — 디코드 길이에 따른 최적화 기여도 변화](#6-실험-e--디코드-길이에-따른-최적화-기여도-변화)
7. [종합 결론](#7-종합-결론)

---

## 1. 방법론

### 1.1 왜 두 종류의 워크로드로 나눴는가

같은 워크로드(요청들이 동시에 몰려 들어오는 상황)로 모든 것을 재면 결과가 서로
오염된다. 초기 실험에서 실제로 이 문제를 겪었다: prefix caching을 껐는데도 TTFT가
거의 그대로였는데, 원인은 캐시 미스인 첫 요청이 오래 걸리는 동안 나머지 요청이 전부
도착해 버려서 스케줄러가 그것들을 한 스텝에 배치로 묶어 처리했기 때문이었다 — 측정된
건 "캐시 없음의 비용"이 아니라 "배치로 묶였을 때의 GPU 병렬성 이득"이었다. 그래서
실험을 두 갈래로 분리했다.

- **TTFT 순수 비교(sequential)**: 요청을 하나씩, 이전 요청이 완전히 끝난 뒤에만
  다음 요청을 admit한다(`continuous_batching=False`). 배치 구성이라는 변수를 없애서
  요청 1개의 TTFT/latency만 깨끗하게 비교한다.
- **처리량 비교(concurrent)**: 요청을 동시에(또는 짧은 간격으로) 도착시켜 Continuous
  Batching이 실제로 처리량을 얼마나 올리는지 본다.

### 1.2 GPU 워밍업이 왜 필수였는가

nano-vLLM은 엔진 생성 시 **decode** 배치 크기별 CUDA Graph를 미리 캡처하지만,
**prefill**(`flash_attn_varlen_func`)은 실제 요청이 들어와야 처음 호출되어 Triton
JIT/cuBLAS 알고리즘 탐색 같은 1회성 비용을 그 자리에서 치른다. 워밍업 없이 측정하면
"첫 요청만 압도적으로 느리고 나머지는 프롬프트 길이·캐시 여부와 무관하게 전부 빠른"
현상이 나타나 캐싱 효과를 완전히 가려버린다. 그래서 모든 실험 앞에
`engine.warmup()`(실제 워크로드와 무관한 더미 프롬프트로 커널을 1회 실행)을 넣고,
그 뒤부터 시간을 잰다. `engine/nanovllm_engine.py`의 `NanoVLLMProfilerEngine.warmup()`.

### 1.3 실측치인지 확인하는 법

`prefix_cache_hit`/`cached_tokens`는 nano-vLLM의 실제 `BlockManager`가 리턴한 값을
그대로 스냅샷한 것이고(직접 계산·추정 없음), `ttft_ms`는 `Scheduler.schedule() →
ModelRunner.run() → Scheduler.postprocess()`를 직접 호출해 실측한 벽시계 시간이다.
`gpu_util_pct`/`mem_util_pct`는 `nvidia-smi`를 `subprocess`로 실제 호출해 파싱한
값이다. 자세한 구현은 `engine/nanovllm_engine.py`, `engine/gpu_metrics.py` 참고.

---

## 2. 실험 A — Prefix Caching의 TTFT 효과

**질문**: 공통 프리픽스가 캐시에 있으면(prefix cache hit) 없을 때보다 TTFT가 얼마나
빨라지는가? 그 효과는 프리픽스 길이에 따라 어떻게 변하는가?

### 2.1 짧은 프리픽스 (SYSTEM_PREFIX, 370토큰 → 1블록·256토큰 캐시)

| | 캐시 히트 | 캐시 미스 | 배수 |
|---|---|---|---|
| 평균 TTFT | 25.30 ms | 25.84 ms | **~1.0x (사실상 없음)** |

`results/comparison_ttft_summary.csv`의 `nanovllm_full_ttft` vs `nanovllm_no_prefix_cache_ttft` 행.

캐시 히트해도 새로 계산할 토큰이 179~194개(미스는 435~450개)로, 겨우 250토큰 차이다.
Qwen3-0.6B 같은 작은 모델은 이 정도 차이가 만드는 실제 GPU 연산 시간이 수 ms
수준이라, 스케줄러/커널 실행의 고정 오버헤드(~20ms대)에 완전히 묻혀버린다.

### 2.2 긴 프리픽스 (LONG_SYSTEM_PREFIX, 1015토큰 → 4블록·1024토큰 캐시)

few-shot 예시를 7개로 늘려 캐시 가능한 블록 수를 1개→4개로 키운 스트레스 테스트.

| | 캐시 히트 (1024토큰 캐시) | 캐시 미스 (전량 재계산, ~1090토큰) | 배수 |
|---|---|---|---|
| 평균 TTFT | 26.82 ms | 36.18 ms | **1.35~1.5x** |

`results/long_full_ttft.csv`(히트), `results/long_no_cache_ttft.csv`(미스, 전부 강제 미스).
두 실험 모두 캐시 미스 쪽 prefill 토큰 수(~1090개)는 거의 동일하게 유지한 채 캐시로
재사용되는 토큰만 256→1024로 늘렸다.

### 2.3 해석

캐시가 절감하는 것은 **prefill 연산량**이지 고정 오버헤드가 아니므로, 캐시된
토큰의 절대량이 커질수록 고정 오버헤드를 뚫고 효과가 드러난다. 이 프로젝트의 실제
production 시스템 프롬프트(370토큰)로는 효과가 미미하지만, RAG/few-shot이 많은
프롬프트(수천 토큰)라면 훨씬 커질 것으로 추정된다 — 실제로 256→1024토큰 구간에서
이미 뚜렷한 증가 추세(1.0x → 1.5x)가 확인됐다.

**재현**:
```bash
cd nvllm_profiler
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --sequential --tag full_ttft --save-csv
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --sequential --disable-prefix-cache --tag no_cache_ttft --save-csv
# 긴 프리픽스로 재검증하려면 --long-prefix 플래그 추가
```

---

## 3. 실험 B — CUDA Graph의 효과

**질문**: CUDA Graph(decode 스텝의 커널 실행을 그래프로 캡처해 재생, Python/커널
실행 오버헤드 제거)를 끄면(`enforce_eager=True`) 얼마나 느려지는가?

| | CUDA Graph ON | CUDA Graph OFF (eager) | 배수 |
|---|---|---|---|
| 평균 TTFT | 25.84 ms | 26.00 ms | 1.01x (거의 없음 — prefill엔 영향 안 줌) |
| 평균 Latency | 109.98 ms | 509.01 ms | **4.63x** |
| 집계 TPS | 217.0 | 47.1 | **4.60x** |

`results/comparison_ttft_summary.csv`의 `nanovllm_full_ttft` vs `nanovllm_no_cuda_graph_ttft` 행.

### 해석

CUDA Graph는 **decode**에만 적용되는 최적화라(prefill은 가변 길이라 그래프로 캡처가
안 됨), TTFT(=prefill 시간)에는 거의 영향이 없고 latency(=대부분 decode 시간)에는
4.6배라는 큰 차이를 만든다. 세 최적화 중 가장 견고하고 확실한 효과다 — 워밍업
유무·프리픽스 길이와 무관하게 항상 재현됐다.

**재현**:
```bash
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --sequential --tag graph_on --save-csv
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --sequential --enforce-eager --tag graph_off --save-csv
```

---

## 4. 실험 C — Continuous Batching의 효과

**질문**: 요청이 순차적으로(한 번에 하나씩) 오는 것과 동시에(겹쳐서) 오는 것 중,
동시 도착일 때 스케줄러가 빈 슬롯을 즉시 채우는 Continuous Batching 덕분에 집계
처리량이 얼마나 오르는가?

| | 순차 처리 (Continuous Batching 사실상 미사용) | 동시 처리 (Continuous Batching) | 배수 |
|---|---|---|---|
| 집계 TPS | 222.3 | 370.2 | **1.67x** |
| 평균 Latency | 107.3 ms | 328.9 ms | (참고용 — 아래 주의사항 참고) |

`results/comparison_throughput_summary.csv`의 `nanovllm_no_continuous_batching_throughput`
vs `nanovllm_full_throughput` 행.

### 주의: 동시 도착 시 TTFT/Latency는 "느려 보일" 수 있다

동시 도착 실험에서 평균 latency가 순차 처리보다 오히려 높게 나오는데(107→329ms),
이건 성능 저하가 아니라 **큐잉 효과**다. 여러 요청이 한 스텝에 배치로 묶이면, 나중에
합류한 요청의 "TTFT"(도착~첫 토큰)에는 앞서 admit된 요청들의 처리를 기다린 시간이
포함된다. 개별 요청 입장에선 조금 더 기다리지만, **시스템 전체로는 훨씬 많은 요청을
같은 시간에 처리**하는 것 — 그래서 집계 처리량(aggregate TPS)이 진짜 지표다.

**재현**:
```bash
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --sequential --tag sequential --save-csv
python scripts/profile_run.py --num-requests 8 --max-output-tokens 24 --tag concurrent --save-csv
```

---

## 5. 실험 D — nano-vLLM 전체 스택 vs HF Transformers baseline

**질문**: PagedAttention·Prefix Caching·Continuous Batching·CUDA Graph를 전부 갖춘
nano-vLLM과, `AutoModelForCausalLM.generate()`(non-paged KV 캐시, 좌측 패딩 정적 배치,
eager attention, prefix caching 없음)로 구현한 HF Transformers baseline을 비교.
baseline은 `Bench_server/baseline.py`와 동일한 방법론을 이 프로젝트의 워크로드(공통
prefix 이메일)에 맞춰 재구성했다 (`engine/transformers_baseline_engine.py`).

### TTFT 비교 (순차 조건 — baseline은 요청 1건, nano-vLLM도 순차 1건씩)

| | nano-vLLM (순차) | Transformers baseline (요청 1건) | 배수 |
|---|---|---|---|
| TTFT | 25.84 ms | 41.54 ms | **1.61x** |
| Latency | 109.98 ms | 626.09 ms | **5.69x** |

### 처리량 비교 (동시 조건 — nano-vLLM 8건 동시 도착 vs baseline 8건 정적 배치)

| | nano-vLLM (동시) | Transformers baseline (정적 배치 8건) | 배수 |
|---|---|---|---|
| 집계 TPS | 370.2 | 245.5 | **1.51x** |
| Latency | 328.9 ms | 782.2 ms | **2.38x** |

`results/comparison_ttft_summary.csv`, `results/comparison_throughput_summary.csv`의
`baseline_transformers_*` 행.

### 해석

- baseline은 **prefix caching이 아예 없어서** 같은 시스템 프롬프트라도 매번 처음부터
  전량 재계산한다 (`cached_tokens=0` 고정).
- baseline은 **정적 배치**라 모든 요청을 한 번에 padded tensor로 묶어 `generate()`를
  1번 호출한다. 그래서 배치 내 모든 요청이 **동일한 TTFT/latency**를 갖는다 — 가장
  늦게 끝날 시퀀스가 끝날 때까지 먼저 끝난 시퀀스도 계속 연산에 참여한다
  (padding token 처리 낭비).
- baseline은 **CUDA Graph도 없다** (eager PyTorch 그대로).

세 가지 최적화가 전부 빠진 baseline과 비교하니, 개별 ablation 배수(캐싱 1.0~1.5x ×
CUDA Graph 4.6x × 배칭 1.67x)가 어느 정도 누적된 형태로 latency 2.4~5.7배 차이가
남는다.

**재현**:
```bash
python scripts/baseline_run.py --num-requests 1 --tag baseline_ttft --save-csv
python scripts/baseline_run.py --num-requests 8 --tag baseline_throughput --save-csv
# 또는 5개 모드를 한 번에: python scripts/run_comparison_suite.py --num-requests 8 --max-output-tokens 24
```

---

## 6. 실험 E — 디코드 길이에 따른 최적화 기여도 변화

**질문**: `max_output_tokens`(생성 토큰 수)를 늘리면 두 최적화의 상대적 중요도가
어떻게 바뀌는가? (TTFT는 prefill 시간이라 출력 길이와 무관해야 하고, CUDA Graph는
decode 스텝마다 절약되므로 출력이 길수록 유리해야 한다는 가설을 검증)

### 6.1 CUDA Graph 절감폭 — 출력이 길어질수록 커짐

| 출력 토큰 | eager latency | graph latency | 배수 |
|---|---|---|---|
| 8 | 180.7 ms | 55.7 ms | 3.24x |
| 32 | 682.4 ms | 144.0 ms | 4.74x |
| 64 | 1336.6 ms | 258.5 ms | 5.17x |
| 128 | 2759.3 ms | 499.2 ms | 5.53x |
| 256 | 5393.5 ms | 980.5 ms | 5.50x (수렴) |

### 6.2 Prefix Caching의 latency 기여도 — 출력이 길어질수록 희석됨

| 출력 토큰 | 캐싱으로 절감된 시간 | 전체 latency 대비 비중 |
|---|---|---|
| 8 | 7.1 ms | **11.2%** |
| 32 | 6.1 ms | 4.1% |
| 64 | 10.2 ms | 3.8% |
| 128 | 9.1 ms | 1.8% |
| 256 | 9.6 ms | **1.0%** |

`results/decode_sweep_summary.csv` (variant: `graph` vs `graph_no_cache`의 latency 차이).

### 해석

절감되는 절대 시간(~7~10ms)은 출력 길이와 무관하게 거의 고정이다(prefill만 줄여주는
최적화이므로). 하지만 전체 latency는 decode 길이에 비례해 계속 늘어나므로, 같은
절감분이 차지하는 비중은 11%→1%로 꾸준히 희석된다. 반대로 CUDA Graph는 decode
스텝마다 절약되므로 스텝 수가 많을수록 누적 절감분이 커지다가, decode가 전체
latency를 지배하는 지점(~128토큰 이상)에서 배수가 5.5배 선에 수렴한다.

**결론**: 두 최적화는 워크로드 성격에 따라 정반대로 중요해진다. 응답이 짧은 태스크
(챗봇 인터랙션, 분류, 짧은 추출 — 이 프로젝트의 시나리오)일수록 **Prefix Caching**이
체감 latency에 영향을 주고, 응답이 긴 태스크(요약, 긴 글 생성)일수록 **CUDA Graph**가
지배적이다.

**재현**:
```bash
python scripts/decode_length_sweep.py --output-lengths 8,32,64,128,256 --num-requests 4
```

---

## 7. 종합 결론

| 최적화 | 확인된 효과 | 어떤 조건에서 뚜렷한가 |
|---|---|---|
| CUDA Graph | latency/TPS 최대 **4.6~5.5배** | 출력이 길수록(decode-bound) 강함 |
| Prefix Caching | TTFT **1.0~1.5배** (캐시 토큰 수에 비례) | 캐시되는 토큰 절대량이 클수록(긴 프리픽스), 출력이 짧을수록 강함 |
| Continuous Batching | 집계 처리량 **1.67배** | 요청이 동시에 몰릴수록 강함 |
| nano-vLLM 전체 vs Transformers baseline | latency **2.4~5.7배**, 처리량 **1.5배** | 세 최적화가 누적된 결과 |

이 결과들은 참조 자료(Continuous Batching, PagedAttention, Prefix Caching, CUDA
Graph)를 그대로 재진술한 게 아니라, **실제 GPU 위에서 이 워크로드·이 모델 규모로
실측했을 때 각 최적화가 정확히 어디서(prefill vs decode), 어떤 조건에서 효과가
나타나거나 사라지는지**를 정량적으로 보여준다. 특히 "짧은 프리픽스에서는 prefix
caching 효과가 고정 오버헤드에 묻힌다"는 발견은 참조 자료에 없는, 실측을 통해서만
얻을 수 있는 2차 인사이트다.

### 원시 데이터

모든 수치의 원본은 `results/` 디렉터리의 CSV에 있다.

| 파일 | 내용 |
|---|---|
| `comparison_ttft_summary.csv` / `comparison_throughput_summary.csv` | 5개 모드 종합 비교 (실험 B·C·D) |
| `long_full_ttft.csv` / `long_no_cache_ttft.csv` | 긴 프리픽스 캐싱 재검증 (실험 A) |
| `decode_sweep_summary.csv` | 디코드 길이 스윕 (실험 E) |
| `nanovllm_full*.csv`, `nanovllm_no_*.csv`, `baseline_transformers*.csv` | 위 요약의 요청 단위 원본 로그 |
