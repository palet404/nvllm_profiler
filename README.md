# nano-vLLM Profiler

nano-vLLM(Qwen3-0.6B, RTX 3070 Ti 8GB)의 세 가지 서빙 최적화 — **Prefix Caching**,
**Continuous Batching**, **CUDA Graph** — 를 실제 GPU 추론으로 실측하는 프로파일링
툴킷. 더미/시뮬레이션 수치는 없으며, 모든 결과는 실제 모델 로드 + 실제 GPU 연산에서
나온다.

## 시나리오

사내 이메일에서 업무 제출 기한과 제출물을 추출하는 어시스턴트를 가정한다
(`workload/generator.py`, `config.py`). 모든 요청은 다음 구조를 갖는다.

```
[공통 시스템 프롬프트: 추출 규칙 + few-shot 예시]  ← 모든 요청에 토큰 단위로 동일
[가변 이메일 본문: 발신팀/제출물/기한/문구가 매번 다름]
```

- **`SYSTEM_PREFIX`** (~370토큰, KV 캐시 1블록): 기본 프리픽스.
- **`LONG_SYSTEM_PREFIX`** (~1015토큰, KV 캐시 4블록): few-shot 예시를 7개로 늘려
  RAG/few-shot이 많은 프로덕션급 시스템 프롬프트 규모를 흉내낸 스트레스 테스트용.
- **`duplicate_ratio`**: 이미 생성한 이메일을 그대로 재전송할 확률. 본문까지 100%
  캐시 히트하는 극단적 케이스를 재현할 때 쓴다.

공통 프리픽스가 모든 요청 앞에 토큰 단위로 동일하게 붙기 때문에, nano-vLLM의
`BlockManager`가 해당 KV 블록을 재계산 없이 참조 카운트만 올려 재사용하는지
(Prefix Caching) 관찰할 수 있다.

## 프로파일러 작동 방식

### 무엇을 재는가 (`engine/nanovllm_engine.py::RequestMetrics`)

| 메트릭 | 의미 |
|---|---|
| `ttft_ms` | Time To First Token — 도착부터 첫 토큰까지 |
| `tps` | 초당 생성 토큰 수 (처리량) |
| `latency_ms` | 도착부터 전체 완료까지 |
| `cached_tokens` / `prefix_cache_hit` | Prefix Cache로 재사용된 KV 토큰 수 |
| `new_prefill_tokens` | 캐시 미스라 새로 prefill해야 한 토큰 수 |
| `batch_size_at_admit` | 요청 도착 시점에 이미 대기/실행 중이던 시퀀스 수 |

`engine/gpu_metrics.py`가 별도 스레드에서 `nvidia-smi`를 폴링해 GPU 사용률/VRAM
시계열(`GPUSnapshot`)을 함께 수집한다.

### 어떻게 재는가

1. **스케줄러 3단계를 직접 호출**: `LLMEngine.step()`을 그대로 쓰지 않고
   `scheduler.schedule() → model_runner.call("run") → scheduler.postprocess()`를
   직접 풀어서 호출한다. `step()`의 반환값만으로는 "이번 스텝에 첫 토큰을 받은"
   시퀀스(TTFT)를 알 수 없기 때문이다.
2. **Prefix Cache 히트량은 스냅샷 타이밍이 핵심**: `postprocess()`가
   `num_cached_tokens`를 "누적 계산된 토큰 수" 카운터로 재사용해버리므로,
   `schedule()` 직후 postprocess가 덮어쓰기 전에 스냅샷을 떠야 진짜 캐시 히트량을
   얻는다.
3. **Ablation 스위치로 최적화 기여도 격리**:
   - `set_prefix_cache_enabled(bool)` — `BlockManager.can_allocate`를 몽키패치해
     캐시 조회만 끄고 블록 할당/해제 로직은 그대로 둔다.
   - `continuous_batching=False` — admit 타이밍만 바꿔 "배치가 완전히 비어야
     다음 배치를 받는" 정적 스케줄링을 재현한다 (스케줄러/블록매니저는 그대로).
   - `enforce_eager` — CUDA Graph on/off.
4. **GPU 워밍업 필수**: 측정 전 더미 프롬프트로 한 번 돌려서 Triton/flash-attn
   JIT, cuBLAS 알고리즘 탐색 같은 1회성 비용을 흡수한다. 그렇지 않으면
   콜드스타트가 캐싱 효과를 완전히 가려버린다.

### 시각화 — Prometheus / Grafana 연동

자체 대시보드는 만들지 않는다. `engine/metrics_exporter.py`가 `RequestMetrics`/
`GPUSnapshot`을 Prometheus 메트릭(Histogram/Counter/Gauge)으로 변환해 Pushgateway로
push하고, Grafana는 Prometheus를 데이터소스로 붙여 시각화한다.

**Pushgateway를 쓰는 이유**: 이 프로젝트의 스크립트(`profile_run.py` 등)는 1회
실행되고 끝나는 배치 작업이라, Prometheus가 pull(scrape)할 시점엔 프로세스가 이미
종료돼 있다 — Pushgateway는 Prometheus 공식 문서가 명시하는 "서비스 레벨 배치
작업" 용도에 정확히 해당한다.

```bash
# 1) Pushgateway / Prometheus / Grafana를 한 네트워크에 기동 (최초 1회)
docker network create nanovllm-monitoring
docker run -d --name pushgateway --network nanovllm-monitoring -p 9091:9091 \
  --restart unless-stopped prom/pushgateway
docker run -d --name prometheus --network nanovllm-monitoring -p 9090:9090 \
  -v "$(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  --restart unless-stopped prom/prometheus
docker run -d --name grafana --network nanovllm-monitoring -p 3000:3000 \
  -v "$(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" \
  -v "$(pwd)/monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
  --restart unless-stopped grafana/grafana

# 2) 벤치마크 실행 시 --push-metrics만 추가하면 자동으로 Pushgateway에 전송됨
python scripts/profile_run.py --num-requests 8 --tag nanovllm_full_ttft --push-metrics
python scripts/run_comparison_suite.py --num-requests 8 --push-metrics   # 5개 모드 전부 push

# 3) http://localhost:3000 (admin/admin) 접속 → "nano-vLLM Profiler" 대시보드 자동 로드됨
```

`monitoring/prometheus.yml`은 `honor_labels: true`로 Pushgateway를 스크레이프한다
(안 하면 push된 `run_tag` 라벨이 `exported_job` 등으로 이름이 바뀐다).
`monitoring/grafana/provisioning/`은 Grafana 기동 시 Prometheus 데이터소스와
대시보드(TTFT/Latency p95, 집계 TPS, cache hit ratio, GPU util/VRAM — 전부
`run_tag`별 비교)를 자동 등록한다.

request_id/arrival_time처럼 계속 바뀌는 값은 라벨로 쓰지 않는다 — Pushgateway는
push된 시계열을 자동 만료시키지 않으므로, `run_tag`/`cache_enabled`/
`cuda_graph_enabled`/`continuous_batching`처럼 cardinality가 유한한 값만 라벨로 쓴다.

## 프로젝트 구조

```
nvllm_profiler/
├── config.py                              MODEL_PATH, SYSTEM_PREFIX, LONG_SYSTEM_PREFIX
├── workload/generator.py                  가상 이메일 워크로드 생성기
├── engine/
│   ├── nanovllm_engine.py                 실제 nanovllm.LLM 래퍼 (핵심 프로파일링 로직)
│   ├── transformers_baseline_engine.py    HF Transformers baseline (비교용)
│   ├── gpu_metrics.py                     nvidia-smi 실측 폴링 데몬
│   └── metrics_exporter.py                RequestMetrics/GPUSnapshot → Prometheus 메트릭 변환 + Pushgateway 전송
├── scripts/
│   ├── profile_run.py                     단일 모드 실행 (ablation 플래그, --push-metrics)
│   ├── baseline_run.py                    Transformers baseline 실행 (--push-metrics)
│   ├── run_comparison_suite.py            5개 모드 자동 비교 (--push-metrics)
│   └── decode_length_sweep.py             출력 길이별 최적화 기여도 스윕
├── monitoring/
│   ├── prometheus.yml                     Pushgateway 스크레이프 설정
│   └── grafana/provisioning, dashboards/  Grafana 데이터소스/대시보드 자동 프로비저닝
├── results/                                CSV 원시 로그 (git-ignored, 로컬 실행 시 생성됨)
└── RESULTS.md                              실험 결과 상세 (방법론 + 수치 + 해석)
```

## 환경

아래는 실측을 검증한 실제 환경이다. 다른 조합(GPU/드라이버/CUDA 버전)에서도 동작할
가능성이 높지만 검증된 것은 아니다.

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 6.17) |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8GB VRAM |
| NVIDIA 드라이버 | 595.71.05 |
| Python | 3.10 (conda env `nano_vllm`) |
| PyTorch | 2.5.1 (`+cu121` 휠, pip로 설치 — 시스템에 별도 CUDA Toolkit/`nvcc` 불필요) |
| nano-vllm | 0.2.0 |
| Transformers | 5.12.1 |
| 모델 | Qwen3-0.6B (`~/huggingface/Qwen3-0.6B` 경로 가정, `config.py`에서 변경 가능) |

**시스템에 CUDA Toolkit을 따로 설치할 필요는 없다** — PyTorch·flash-attn 모두 CUDA
런타임이 포함된 pip 휠(`torch==2.5.1+cu121`, `nvidia-cu12-*` 패키지들)로 설치되고,
필요한 건 GPU와 그에 맞는 NVIDIA 드라이버뿐이다.

## 설치

GPU가 있는 Ubuntu 24.04 머신에서, NVIDIA 드라이버(`nvidia-smi`가 동작하는지로 확인)만
설치돼 있으면 된다.

### 방법 1 — conda로 한 번에 (권장)

```bash
conda env create -f environment.yml
conda activate nano_vllm
```

`environment.yml`은 Python 3.10 conda 환경을 만들고, 그 안에 pip로
`torch==2.5.1+cu121` → `nano-vllm`(flash-attn/triton/xxhash 자동 포함) →
`transformers` → `pandas`를 순서대로 설치한다.

### 방법 2 — 기존 conda(또는 venv) 환경에 수동 설치

`nano-vllm`의 의존 패키지인 `flash-attn`이 빌드/휠 선택 시점에 이미 설치된 torch의
ABI를 참조하므로, **torch를 먼저 설치**해야 한다.

```bash
conda create -n nano_vllm python=3.10 -y
conda activate nano_vllm
pip install -r requirements.txt
```

`requirements.txt`의 패키지 순서(torch → nano-vllm → transformers → pandas)를 그대로
따르면 된다. 만약 `flash-attn` 설치 단계에서 소스 빌드가 시도되며 실패한다면(드문
경우 — 이 환경에서는 PyPI의 prebuilt 휠이 그대로 설치됨), torch가 이미 설치된
상태에서 아래처럼 재시도한다.

```bash
pip install nano-vllm==0.2.0 --no-build-isolation
```

### 설치 확인

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 실행

```bash
# 단일 실행
python scripts/profile_run.py --num-requests 20 --duplicate-ratio 0.4

# ablation 플래그
python scripts/profile_run.py --num-requests 8 --enforce-eager           # CUDA Graph 끄기
python scripts/profile_run.py --num-requests 8 --disable-prefix-cache    # Prefix Caching 끄기
python scripts/profile_run.py --num-requests 8 --sequential              # Continuous Batching 끄기

# 5개 모드(정상/3가지 ablation/baseline) 자동 비교
python scripts/run_comparison_suite.py --num-requests 8 --max-output-tokens 24

# 출력 길이별 스윕
python scripts/decode_length_sweep.py --output-lengths 8,32,64,128,256 --num-requests 4

# Transformers baseline만 단독 실행
python scripts/baseline_run.py --num-requests 8 --tag baseline_throughput --save-csv
```

## 실험 결과 요약

자세한 방법론과 수치, 해석은 [RESULTS.md](RESULTS.md)에 정리했다. 핵심만 요약하면:

| 최적화 | 확인된 효과 | 어떤 조건에서 뚜렷한가 |
|---|---|---|
| CUDA Graph | latency/TPS 최대 **4.6~5.5배** | 출력이 길수록 (decode-bound) 강함 |
| Prefix Caching | TTFT **1.0~1.5배** (캐시 토큰 수에 비례) | 캐시 토큰 절대량이 클수록, 출력이 짧을수록 강함 |
| Continuous Batching | 집계 처리량 **1.67배** | 요청이 동시에 몰릴수록 강함 |
| nano-vLLM 전체 vs Transformers baseline | latency **2.4~5.7배**, 처리량 **1.5배** | 세 최적화가 누적된 결과 |


## 앞으로 추가해야 할 것

- **더 큰 모델/긴 컨텍스트 검증**: 현재 Qwen3-0.6B + 8GB VRAM 제약으로 실험한
  결과이므로, 더 큰 모델이나 실제 RAG 수준의 긴 프리픽스(수천 토큰)에서 Prefix
  Caching 효과가 얼마나 더 커지는지 추가 검증이 필요하다.
- **동시 요청 규모 확대**: 현재 8건 수준의 burst로 Continuous Batching을 검증했는데,
  더 많은 동시 요청(수십~수백 건)에서의 스케일링도 확인할 가치가 있다.

## 참고

- 실제 nano-vLLM 내부 동작(`Sequence.__len__`, `BlockManager.can_allocate` 등)을
  기반으로 계측했으므로, 코드를 수정할 때는
  `nanovllm/engine/scheduler.py`, `block_manager.py`, `sequence.py` 소스를 먼저
  확인할 것을 권장한다.
- 이 프로젝트는 Knou/Bench_server(HF Transformers vs nano-vLLM 실측 벤치마크)와는
  별개의 독립 포트폴리오 프로젝트다.
