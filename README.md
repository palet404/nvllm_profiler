# nano-vLLM Profiler

nano-vLLM(Qwen3-0.6B)과 HF Transformers baseline을 실제 GPU 추론으로 실측 비교하는
프로파일링 툴킷. 더미/시뮬레이션 수치는 없고, 모든 결과는 실제 모델 로드 + 실제 GPU
연산 + Prometheus에 실제로 저장된 값에서 나온다.

## 환경

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 6.17) |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8GB VRAM |
| NVIDIA 드라이버 | 595.71.05 |
| Python | 3.10 (conda env `nano_vllm`) |
| PyTorch | 2.5.1+cu121 |
| nano-vllm | 0.2.0 (FlashAttention + CUDA Graph + PagedAttention 내장) |
| Transformers | 5.12.1 (baseline, `attn_implementation="flash_attention_2"`) |
| 모델 | Qwen3-0.6B (`--model-path`, 기본값 `~/huggingface/Qwen3-0.6B`) |

CUDA Toolkit을 시스템에 따로 설치할 필요는 없다 — `torch==2.5.1+cu121`, `flash-attn`
모두 CUDA 런타임이 포함된 pip 휠로 설치된다. 필요한 건 GPU와 그에 맞는 NVIDIA
드라이버뿐이다.

### 설치 — conda로 한 번에 (권장)

```bash
conda env create -f environment.yml
conda activate nano_vllm

# 설치 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`environment.yml`은 Python 3.10 conda 환경을 만들고, 그 안에 pip로 `torch==2.5.1+cu121`
→ `nano-vllm==0.2.0`(flash-attn/triton/xxhash 자동 포함) → `transformers==5.12.1` →
`pandas`/`prometheus-client`/`datasets`를 이 순서로 설치한다(torch가 먼저 설치돼야
flash-attn이 올바른 ABI로 빌드/선택됨). `requirements.txt`도 같은 패키지/버전을
담고 있어 `pip install -r requirements.txt`로 기존 conda/venv 환경에 수동 설치해도
동일하다.

nano-vLLM과 Transformers는 같은 GPU(8GB)에 동시에 못 올라가므로(VRAM 충돌) 항상
별도 프로세스로 순차 실행한다.

## 프로파일러 구조

```
nvllm_profiler/
├── workload/dataset_loaders.py            공개 벤치마크 데이터셋(SQuAD 등) → 프롬프트 로더
├── engine/
│   ├── nanovllm_engine.py                 실제 nanovllm.LLM 래퍼 (PagedAttention/CUDA Graph/Prefix Caching)
│   ├── transformers_baseline_engine.py    HF Transformers baseline — flash_attention_2 + 수동 prefill/decode 루프
│   ├── gpu_metrics.py                     nvidia-smi 실측 폴링 데몬
│   └── metrics_exporter.py                RequestMetrics/GPUSnapshot → Prometheus 메트릭 변환 + Pushgateway 전송
├── scripts/
│   ├── _common.py                         두 프로파일링 스크립트가 공유하는 배관(시나리오 로딩, Pushgateway 출력, PromQL 헬퍼)
│   ├── profile_nanovllm.py                nano-vLLM 실측 (technique/ablation 플래그, --push-metrics)
│   └── profile_transformers_baseline.py   HF Transformers baseline 실측 (--dataset 4종, --query-only, --push-metrics)
├── utils/
│   └── export_metrics_from_prometheus.py  Prometheus DB에 쌓인 run_tag 하나를 PromQL로 조회해 CSV로 재구성
├── monitoring/
│   ├── prometheus.yml                     Pushgateway 스크레이프 설정 (honor_labels: true)
│   └── grafana/                           대시보드 JSON + 프로비저닝 (run_tag 필터 변수 포함)
└── results/                                CSV 산출물 (git-ignored, utils/export_metrics_from_prometheus.py가 생성)
```

**데이터 흐름**: 실행 스크립트(`profile_nanovllm.py`/`profile_transformers_baseline.py`)가
요청을 하나씩 처리하며 `RequestMetrics`(TTFT/latency/tps/prompt_tokens/output_tokens 등)를
쌓고, `--push-metrics`를 주면 `engine/metrics_exporter.py`가 이걸 Prometheus 메트릭으로
변환해 **Pushgateway**로 push한다. Prometheus가 Pushgateway를 5초 간격으로 스크레이프해
저장하고, Grafana가 그 위에 대시보드를 그린다. 로컬 CSV를 직접 저장하는 경로는 없다 —
결과는 항상 Prometheus가 실제로 갖고 있는 값을 `utils/export_metrics_from_prometheus.py`로
다시 꺼내서 확인한다.

Pushgateway를 쓰는 이유: 이 프로젝트의 스크립트는 1회 실행되고 끝나는 배치 작업이라,
Prometheus가 직접 pull(scrape)할 시점엔 프로세스가 이미 종료돼 있기 때문이다.

## 모니터링 스택 기동 (최초 1회)

```bash
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
# → http://localhost:3000 (admin/admin) 에서 "nano-vLLM Profiler" 대시보드 자동 로드
```

## 실행했던 실험 재현하기

SQuAD `Sexual_orientation` 문서를 프리픽스로 고정하고, 같은 30개 질문으로 세 가지
조건을 비교한다: Transformers(prefix 포함) vs Transformers(prefix 없이 query만) vs
nano-vLLM(prefix caching 적용).

```bash
# 1) Transformers baseline — 문서 + 질문 (매 요청 prefix 재계산)
python scripts/profile_transformers_baseline.py \
  --dataset squad --squad-title Sexual_orientation \
  --num-requests 30 --target-prefix-tokens 1900 --max-output-tokens 48 \
  --tag "transformers(title+query)" --push-metrics

# 2) Transformers baseline — 질문만 (prefix 재계산 비용 분리한 대조군)
python scripts/profile_transformers_baseline.py \
  --dataset squad --squad-title Sexual_orientation \
  --num-requests 30 --target-prefix-tokens 1900 --max-output-tokens 48 \
  --query-only \
  --tag "transformers(query_only)" --push-metrics

# 3) nano-vLLM — 문서 + 질문, prefix caching 적용, 순차 처리(batch=1)로 통제
python scripts/profile_nanovllm.py \
  --technique prefix_cache \
  --dataset squad --squad-title Sexual_orientation \
  --num-requests 30 --target-prefix-tokens 1900 --max-output-tokens 48 \
  --sequential \
  --tag "nanovllm(title+query)" --push-metrics

# 4) Prometheus에서 결과 CSV로 재구성 (각 run_tag마다)
python utils/export_metrics_from_prometheus.py --run-tag "transformers(title+query)"
python utils/export_metrics_from_prometheus.py --run-tag "transformers(query_only)"
python utils/export_metrics_from_prometheus.py --run-tag "nanovllm(title+query)"
```

`--sequential`(nano-vLLM)은 Continuous Batching 이득을 배제해서, Transformers baseline과
똑같이 배치 크기 1·순차 도착 조건에서 prefix caching + CUDA Graph + FlashAttention만의
효과를 1:1로 비교하기 위한 것이다. 두 스크립트의 `--target-prefix-tokens`는 반드시
같은 값을 써야 한다 — 값이 다르면 포함되는 문단 집합이 달라져서 같은 30개 질문을
비교한다는 보장이 깨진다.

### 결과 확인

```bash
# 요청별 TTFT (request_index 순서대로) — @csv는 값에 큰따옴표를 씌우므로 sort -n 전에 벗겨낸다
curl -s 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=nanovllm_request_ttft_milliseconds{run_tag="transformers(title+query)"}' \
  | jq -r '.data.result[] | [.metric.request_index, .value[1]] | @csv' | tr -d '"' | sort -t, -k1 -n

# 집계 평균 (Histogram의 _sum/_count) — 반드시 한 줄로: --data-urlencode 값 앞에 개행이 섞이면
# "query" 대신 "\nquery"라는 필드명으로 전송돼 Prometheus가 파싱을 못 한다
curl -s 'http://localhost:9090/api/v1/query' --data-urlencode 'query=nanovllm_ttft_milliseconds_sum{run_tag="transformers(title+query)"} / nanovllm_ttft_milliseconds_count{run_tag="transformers(title+query)"}' | jq .
```

Grafana 대시보드는 상단 `run_tag` 드롭다운(다중 선택 + All)으로 원하는 run만 골라
TTFT/Latency/TPS/GPU Util/VRAM Util 패널을 볼 수 있다.

## 다른 시나리오로 실행하기

```bash
# nano-vLLM: technique별 기본 데이터셋 자동 적용 (--dataset 생략 가능)
python scripts/profile_nanovllm.py --technique prefix_cache --num-requests 12 --push-metrics
python scripts/profile_nanovllm.py --technique cuda_graph --num-requests 8 --max-output-tokens 256 --push-metrics
python scripts/profile_nanovllm.py --technique continuous_batching --num-requests 24 --arrival-interval-s 0.05 --push-metrics

# ablation 플래그로 최적화 하나씩 끄기
python scripts/profile_nanovllm.py --technique cuda_graph --num-requests 8 --enforce-eager --push-metrics          # CUDA Graph 끄기
python scripts/profile_nanovllm.py --technique prefix_cache --num-requests 12 --disable-prefix-cache --push-metrics # Prefix Caching 끄기
python scripts/profile_nanovllm.py --technique continuous_batching --num-requests 24 --sequential --push-metrics    # Continuous Batching 끄기

# Transformers baseline: --dataset 4종(tool_catalog/squad/math500/kmmlu) 전부 지원
python scripts/profile_transformers_baseline.py --dataset math500 --num-requests 8 --max-output-tokens 256 --push-metrics
```

`--dataset` 종류: `tool_catalog`(glaive-function-calling-v2), `squad`(rajpurkar/squad),
`math500`(HuggingFaceH4/MATH-500), `kmmlu`(HAERAE-HUB/KMMLU).

**모델/GPU 설정도 전부 CLI 인자다** (별도 설정 파일 없음, 두 프로파일링 스크립트 공통):

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--model-path` | `~/huggingface/Qwen3-0.6B` | 로드할 모델 경로 |
| `--max-model-len` | `8192` | [nano-vLLM만] KV 캐시가 수용할 최대 시퀀스 길이 |
| `--gpu-memory-utilization` | `0.85` | [nano-vLLM만] 예약할 VRAM 비율 |
| `--gpu-poll-interval-s` | `0.5` | nvidia-smi 폴링 주기(초) |
| `--gpu-history-maxlen` | `600` | GPU 스냅샷 히스토리 보관 개수 |

```bash
python scripts/profile_nanovllm.py --technique prefix_cache --num-requests 12 \
  --model-path ~/huggingface/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.7 --push-metrics
```
