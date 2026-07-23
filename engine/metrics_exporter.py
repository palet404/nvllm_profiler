"""
engine/metrics_exporter.py
RequestMetrics/GPUSnapshot 실측값을 Prometheus 메트릭으로 변환해 Pushgateway로
1회 push하는 어댑터. 기존 RequestMetrics/GPUSnapshot dataclass와 CSV 저장 로직은
건드리지 않는다 — scripts/profile_nanovllm.py·profile_transformers_baseline.py가 run_workload() 이후 결과를
그대로 넘겨서 호출하는 선택적 레이어다.

Pushgateway를 쓰는 이유: 이 프로젝트의 스크립트(profile_nanovllm.py 등)는 1회 실행되고
끝나는 배치 작업이라, Prometheus가 pull(scrape)할 시점에 프로세스가 이미 종료돼
있다. Prometheus 공식 문서가 Pushgateway의 유일하게 타당한 용도로 명시하는
"서비스 레벨 배치 작업"에 정확히 해당하는 케이스다.

request_id/arrival_time/prompt_preview처럼 계속 바뀌는 값은 라벨로 쓰지 않는다 —
Pushgateway는 push된 시계열을 자동 만료시키지 않으므로, cardinality가 유한한
run_tag/cache_enabled/cuda_graph_enabled/continuous_batching만 라벨로 둔다.

request_index(도착 순서, 0부터)는 예외적으로 라벨에 포함시킨다 — request_id와
달리 --num-requests로 상한이 정해진 유한 값이고, push_to_gateway()가 매번 해당
job의 이전 메트릭 전체를 교체하므로(merge가 아님) 재실행 시 이전 인덱스가 쌓이지
않는다. 목적은 "run 전체를 percentile 하나로 뭉갠 히스토그램"이 아니라 "요청이
진행되면서 TTFT/latency가 실제로 어떻게 변하는지"를 요청 단위로 그대로 남기는 것 —
Grafana의 Trend 패널(시간이 아닌 임의 숫자 필드를 X축으로 쓰는 패널)로 request_index를
X축 삼아 꺾은선으로 그린다.
"""
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway, pushadd_to_gateway

from engine.gpu_metrics import GPUSnapshot
from engine.nanovllm_engine import RequestMetrics

TTFT_BUCKETS = (10, 20, 30, 50, 75, 100, 150, 250, 400, 600, 1000, 2000, 5000)
TPS_BUCKETS = (5, 10, 20, 40, 60, 100, 150, 250)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)

LABEL_NAMES = ["run_tag", "cache_enabled", "cuda_graph_enabled", "continuous_batching"]


def build_registry(
    results: list[RequestMetrics],
    gpu_history: list[GPUSnapshot],
    run_tag: str,
    cache_enabled: bool,
    cuda_graph_enabled: bool,
    continuous_batching: bool,
    gpu_snapshots_by_index: Optional[list[tuple[int, GPUSnapshot]]] = None,
) -> CollectorRegistry:
    """
    이번 run 1회분의 실측치를 담은 CollectorRegistry를 만든다. 기본(글로벌)
    registry 대신 매 호출마다 새 registry를 쓰는 이유는, 기본 registry에는
    ProcessCollector 등 이 프로젝트와 무관한 메트릭이 섞여 push되는 걸 막기 위함이다.
    """
    registry = CollectorRegistry()
    label_values = (
        run_tag,
        str(cache_enabled).lower(),
        str(cuda_graph_enabled).lower(),
        str(continuous_batching).lower(),
    )

    ttft_hist = Histogram(
        "nanovllm_ttft_milliseconds", "Time to first token (ms)",
        labelnames=LABEL_NAMES, buckets=TTFT_BUCKETS, registry=registry,
    )
    latency_hist = Histogram(
        "nanovllm_latency_milliseconds", "End-to-end request latency (ms)",
        labelnames=LABEL_NAMES, buckets=TTFT_BUCKETS, registry=registry,
    )
    tps_hist = Histogram(
        "nanovllm_request_tokens_per_second", "Per-request throughput (tok/s)",
        labelnames=LABEL_NAMES, buckets=TPS_BUCKETS, registry=registry,
    )
    batch_hist = Histogram(
        "nanovllm_batch_size_at_admit", "Concurrent sequences at admit time",
        labelnames=LABEL_NAMES, buckets=BATCH_BUCKETS, registry=registry,
    )
    prompt_tokens_total = Counter(
        "nanovllm_prompt_tokens_total", "Cumulative prompt tokens",
        labelnames=LABEL_NAMES, registry=registry,
    )
    output_tokens_total = Counter(
        "nanovllm_output_tokens_total", "Cumulative output tokens",
        labelnames=LABEL_NAMES, registry=registry,
    )
    cached_tokens_total = Counter(
        "nanovllm_cached_tokens_total", "Cumulative prefix-cache-hit tokens",
        labelnames=LABEL_NAMES, registry=registry,
    )
    new_prefill_tokens_total = Counter(
        "nanovllm_new_prefill_tokens_total", "Cumulative freshly-prefilled tokens",
        labelnames=LABEL_NAMES, registry=registry,
    )
    cache_hit_requests_total = Counter(
        "nanovllm_cache_hit_requests_total", "Requests by prefix cache hit/miss",
        labelnames=LABEL_NAMES + ["hit"], registry=registry,
    )
    aggregate_tps = Gauge(
        "nanovllm_aggregate_tokens_per_second", "Aggregate wall-clock throughput (tok/s)",
        labelnames=LABEL_NAMES, registry=registry,
    )
    gpu_util = Gauge(
        "nanovllm_gpu_util_percent", "GPU utilization (%) - mean over run",
        labelnames=LABEL_NAMES, registry=registry,
    )
    gpu_mem_util = Gauge(
        "nanovllm_gpu_mem_util_percent", "GPU VRAM utilization (%) - mean over run",
        labelnames=LABEL_NAMES, registry=registry,
    )
    last_success = Gauge(
        "nanovllm_run_last_success_unixtime", "Timestamp this run last completed successfully",
        labelnames=LABEL_NAMES, registry=registry,
    )

    # 요청별 실측값을 percentile로 뭉개지 않고 그대로 보존하는 Gauge. request_index는
    # 도착 순서(0부터) — Grafana Trend 패널에서 X축으로 써서 "요청이 진행되며 값이 어떻게
    # 바뀌는지"를 꺾은선으로 그리기 위함.
    per_request_label_names = LABEL_NAMES + ["request_index"]
    request_ttft = Gauge(
        "nanovllm_request_ttft_milliseconds", "TTFT (ms) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_latency = Gauge(
        "nanovllm_request_latency_milliseconds", "End-to-end latency (ms) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_tps = Gauge(
        "nanovllm_request_tps", "Throughput (tok/s) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_batch_size = Gauge(
        "nanovllm_request_batch_size_at_admit", "Concurrent sequences at admit time for a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    # 병목(느린 요청) 재현용 — 그 요청의 입력/출력 토큰 shape을 그대로 남겨서, TTFT/latency가
    # 튄 request_index를 찾으면 그 프롬프트/출력 길이 조합으로 바로 재현할 수 있게 한다.
    # prompt_tokens_total/output_tokens_total(Counter)은 누적값이라 요청 단위로 못 쪼개므로 별도 Gauge로 둔다.
    request_prompt_tokens = Gauge(
        "nanovllm_request_prompt_tokens", "Prompt token count of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_output_tokens = Gauge(
        "nanovllm_request_output_tokens", "Output token count of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    # request_index별 GPU 스냅샷 — wall-clock 시계열 대신 "요청 순서"를 X축으로 삼아
    # TTFT/latency 패널과 똑같은 Trend 패널로 그린다(짧은 run에서는 Prometheus
    # scrape_interval(5s)보다 요청이 더 빨리 끝나 wall-clock 시계열이 계단식으로만
    # 찍히는 문제를 피하기 위함 — request_index 기준이면 요청 수만큼 점이 보장된다).
    request_gpu_util = Gauge(
        "nanovllm_request_gpu_util_percent", "GPU utilization (%) sampled when this request finished, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_gpu_mem_util = Gauge(
        "nanovllm_request_gpu_mem_util_percent", "GPU VRAM utilization (%) sampled when this request finished, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )

    for r in results:
        ttft_hist.labels(*label_values).observe(r.ttft_ms)
        latency_hist.labels(*label_values).observe(r.latency_ms)
        tps_hist.labels(*label_values).observe(r.tps)
        batch_hist.labels(*label_values).observe(r.batch_size_at_admit)
        prompt_tokens_total.labels(*label_values).inc(r.prompt_tokens)
        output_tokens_total.labels(*label_values).inc(r.output_tokens)
        cached_tokens_total.labels(*label_values).inc(r.cached_tokens)
        new_prefill_tokens_total.labels(*label_values).inc(r.new_prefill_tokens)
        cache_hit_requests_total.labels(*label_values, "true" if r.prefix_cache_hit else "false").inc()

    for idx, r in enumerate(sorted(results, key=lambda r: r.arrival_time)):
        request_ttft.labels(*label_values, str(idx)).set(r.ttft_ms)
        request_latency.labels(*label_values, str(idx)).set(r.latency_ms)
        request_tps.labels(*label_values, str(idx)).set(r.tps)
        request_batch_size.labels(*label_values, str(idx)).set(r.batch_size_at_admit)
        request_prompt_tokens.labels(*label_values, str(idx)).set(r.prompt_tokens)
        request_output_tokens.labels(*label_values, str(idx)).set(r.output_tokens)

    if results:
        total_output = sum(r.output_tokens for r in results)
        wall_clock_s = (
            max(r.arrival_time + r.latency_ms / 1000 for r in results)
            - min(r.arrival_time for r in results)
        )
        aggregate_tps.labels(*label_values).set(total_output / wall_clock_s if wall_clock_s > 0 else 0.0)

    if gpu_history:
        utils = [s.gpu_util_pct for s in gpu_history]
        mems = [s.mem_util_pct for s in gpu_history]
        gpu_util.labels(*label_values).set(sum(utils) / len(utils))
        gpu_mem_util.labels(*label_values).set(sum(mems) / len(mems))

    for request_index, snapshot in gpu_snapshots_by_index or []:
        request_gpu_util.labels(*label_values, str(request_index)).set(snapshot.gpu_util_pct)
        request_gpu_mem_util.labels(*label_values, str(request_index)).set(snapshot.mem_util_pct)

    last_success.labels(*label_values).set_to_current_time()
    return registry


def push_run_metrics(
    results: list[RequestMetrics],
    gpu_history: list[GPUSnapshot],
    run_tag: str,
    cache_enabled: bool,
    cuda_graph_enabled: bool,
    continuous_batching: bool,
    pushgateway_url: str = "localhost:9091",
    gpu_snapshots_by_index: Optional[list[tuple[int, GPUSnapshot]]] = None,
) -> None:
    """
    job=run_tag로 push한다. 같은 run_tag를 재실행하면 Pushgateway가 이전 값을
    덮어쓴다(grouping key가 동일하면 갱신) — job이 run_tag별로 갈라져 있어야
    서로 다른 ablation 모드의 시계열이 섞이지 않는다.

    gpu_snapshots_by_index: push_live_progress()로 실행 중 쌓아온 request_index별 GPU
    스냅샷. push_to_gateway()는 PUT(job 전체 교체)라 여기서도 다시 채워 넣지 않으면
    실행 중 쌓인 nanovllm_request_gpu_util_percent 시계열이 최종 push에서 사라진다.
    """
    registry = build_registry(
        results, gpu_history, run_tag, cache_enabled, cuda_graph_enabled, continuous_batching,
        gpu_snapshots_by_index=gpu_snapshots_by_index,
    )
    push_to_gateway(pushgateway_url, job=run_tag, registry=registry)


def push_live_progress(
    results_so_far: list[tuple[int, RequestMetrics]],
    run_tag: str,
    cache_enabled: bool,
    cuda_graph_enabled: bool,
    continuous_batching: bool,
    pushgateway_url: str = "localhost:9091",
    gpu_snapshots_by_index: Optional[list[tuple[int, GPUSnapshot]]] = None,
) -> None:
    """
    요청이 끝날 때마다(run 전체가 끝나길 기다리지 않고) '지금까지 끝난 요청 전부'의
    request_ttft/latency/tps/batch_size Gauge를 다시 채워서 push한다.
    push_run_metrics()는 run 전체가 끝나야 호출되므로 Grafana에 결과가 한꺼번에
    나타나지만, 이 함수는 run_workload()의 on_result 콜백에서 매 요청마다 불려
    대시보드가 실시간으로 채워지는 시연(라이브 데모)용이다.

    반드시 '지금까지 끝난 요청 전부'를 매번 다시 채워야 한다 — Pushgateway의
    POST(pushadd)는 label 조합 단위가 아니라 "메트릭 이름(family)" 단위로
    교체한다. 즉 이번 push에 request_index=5 하나만 담으면, 그 순간
    nanovllm_request_ttft_milliseconds 전체가 이 값 하나로 교체되면서 이전에
    push된 request_index=0..4가 전부 사라진다(실측으로 확인함 — 매 요청마다
    직전 인덱스 하나만 남고 이전 것들이 사라지는 버그가 있었다). 그래서 매번
    누적된 전체 리스트를 다시 그려 넣어야 "새 점이 이전 점들 옆에 이어붙는"
    형태로 보인다.
    집계 히스토그램/카운터(build_registry)는 여기서 채우지 않는다 — run이 끝난
    뒤 push_run_metrics()가 한 번에 완성된 값으로 채운다(도중에 부분합을
    push하면 Counter가 잘못 누적될 수 있어서다).

    gpu_snapshots_by_index: 지금까지 완료된 각 요청이 끝난 시점의 (request_index,
    GPUSnapshot) 쌍 전부(scripts/_common.py의 make_live_pusher가 gpu_daemon.latest()로
    누적). wall-clock 시계열(Prometheus scrape_interval 기준)이 아니라 request_ttft와
    똑같이 request_index를 X축으로 삼는 Trend 패널용이라, 짧은 run에서도 요청 수만큼
    점이 보장된다.
    """
    registry = CollectorRegistry()
    label_values = (
        run_tag,
        str(cache_enabled).lower(),
        str(cuda_graph_enabled).lower(),
        str(continuous_batching).lower(),
    )
    per_request_label_names = LABEL_NAMES + ["request_index"]

    request_ttft = Gauge(
        "nanovllm_request_ttft_milliseconds", "TTFT (ms) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_latency = Gauge(
        "nanovllm_request_latency_milliseconds", "End-to-end latency (ms) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_tps = Gauge(
        "nanovllm_request_tps", "Throughput (tok/s) of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_batch_size = Gauge(
        "nanovllm_request_batch_size_at_admit", "Concurrent sequences at admit time for a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_prompt_tokens = Gauge(
        "nanovllm_request_prompt_tokens", "Prompt token count of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )
    request_output_tokens = Gauge(
        "nanovllm_request_output_tokens", "Output token count of a single request, by arrival order",
        labelnames=per_request_label_names, registry=registry,
    )

    for request_index, result in results_so_far:
        request_ttft.labels(*label_values, str(request_index)).set(result.ttft_ms)
        request_latency.labels(*label_values, str(request_index)).set(result.latency_ms)
        request_tps.labels(*label_values, str(request_index)).set(result.tps)
        request_batch_size.labels(*label_values, str(request_index)).set(result.batch_size_at_admit)
        request_prompt_tokens.labels(*label_values, str(request_index)).set(result.prompt_tokens)
        request_output_tokens.labels(*label_values, str(request_index)).set(result.output_tokens)

    if gpu_snapshots_by_index:
        request_gpu_util = Gauge(
            "nanovllm_request_gpu_util_percent", "GPU utilization (%) sampled when this request finished, by arrival order",
            labelnames=per_request_label_names, registry=registry,
        )
        request_gpu_mem_util = Gauge(
            "nanovllm_request_gpu_mem_util_percent", "GPU VRAM utilization (%) sampled when this request finished, by arrival order",
            labelnames=per_request_label_names, registry=registry,
        )
        for request_index, snapshot in gpu_snapshots_by_index:
            request_gpu_util.labels(*label_values, str(request_index)).set(snapshot.gpu_util_pct)
            request_gpu_mem_util.labels(*label_values, str(request_index)).set(snapshot.mem_util_pct)

    pushadd_to_gateway(pushgateway_url, job=run_tag, registry=registry)
