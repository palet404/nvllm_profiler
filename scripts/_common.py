"""
scripts/_common.py
profile_nanovllm.py / profile_transformers_baseline.py가 공유하는 배관 코드.
"엔진을 어떻게 구동해서 지표를 뽑아내는가"(엔진마다 다름)는 건드리지 않고, 두
스크립트에서 그대로 반복되던 부분만 모았다:

  - 시나리오(데이터셋) 로딩 디스패치 + technique별 기본 데이터셋/기본 워크로드 크기
  - --tag/--push-metrics/--pushgateway-url CLI 플래그와 그 처리(Pushgateway 전송,
    라이브 push 콜백)
  - Prometheus PromQL 조회 헬퍼(utils/export_metrics_from_prometheus.py가 사용)
  - GPU 프로파일링 데몬 start/stop

결과 확인은 전부 Prometheus/Grafana 경로 하나로 통일한다 — results/*.csv를 로컬에
따로 저장하는 경로는 없다(과거엔 --save-csv가 있었으나, "로컬 CSV vs Prometheus발
CSV" 두 소스가 섞여 헷갈린다는 이유로 제거함. utils/export_metrics_from_prometheus.py
로 언제든 Prometheus DB에서 CSV를 재구성할 수 있다).

nano-vLLM과 Transformers는 같은 GPU(8GB)에 동시에 못 올라가므로(VRAM 충돌) 반드시
별도 프로세스로 떠야 한다 — 그래서 이 모듈은 "공유 로직을 import하는 대상"일 뿐,
두 엔진을 한 프로세스로 합치지는 않는다.
"""
import json
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from typing import Callable, Optional

from engine.gpu_metrics import GPUProfilerDaemon, GPUSnapshot
from engine.nanovllm_engine import RequestMetrics
from workload.dataset_loaders import (
    load_continuous_batching_prompts,
    load_cuda_graph_prompts,
    load_prefix_cache_prompts,
    load_squad_context_prompts,
)

# 시나리오(데이터셋) 이름 → 로더. 두 프로파일링 스크립트가 동일하게 쓴다.
DATASET_LOADERS = {
    "tool_catalog": load_prefix_cache_prompts,
    "squad": load_squad_context_prompts,
    "math500": load_cuda_graph_prompts,
    "kmmlu": load_continuous_batching_prompts,
}

# --dataset을 생략했을 때 쓸 technique별 기본 데이터셋 — 그 기법의 효과가 가장
# 뚜렷하게 드러나는 시나리오. 강제는 아니라서 임의 조합도 가능하다.
TECHNIQUE_DEFAULT_DATASET = {
    "cuda_graph": "math500",
    "prefix_cache": "tool_catalog",
    "continuous_batching": "kmmlu",
}

# technique(ablation 스위치)별 기본 워크로드 크기 — 그 최적화 효과가 가장 뚜렷하게
# 드러나는 조건. prefix_cache는 반드시 arrival_interval_s > 0이어야 한다: burst(0)로
# 도착시키면 요청들이 첫 prefill 배치 하나로 묶여버리고, 그 배치 안에서는 아직 어떤
# 요청도 블록 해시가 등록되기 전이라 서로의 공통 프리픽스를 캐시 히트할 기회가 없다.
TECHNIQUE_DEFAULTS = {
    "cuda_graph": {"max_output_tokens": 256, "arrival_interval_s": 0.0, "num_requests": 8},
    "prefix_cache": {"max_output_tokens": 48, "arrival_interval_s": 0.1, "num_requests": 12},
    "continuous_batching": {"max_output_tokens": 48, "arrival_interval_s": 0.05, "num_requests": 24},
}


def load_scenario(
    dataset: str,
    n: int,
    seed: int,
    tokenizer,
    target_prefix_tokens: int = 1200,
    min_math_level: int = 3,
    num_categories: int = 6,
    squad_title: Optional[str] = None,
) -> tuple[list[str], Optional[str]]:
    """
    dataset 이름만 보고 프롬프트를 로딩한다 (technique과 무관). tool_catalog/squad는
    (prompts, prefix_text)를, math500/kmmlu는 (prompts, None)을 반환한다 — prefix_text가
    있어야 "프리픽스만 잘라내기"(예: baseline의 --query-only) 같은 후처리가 가능하다.

    squad_title: [squad 전용] 무작위 문서 대신 특정 Wikipedia title로 고정한다(예:
    시연 영상에서 매번 같은 문서가 나오게). "Sexual_orientation"처럼 언더스코어
    포함 원문 title을 그대로 넘긴다.
    """
    if dataset in ("tool_catalog", "squad"):
        load_fn = DATASET_LOADERS[dataset]
        extra_kwargs = {"title": squad_title} if dataset == "squad" and squad_title else {}
        print(f"[{dataset}] 프리픽스(목표 {target_prefix_tokens}토큰) + query {n}개 로딩 중..." + (f" (title={squad_title})" if extra_kwargs else ""))
        prompts, prefix_text = load_fn(
            n, seed=seed, target_prefix_tokens=target_prefix_tokens, tokenizer=tokenizer, **extra_kwargs
        )
        prefix_tokens = len(tokenizer.encode(prefix_text))
        print(f"[{dataset}] 실제 고정 프리픽스 길이: {prefix_tokens} 토큰")
        return prompts, prefix_text

    if dataset == "math500":
        print(f"[math500] HuggingFaceH4/MATH-500에서 난이도 {min_math_level}+ 문제 {n}개 로딩 중...")
        return DATASET_LOADERS[dataset](n, seed=seed, min_level=min_math_level), None

    print(f"[kmmlu] HAERAE-HUB/KMMLU에서 {num_categories}개 과목 섞어 문제 {n}개 로딩 중...")
    return DATASET_LOADERS[dataset](n, seed=seed, num_categories=num_categories), None


def add_output_args(parser) -> None:
    """--tag/--push-metrics/--pushgateway-url — 프로파일링 스크립트 공통."""
    parser.add_argument("--tag", type=str, default=None, help="run_tag(Prometheus job 라벨)")
    parser.add_argument("--push-metrics", action="store_true", help="실측 결과를 Prometheus Pushgateway로 전송")
    parser.add_argument("--pushgateway-url", type=str, default="localhost:9091")


def add_gpu_daemon_args(parser) -> None:
    """--gpu-poll-interval-s/--gpu-history-maxlen — GPUProfilerDaemon 설정, 두 프로파일링 스크립트 공통."""
    parser.add_argument("--gpu-poll-interval-s", type=float, default=0.5, help="nvidia-smi 폴링 주기(초)")
    parser.add_argument(
        "--gpu-history-maxlen", type=int, default=600,
        help="GPU 스냅샷 히스토리 보관 개수 (기본 600 = 0.5s 간격 기준 5분)",
    )


@contextmanager
def gpu_profiling(interval_s: float = 0.5, maxlen: int = 600):
    """GPUProfilerDaemon을 start/stop으로 감싼다. 사용 후 daemon.history()로 스냅샷을 꺼낸다."""
    daemon = GPUProfilerDaemon(interval_s=interval_s, maxlen=maxlen)
    daemon.start()
    try:
        yield daemon
    finally:
        daemon.stop()


def make_live_pusher(
    run_tag: str, cache_enabled: bool, cuda_graph_enabled: bool, continuous_batching: bool, pushgateway_url: str,
    gpu_daemon: Optional[GPUProfilerDaemon] = None,
) -> tuple[Callable[[int, RequestMetrics], None], list[tuple[int, GPUSnapshot]]]:
    """
    --push-metrics일 때 run_workload*()의 on_result 콜백으로 넘길 함수를 만든다.

    gpu_daemon을 넘기면 요청이 끝날 때마다 그 순간의 GPU 스냅샷(gpu_daemon.latest())을
    request_index와 함께 누적한다 — wall-clock 시계열이 아니라 request_ttft와 똑같이
    "요청 순서"를 X축으로 삼아 Grafana Trend 패널에 그리기 위함(짧은 run에서는
    Prometheus scrape_interval보다 요청이 더 빨리 끝나 wall-clock 기준으로는 점이
    몇 개 안 찍히는 문제가 있었다).

    반환된 두 번째 값(누적 리스트)은 run이 끝난 뒤 finalize_output()에 그대로 넘겨야
    한다 — push_run_metrics()의 최종 push(PUT, job 전체 교체)가 이 값을 다시 채워 넣지
    않으면 실행 중 쌓인 시계열이 사라진다.
    """
    from engine.metrics_exporter import push_live_progress

    live_results: list[tuple[int, RequestMetrics]] = []
    live_gpu_snapshots: list[tuple[int, GPUSnapshot]] = []

    def on_result(idx: int, result: RequestMetrics) -> None:
        live_results.append((idx, result))
        if gpu_daemon is not None:
            snapshot = gpu_daemon.latest()
            if snapshot is not None:
                live_gpu_snapshots.append((idx, snapshot))
        push_live_progress(
            live_results, run_tag=run_tag,
            cache_enabled=cache_enabled, cuda_graph_enabled=cuda_graph_enabled,
            continuous_batching=continuous_batching,
            pushgateway_url=pushgateway_url,
            gpu_snapshots_by_index=live_gpu_snapshots,
        )
        print(f"  [live] req#{idx} 완료 (TTFT {result.ttft_ms:.1f}ms) → Pushgateway 전송 ({len(live_results)}건 누적)")

    return on_result, live_gpu_snapshots


def finalize_output(
    results: list[RequestMetrics],
    gpu_history: list[GPUSnapshot],
    run_tag: str,
    args,
    cache_enabled: bool,
    cuda_graph_enabled: bool,
    continuous_batching: bool,
    gpu_snapshots_by_index: Optional[list[tuple[int, GPUSnapshot]]] = None,
) -> None:
    """main()에서 반복되던 꼬리 부분: --push-metrics면 Pushgateway로 전송한다.
    결과 확인은 이 경로 하나뿐이다 — utils/export_metrics_from_prometheus.py로
    언제든 Prometheus DB에서 CSV를 재구성할 수 있다."""
    if args.push_metrics:
        from engine.metrics_exporter import push_run_metrics

        push_run_metrics(
            results, gpu_history, run_tag=run_tag,
            cache_enabled=cache_enabled, cuda_graph_enabled=cuda_graph_enabled,
            continuous_batching=continuous_batching,
            pushgateway_url=args.pushgateway_url,
            gpu_snapshots_by_index=gpu_snapshots_by_index,
        )
        print(f"\nPushgateway로 메트릭 전송 완료 ({args.pushgateway_url}, job={run_tag})")


def query_prometheus(prometheus_url: str, promql: str) -> list[dict]:
    """PromQL을 Prometheus HTTP API(/api/v1/query)로 던지고 result 배열을 그대로 반환한다."""
    url = f"{prometheus_url}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Prometheus({prometheus_url})에 연결할 수 없습니다: {e}") from e
    if payload["status"] != "success":
        raise RuntimeError(f"Prometheus 쿼리 실패: {payload}")
    return payload["data"]["result"]


def prometheus_scalar(prometheus_url: str, promql: str) -> Optional[float]:
    """PromQL 결과의 첫 값 하나만 float으로 뽑는다. 결과가 없으면(아직 스크레이프 전 등) None."""
    results = query_prometheus(prometheus_url, promql)
    return float(results[0]["value"][1]) if results else None
