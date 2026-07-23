"""
utils/export_metrics_from_prometheus.py
--push-metrics로 Pushgateway → Prometheus에 쌓인 run_tag 하나의 실측치를 PromQL로
조회해 CSV로 뽑아낸다. results/*.csv(로컬 저장본)를 다시 읽는 게 아니라, "Prometheus
DB가 실제로 갖고 있는 값"을 소스로 삼는다 — Pushgateway/Prometheus 경로가 실제로
동작한다는 걸 보여줄 때(발표 자료, 데모 등) 쓴다. profile_nanovllm.py/
profile_transformers_baseline.py처럼 "실행" 스크립트가 아니라 사후 분석용 유틸이라
scripts/ 밖의 최상위 utils/ 아래에 따로 둔다.

가져올 수 있는 것과 없는 것:
  - 요청별 ttft_ms/latency_ms/tps/batch_size_at_admit/prompt_tokens/output_tokens:
    request_index 라벨로 Gauge가 개별 저장돼 있어(engine/metrics_exporter.py) 그대로
    복원 가능. prompt_tokens/output_tokens는 병목 재현용 — TTFT/latency가 튄
    request_index를 찾으면 그 프롬프트/출력 길이 조합으로 바로 재현할 수 있다.
    (nanovllm_prompt_tokens_total/nanovllm_output_tokens_total은 별개의 누적
    Counter라 요청 단위로 못 쪼갠다 — 그건 집계 CSV의 prompt_tokens_total로만 쓴다)
  - GPU 사용률: run 전체 평균 Gauge 하나뿐이라 시계열은 못 뽑는다 — 집계 CSV에
    평균값 한 줄로만 들어간다

실행 예:
    python utils/export_metrics_from_prometheus.py --run-tag demo_squad_sexual_orientation
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import prometheus_scalar, query_prometheus

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# request_index 라벨로 저장된 요청별 Gauge — engine/metrics_exporter.py의 per_request_label_names 참고
PER_REQUEST_METRICS = {
    "ttft_ms": "nanovllm_request_ttft_milliseconds",
    "latency_ms": "nanovllm_request_latency_milliseconds",
    "tps": "nanovllm_request_tps",
    "batch_size_at_admit": "nanovllm_request_batch_size_at_admit",
    # 병목 재현용 — 그 request_index의 입력/출력 토큰 shape을 그대로 남겨서, TTFT/latency가
    # 튄 요청을 찾으면 바로 같은 길이의 프롬프트로 재현할 수 있게 한다.
    "prompt_tokens": "nanovllm_request_prompt_tokens",
    "output_tokens": "nanovllm_request_output_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prometheus에 push된 run_tag 하나의 실측치를 PromQL로 조회해 CSV로 저장")
    parser.add_argument("--run-tag", required=True, help="profile_nanovllm.py/profile_transformers_baseline.py에 준 --tag")
    parser.add_argument("--prometheus-url", type=str, default="http://localhost:9090")
    parser.add_argument("--out", type=str, default=None, help="미지정 시 results/<run_tag>_from_prometheus.csv")
    return parser.parse_args()


def fetch_per_request(prometheus_url: str, run_tag: str) -> list[dict]:
    """request_index별 ttft/latency/tps/batch_size_at_admit을 모아 index 기준으로 합친다."""
    rows: dict[int, dict] = {}
    for column, metric in PER_REQUEST_METRICS.items():
        for r in query_prometheus(prometheus_url, f'{metric}{{run_tag="{run_tag}"}}'):
            idx = int(r["metric"]["request_index"])
            rows.setdefault(idx, {"request_index": idx})[column] = float(r["value"][1])
    return [rows[idx] for idx in sorted(rows)]


def fetch_aggregate(prometheus_url: str, run_tag: str) -> dict:
    def avg(metric: str) -> Optional[float]:
        return prometheus_scalar(
            prometheus_url,
            f'{metric}_sum{{run_tag="{run_tag}"}} / {metric}_count{{run_tag="{run_tag}"}}',
        )

    return {
        "avg_ttft_ms": avg("nanovllm_ttft_milliseconds"),
        "avg_latency_ms": avg("nanovllm_latency_milliseconds"),
        "aggregate_tps": prometheus_scalar(prometheus_url, f'nanovllm_aggregate_tokens_per_second{{run_tag="{run_tag}"}}'),
        "gpu_util_pct_mean": prometheus_scalar(prometheus_url, f'nanovllm_gpu_util_percent{{run_tag="{run_tag}"}}'),
        "gpu_mem_util_pct_mean": prometheus_scalar(prometheus_url, f'nanovllm_gpu_mem_util_percent{{run_tag="{run_tag}"}}'),
        "prompt_tokens_total": prometheus_scalar(prometheus_url, f'nanovllm_prompt_tokens_total{{run_tag="{run_tag}"}}'),
        "output_tokens_total": prometheus_scalar(prometheus_url, f'nanovllm_output_tokens_total{{run_tag="{run_tag}"}}'),
    }


def main() -> None:
    args = parse_args()

    per_request = fetch_per_request(args.prometheus_url, args.run_tag)
    if not per_request:
        print(
            f"[경고] run_tag='{args.run_tag}'에 대한 request_index별 메트릭이 없습니다. "
            "--push-metrics로 실행했는지, run_tag 철자가 맞는지 확인하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.run_tag}_from_prometheus.csv"
    fieldnames = ["request_index", "ttft_ms", "latency_ms", "tps", "batch_size_at_admit", "prompt_tokens", "output_tokens"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_request)
    print(f"요청별 CSV 저장 완료: {out_path} ({len(per_request)}행)")

    aggregate = fetch_aggregate(args.prometheus_url, args.run_tag)
    agg_path = out_path.with_name(out_path.stem + "_aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    print(f"집계 CSV 저장 완료: {agg_path}")
    print()
    for k, v in aggregate.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
