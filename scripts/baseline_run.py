"""
scripts/baseline_run.py
engine/transformers_baseline_engine.py(HF Transformers, non-paged KV 캐시, 정적
패딩 배치, eager attention)를 실제 GPU에 로드해서 이 프로젝트의 워크로드(공통
SYSTEM_PREFIX 이메일)를 흘려보내는 테스트 스크립트. nano-vLLM과 동일한 워크로드
생성기·RequestMetrics 스키마·CSV 형식을 공유해 scripts/profile_run.py 결과와
바로 비교할 수 있다.

nano-vLLM 프로세스와 동시에 실행하면 VRAM이 충돌하므로 반드시 별도 프로세스로
실행한다 (scripts/run_comparison_suite.py도 subprocess로 분리해서 돌린다).

실행 예:
    python scripts/baseline_run.py --num-requests 10 --tag baseline_transformers --save-csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from engine.gpu_metrics import GPUProfilerDaemon
from engine.nanovllm_engine import RequestMetrics
from engine.transformers_baseline_engine import TransformersBaselineEngine
from workload.generator import generate_prompts

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF Transformers baseline 실측 실행기")
    parser.add_argument("--num-requests", type=int, default=10, help="생성할 이메일 분석 요청 수")
    parser.add_argument("--seed", type=int, default=42, help="워크로드 생성 시드")
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        default=0.3,
        help="워크로드 생성기 일관성을 위한 옵션(baseline 자체 동작에는 영향 없음 — prefix caching이 없음)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=48, help="요청당 생성 토큰 수")
    parser.add_argument("--long-prefix", action="store_true", help="config.LONG_SYSTEM_PREFIX 사용 (nano-vLLM 쪽과 동일 워크로드로 맞출 때)")
    parser.add_argument("--save-csv", action="store_true", help="results/ 에 CSV로 결과 저장")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="지정하면 results/<tag>.csv로 고정 파일명 저장 (비교 스위트가 모드를 식별하는 데 사용)",
    )
    parser.add_argument(
        "--push-metrics",
        action="store_true",
        help="실측 결과를 Prometheus Pushgateway로 전송 (job=--tag 또는 자동 생성된 태그)",
    )
    parser.add_argument(
        "--pushgateway-url",
        type=str,
        default="localhost:9091",
        help="Pushgateway 주소 (--push-metrics와 함께 사용)",
    )
    return parser.parse_args()


def run_profiling(args: argparse.Namespace) -> tuple[list[RequestMetrics], list]:
    system_prefix = config.LONG_SYSTEM_PREFIX if args.long_prefix else config.SYSTEM_PREFIX
    prompts = generate_prompts(
        args.num_requests, seed=args.seed, duplicate_ratio=args.duplicate_ratio, system_prefix=system_prefix
    )

    gpu_daemon = GPUProfilerDaemon()
    gpu_daemon.start()

    print("HF Transformers baseline 모델 로딩 중 (non-paged KV 캐시, eager attention)...")
    t0 = time.perf_counter()
    engine = TransformersBaselineEngine()
    print(f"모델 로딩 완료 ({time.perf_counter() - t0:.1f}s)")

    print("GPU 커널 워밍업 중 (측정 구간에서 콜드스타트 비용을 제외하기 위함)...")
    engine.warmup()
    print("워밍업 완료\n")

    print(f"정적 배치 {len(prompts)}건을 한 번에 generate() 호출 중...")
    results = engine.run_workload(prompts, max_output_tokens=args.max_output_tokens)

    gpu_daemon.stop()
    return results, gpu_daemon.history()


def summarize(results: list[RequestMetrics], gpu_history: list) -> None:
    df = pd.DataFrame(
        [
            {
                "id": r.request_id,
                "prompt_tok": r.prompt_tokens,
                "out_tok": r.output_tokens,
                "ttft_ms": round(r.ttft_ms, 2),
                "tps": round(r.tps, 1),
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ]
    )

    print("=== 요청별 프로파일링 결과 (실측, baseline) ===")
    print(df.to_string(index=False))

    total_output_tokens = sum(r.output_tokens for r in results)
    latency_s = results[0].latency_ms / 1000  # 정적 배치라 전원 동일

    print("\n=== 정적 배치 요약 (실측) ===")
    print(f"  배치 크기 : {len(results)} (전원 동시 도착 → 동일 TTFT/latency)")
    print(f"  TTFT (배치 공통) : {results[0].ttft_ms:.2f} ms")
    print(f"  전체 처리량(집계 TPS) : {total_output_tokens} tok / {latency_s:.2f}s "
          f"≈ {total_output_tokens / latency_s:.1f} tok/s")

    print("\n=== GPU 실측 스냅샷 (nvidia-smi) ===")
    if gpu_history:
        utils = [s.gpu_util_pct for s in gpu_history]
        mems = [s.mem_util_pct for s in gpu_history]
        print(f"  샘플 수 : {len(gpu_history)}")
        print(f"  GPU Util 평균/최대 : {sum(utils)/len(utils):.1f}% / {max(utils):.1f}%")
        print(f"  VRAM Util 평균/최대 : {sum(mems)/len(mems):.1f}% / {max(mems):.1f}%")
    else:
        print("  nvidia-smi를 사용할 수 없어 GPU 스냅샷을 수집하지 못했습니다.")


def save_csv(results: list[RequestMetrics], tag: str | None) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    filename = f"{tag}.csv" if tag else f"baseline_run_{int(time.time())}.csv"
    out_path = RESULTS_DIR / filename
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    results, gpu_history = run_profiling(args)
    print()
    summarize(results, gpu_history)

    if args.save_csv or args.tag:
        path = save_csv(results, args.tag)
        print(f"\nCSV 저장 완료: {path}")

    if args.push_metrics:
        from engine.metrics_exporter import push_run_metrics

        run_tag = args.tag or f"baseline_run_{int(time.time())}"
        # baseline은 prefix caching/CUDA Graph/continuous batching이 전혀 없다.
        push_run_metrics(
            results,
            gpu_history,
            run_tag=run_tag,
            cache_enabled=False,
            cuda_graph_enabled=False,
            continuous_batching=False,
            pushgateway_url=args.pushgateway_url,
        )
        print(f"\nPushgateway로 메트릭 전송 완료 ({args.pushgateway_url}, job={run_tag})")


if __name__ == "__main__":
    main()
