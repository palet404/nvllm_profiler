"""
scripts/decode_length_sweep.py
max_output_tokens(디코드 구간 길이)를 늘려가면서, prefill-bound인 TTFT/Prefix
Caching과 decode-bound인 CUDA Graph 최적화의 상대적 기여도가 어떻게 변하는지
실측한다.

가설:
  - TTFT는 prefill 시간이라 max_output_tokens와 거의 무관해야 한다.
  - latency = TTFT + decode_steps * time_per_step 이므로, 출력이 길어질수록
    latency에서 Prefix Caching의 절감폭(=TTFT 절감분)이 차지하는 비중은
    점점 작아진다(희석). 반대로 CUDA Graph는 decode 스텝마다 절약되므로,
    출력이 길어질수록 총 latency 절감폭(스텝 수 × 스텝당 절감)이 커진다.

한 변형(variant)당 엔진을 1번만 로드하고 max_output_tokens 여러 값을 순회하며
반복 측정해 로딩 비용을 amortize한다. CUDA Graph on/off만 엔진 재초기화가
필요해서 별도 프로세스(graph / eager)로 분리한다 — graph 프로세스 안에서
prefix cache on/off까지 같이 재서(엔진 런타임 토글이라 재초기화 불필요) 실행
횟수를 최소화한다.

실행 예:
    python scripts/decode_length_sweep.py --output-lengths 8,32,64,128,256
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from engine.nanovllm_engine import NanoVLLMProfilerEngine
from workload.generator import generate_prompts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="디코드 길이에 따른 CUDA Graph / Prefix Caching 기여도 스윕")
    parser.add_argument("--output-lengths", type=str, default="8,32,64,128,256")
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-ratio", type=float, default=0.0)
    parser.add_argument(
        "--long-prefix", action=argparse.BooleanOptionalAction, default=True,
        help="LONG_SYSTEM_PREFIX 사용 (기본 True — 캐시 효과가 보이는 규모에서 희석 여부를 봐야 하므로). "
        "--no-long-prefix로 끌 수 있음",
    )
    parser.add_argument("--variant", choices=["graph", "eager"], default=None, help="내부용 — 지정하면 워커 모드로 실행")
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    system_prefix = config.LONG_SYSTEM_PREFIX if args.long_prefix else config.SYSTEM_PREFIX
    prompts = generate_prompts(
        args.num_requests, seed=args.seed, duplicate_ratio=args.duplicate_ratio, system_prefix=system_prefix
    )
    lengths = [int(x) for x in args.output_lengths.split(",")]
    enforce_eager = args.variant == "eager"

    print(f"[{args.variant}] 엔진 로딩 중 (enforce_eager={enforce_eager})...")
    t0 = time.perf_counter()
    engine = NanoVLLMProfilerEngine(enforce_eager=enforce_eager)
    engine.warmup()
    print(f"[{args.variant}] 로딩+워밍업 완료 ({time.perf_counter() - t0:.1f}s)")

    rows = []
    for T in lengths:
        results = engine.run_workload(prompts, max_output_tokens=T, continuous_batching=False)
        for r in results:
            rows.append(
                {
                    "variant": args.variant,
                    "max_output_tokens": T,
                    "request_id": r.request_id,
                    "prefix_cache_hit": r.prefix_cache_hit,
                    "ttft_ms": r.ttft_ms,
                    "tps": r.tps,
                    "latency_ms": r.latency_ms,
                    "output_tokens": r.output_tokens,
                }
            )
        avg_ttft = sum(r.ttft_ms for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        print(f"[{args.variant}]         T={T:4d}  avg_ttft={avg_ttft:7.2f}ms  avg_latency={avg_latency:9.2f}ms")

        if args.variant == "graph":
            # 같은 엔진(CUDA Graph 켜진 상태)으로 prefix cache만 꺼서, 캐싱의
            # latency 기여도가 출력 길이에 따라 희석되는지도 같이 확인한다.
            results_nc = engine.run_workload(
                prompts, max_output_tokens=T, continuous_batching=False, disable_prefix_cache=True
            )
            for r in results_nc:
                rows.append(
                    {
                        "variant": "graph_no_cache",
                        "max_output_tokens": T,
                        "request_id": r.request_id,
                        "prefix_cache_hit": r.prefix_cache_hit,
                        "ttft_ms": r.ttft_ms,
                        "tps": r.tps,
                        "latency_ms": r.latency_ms,
                        "output_tokens": r.output_tokens,
                    }
                )
            avg_ttft_nc = sum(r.ttft_ms for r in results_nc) / len(results_nc)
            avg_latency_nc = sum(r.latency_ms for r in results_nc) / len(results_nc)
            print(
                f"[graph_no_cache] T={T:4d}  avg_ttft={avg_ttft_nc:7.2f}ms  avg_latency={avg_latency_nc:9.2f}ms"
            )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"decode_sweep_{args.variant}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{args.variant}] 저장: {out_path}")


def analyze() -> None:
    graph_path = RESULTS_DIR / "decode_sweep_graph.csv"
    eager_path = RESULTS_DIR / "decode_sweep_eager.csv"
    if not graph_path.exists() or not eager_path.exists():
        print("결과 CSV가 없습니다.")
        return

    df = pd.concat([pd.read_csv(graph_path), pd.read_csv(eager_path)], ignore_index=True)
    summary = (
        df.groupby(["variant", "max_output_tokens"])
        .agg(
            avg_ttft_ms=("ttft_ms", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_tps=("tps", "mean"),
            cache_hit_rate=("prefix_cache_hit", "mean"),
        )
        .reset_index()
    )

    print(f"\n{'='*100}\n=== 디코드 길이(max_output_tokens)에 따른 지표 변화 ===\n{'='*100}")
    print(summary.to_string(index=False))

    pivot_latency = summary[summary.variant.isin(["graph", "eager"])].pivot(
        index="max_output_tokens", columns="variant", values="avg_latency_ms"
    )
    pivot_latency["cuda_graph_saving_ms"] = pivot_latency["eager"] - pivot_latency["graph"]
    pivot_latency["cuda_graph_speedup_x"] = pivot_latency["eager"] / pivot_latency["graph"]
    print(f"\n{'='*100}\n=== CUDA Graph 절감폭: 출력이 길어질수록 커지는가? ===\n{'='*100}")
    print(pivot_latency.to_string())

    pivot_cache = summary[summary.variant.isin(["graph", "graph_no_cache"])].pivot(
        index="max_output_tokens", columns="variant", values="avg_latency_ms"
    )
    pivot_cache["prefix_cache_saving_ms"] = pivot_cache["graph_no_cache"] - pivot_cache["graph"]
    pivot_cache["saving_share_of_latency_pct"] = (
        100 * pivot_cache["prefix_cache_saving_ms"] / pivot_cache["graph_no_cache"]
    )
    print(f"\n{'='*100}\n=== Prefix Caching의 latency 기여도: 출력이 길어질수록 희석되는가? ===\n{'='*100}")
    print(pivot_cache.to_string())

    summary_path = RESULTS_DIR / "decode_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n요약 CSV: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.variant:
        run_worker(args)
        return

    for variant in ["graph", "eager"]:
        cmd = [
            PYTHON, str(Path(__file__).resolve()),
            "--variant", variant,
            "--output-lengths", args.output_lengths,
            "--num-requests", str(args.num_requests),
            "--seed", str(args.seed),
            "--duplicate-ratio", str(args.duplicate_ratio),
            "--long-prefix" if args.long_prefix else "--no-long-prefix",
        ]
        print(f"\n{'='*70}\n[{variant}] 서브프로세스 실행\n{'='*70}")
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f"[{variant}] 종료 (elapsed {time.perf_counter() - t0:.1f}s)")

    analyze()


if __name__ == "__main__":
    main()
