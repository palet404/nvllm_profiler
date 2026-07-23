"""
scripts/profile_nanovllm.py
실제 nano-vLLM 엔진(engine/nanovllm_engine.py)을 공개 벤치마크 데이터셋으로 실측한다.
(구 scripts/profile_real_datasets.py — nano-vLLM을 구동한다는 게 이름에 드러나도록 개명)

--technique(어떤 ablation 스위치를 켜고 끌지: enforce_eager/disable_prefix_cache/
sequential)와 --dataset(어떤 시나리오를 재생할지)은 서로 독립이다. --dataset을
생략하면 그 technique의 효과가 가장 뚜렷하게 드러나는 기본 데이터셋
(scripts/_common.py의 TECHNIQUE_DEFAULT_DATASET)을 쓰지만, 원하는 technique×dataset
조합을 자유롭게 고를 수 있다.

  tool_catalog : glaiveai/glaive-function-calling-v2 (고정 tool 카탈로그 프리픽스 + query)
  squad        : rajpurkar/squad (같은 문서 프리픽스 + 여러 질문)
  math500      : HuggingFaceH4/MATH-500 (긴 풀이가 필요한 수학 문제)
  kmmlu        : HAERAE-HUB/KMMLU (여러 과목을 섞어 도메인/길이가 제각각인 동시 요청)

실행 예:
    python scripts/profile_nanovllm.py --technique cuda_graph --num-requests 8 --max-output-tokens 256
    python scripts/profile_nanovllm.py --technique cuda_graph --num-requests 8 --max-output-tokens 256 --enforce-eager

    python scripts/profile_nanovllm.py --technique prefix_cache --num-requests 12
    python scripts/profile_nanovllm.py --technique prefix_cache --num-requests 12 --disable-prefix-cache
    python scripts/profile_nanovllm.py --technique prefix_cache --dataset squad --num-requests 12

    python scripts/profile_nanovllm.py --technique continuous_batching --num-requests 24 --arrival-interval-s 0.05
    python scripts/profile_nanovllm.py --technique continuous_batching --num-requests 24 --arrival-interval-s 0.05 --sequential
    python scripts/profile_nanovllm.py --technique continuous_batching --dataset tool_catalog --num-requests 24

결과 확인은 --push-metrics로 Prometheus Pushgateway에 보내는 경로 하나뿐이다 — 로컬
CSV 저장 옵션은 없다. utils/export_metrics_from_prometheus.py로 언제든 Prometheus
DB에서 CSV를 재구성할 수 있다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from engine.nanovllm_engine import NanoVLLMProfilerEngine, RequestMetrics
from scripts._common import (
    DATASET_LOADERS,
    TECHNIQUE_DEFAULT_DATASET,
    TECHNIQUE_DEFAULTS,
    add_gpu_daemon_args,
    add_output_args,
    finalize_output,
    gpu_profiling,
    load_scenario,
    make_live_pusher,
)

DEFAULT_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실제 벤치마크 데이터셋으로 nano-vLLM 세 가지 최적화 실측")
    parser.add_argument(
        "--technique",
        choices=list(TECHNIQUE_DEFAULTS),
        required=True,
        help="테스트할 ablation 스위치(enforce_eager/disable_prefix_cache/sequential)와 "
             "num_requests/max_output_tokens/arrival_interval_s 기본값을 결정",
    )
    parser.add_argument(
        "--dataset", choices=list(DATASET_LOADERS), default=None,
        help="재생할 시나리오. 미지정 시 --technique의 기본 데이터셋 사용. "
             "--technique과 독립적으로 아무 조합이나 지정 가능",
    )
    parser.add_argument("--num-requests", type=int, default=None, help="미지정 시 기법별 기본값 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=None, help="미지정 시 기법별 기본값 사용")
    parser.add_argument("--arrival-interval-s", type=float, default=None, help="미지정 시 기법별 기본값 사용")
    parser.add_argument("--min-math-level", type=int, default=3, help="[math500] MATH-500 난이도 하한(1~5)")
    parser.add_argument(
        "--target-prefix-tokens", type=int, default=1200,
        help="[tool_catalog/squad] 고정 프리픽스의 목표 토큰 수",
    )
    parser.add_argument("--num-categories", type=int, default=6, help="[kmmlu] 섞어 뽑을 과목 수")
    parser.add_argument(
        "--squad-title", type=str, default=None,
        help="[squad 전용] 무작위 문서 대신 특정 Wikipedia title로 고정 (예: Sexual_orientation)",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="CUDA Graph ablation: eager 모드로 실행")
    parser.add_argument(
        "--disable-prefix-cache", action="store_true", help="Prefix Caching ablation: 블록 해시 재사용을 강제로 끔"
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Continuous Batching ablation: 배치가 완전히 비어야 다음 요청 admit"
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="로드할 모델 경로")
    parser.add_argument("--max-model-len", type=int, default=8192, help="nano-vLLM KV 캐시가 수용할 최대 시퀀스 길이")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="nano-vLLM이 예약할 VRAM 비율")
    add_gpu_daemon_args(parser)
    add_output_args(parser)
    return parser.parse_args()


def run_profiling(args: argparse.Namespace, run_tag: str) -> tuple[list[RequestMetrics], list, list]:
    with gpu_profiling(interval_s=args.gpu_poll_interval_s, maxlen=args.gpu_history_maxlen) as gpu_daemon:
        print("nano-vLLM 엔진 로딩 중..." + ("" if args.enforce_eager else " (CUDA Graph 캡처 포함)"))
        t0 = time.perf_counter()
        engine = NanoVLLMProfilerEngine(
            model_path=args.model_path,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(f"엔진 로딩 완료 ({time.perf_counter() - t0:.1f}s)")

        prompts, _ = load_scenario(
            args.dataset, args.num_requests, args.seed, engine.llm.tokenizer,
            target_prefix_tokens=args.target_prefix_tokens,
            min_math_level=args.min_math_level,
            num_categories=args.num_categories,
            squad_title=args.squad_title,
        )

        print("GPU 커널 워밍업 중...")
        engine.warmup()
        print("워밍업 완료\n")

        on_result = None
        gpu_snapshots_by_index: list = []
        if args.push_metrics:
            on_result, gpu_snapshots_by_index = make_live_pusher(
                run_tag,
                cache_enabled=not args.disable_prefix_cache,
                cuda_graph_enabled=not args.enforce_eager,
                continuous_batching=not args.sequential,
                pushgateway_url=args.pushgateway_url,
                gpu_daemon=gpu_daemon,
            )

        results = engine.run_workload(
            prompts,
            max_output_tokens=args.max_output_tokens,
            arrival_interval_s=args.arrival_interval_s,
            seed=args.seed,
            continuous_batching=not args.sequential,
            disable_prefix_cache=args.disable_prefix_cache,
            on_result=on_result,
        )
        gpu_history = gpu_daemon.history()

    return results, gpu_history, gpu_snapshots_by_index


def summarize(results: list[RequestMetrics], gpu_history: list, label: str) -> None:
    df = pd.DataFrame(
        [
            {
                "id": r.request_id,
                "cache_hit": r.prefix_cache_hit,
                "prompt_tok": r.prompt_tokens,
                "cached_tok": r.cached_tokens,
                "new_tok": r.new_prefill_tokens,
                "out_tok": r.output_tokens,
                "ttft_ms": round(r.ttft_ms, 2),
                "tps": round(r.tps, 1),
                "latency_ms": round(r.latency_ms, 1),
                "batch_at_admit": r.batch_size_at_admit,
            }
            for r in results
        ]
    )

    print(f"=== [{label}] 요청별 프로파일링 결과 (실측, 실제 데이터셋) ===")
    print(df.to_string(index=False))

    total_output_tokens = sum(r.output_tokens for r in results)
    wall_clock_s = max(r.arrival_time + r.latency_ms / 1000 for r in results) - min(
        r.arrival_time for r in results
    )
    aggregate_tps = total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    print(f"\n평균 latency: {df['latency_ms'].mean():.1f} ms   평균 TPS: {df['tps'].mean():.1f}")
    print(f"집계 처리량: {total_output_tokens} tok / {wall_clock_s:.2f}s ≈ {aggregate_tps:.1f} tok/s")
    print(f"최대 동시 배치 크기: {df['batch_at_admit'].max() + 1}")

    hit_df = df[df["cache_hit"]]
    miss_df = df[~df["cache_hit"]]
    if not hit_df.empty and not miss_df.empty and hit_df["ttft_ms"].mean() > 0:
        speedup = miss_df["ttft_ms"].mean() / hit_df["ttft_ms"].mean()
        print(f"Prefix Cache 히트 {len(hit_df)}/{len(df)} — TTFT 단축 배수: {speedup:.1f}x")

    if gpu_history:
        utils = [s.gpu_util_pct for s in gpu_history]
        mems = [s.mem_util_pct for s in gpu_history]
        print(
            f"\nGPU Util 평균/최대: {sum(utils)/len(utils):.1f}% / {max(utils):.1f}%   "
            f"VRAM Util 평균/최대: {sum(mems)/len(mems):.1f}% / {max(mems):.1f}%"
        )
    else:
        print("\nnvidia-smi를 사용할 수 없어 GPU 스냅샷을 수집하지 못했습니다.")


def main() -> None:
    args = parse_args()
    defaults = TECHNIQUE_DEFAULTS[args.technique]
    if args.num_requests is None:
        args.num_requests = defaults["num_requests"]
    if args.max_output_tokens is None:
        args.max_output_tokens = defaults["max_output_tokens"]
    if args.arrival_interval_s is None:
        args.arrival_interval_s = defaults["arrival_interval_s"]
    if args.dataset is None:
        args.dataset = TECHNIQUE_DEFAULT_DATASET[args.technique]

    run_tag = args.tag or f"nanovllm_{args.technique}_{args.dataset}_{int(time.time())}"
    results, gpu_history, gpu_snapshots_by_index = run_profiling(args, run_tag)
    print()
    summarize(results, gpu_history, f"{args.technique}/{args.dataset}")

    finalize_output(
        results, gpu_history, run_tag, args,
        cache_enabled=not args.disable_prefix_cache,
        cuda_graph_enabled=not args.enforce_eager,
        continuous_batching=not args.sequential,
        gpu_snapshots_by_index=gpu_snapshots_by_index,
    )


if __name__ == "__main__":
    main()
