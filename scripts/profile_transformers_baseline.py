"""
scripts/profile_transformers_baseline.py
HF Transformers baseline 엔진(engine/transformers_baseline_engine.py)을 공개 벤치마크
데이터셋으로 실측한다 — "nano-vLLM의 최적화들이 전혀 없으면 어떻게 되는가"를 보여주는
대조군. (구 scripts/profile_real_datasets_baseline.py — Transformers를 구동한다는 게
이름에 드러나도록 개명, 데이터셋도 tool_catalog/squad뿐 아니라 math500/kmmlu까지 지원
하도록 확장해서 scripts/profile_nanovllm.py의 어떤 --dataset과도 1:1 비교 가능)

요청을 하나씩 순차 처리한다(TransformersBaselineEngine.run_workload_sequential,
"Variant A": flash_attention_2 + prefill/decode 수동 루프, generate() 안 씀) —
continuous batching도, prefix caching도, CUDA Graph도 없다. 같은 --dataset/--seed로
scripts/profile_nanovllm.py와 동일한 프롬프트 집합을 만들 수 있어 요청 단위 1:1
비교가 가능하다.

nano-vLLM 프로세스와 동시에 실행하면 VRAM이 충돌하므로 반드시 별도 프로세스로 실행한다.

실행 예:
    python scripts/profile_transformers_baseline.py --dataset tool_catalog --num-requests 12 --tag baseline_prefix_cache --push-metrics
    python scripts/profile_transformers_baseline.py --dataset math500 --num-requests 8 --max-output-tokens 256
    python scripts/profile_transformers_baseline.py --dataset tool_catalog --num-requests 12 --query-only --push-metrics

결과 확인은 Prometheus/Grafana 경로 하나뿐이다 — 로컬 CSV 저장 옵션은 없다.
utils/export_metrics_from_prometheus.py로 언제든 Prometheus DB에서 CSV를 재구성할 수 있다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from engine.nanovllm_engine import RequestMetrics
from engine.transformers_baseline_engine import TransformersBaselineEngine
from scripts._common import (
    DATASET_LOADERS,
    add_gpu_daemon_args,
    add_output_args,
    finalize_output,
    gpu_profiling,
    load_scenario,
    make_live_pusher,
)

DEFAULT_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공개 벤치마크 데이터셋으로 HF Transformers baseline 실측")
    parser.add_argument(
        "--dataset", choices=list(DATASET_LOADERS), default="tool_catalog",
        help="tool_catalog=glaive-function-calling-v2, squad=rajpurkar/squad, "
             "math500=HuggingFaceH4/MATH-500, kmmlu=HAERAE-HUB/KMMLU",
    )
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument(
        "--target-prefix-tokens", type=int, default=1200, help="[tool_catalog/squad] 고정 프리픽스의 목표 토큰 수"
    )
    parser.add_argument("--min-math-level", type=int, default=3, help="[math500] MATH-500 난이도 하한(1~5)")
    parser.add_argument("--num-categories", type=int, default=6, help="[kmmlu] 섞어 뽑을 과목 수")
    parser.add_argument(
        "--squad-title", type=str, default=None,
        help="[squad 전용] 무작위 문서 대신 특정 Wikipedia title로 고정 (예: Sexual_orientation). "
        "시연 영상처럼 매번 같은 문서가 나와야 할 때 사용",
    )
    parser.add_argument(
        "--query-only", action="store_true",
        help="[tool_catalog/squad 전용] 프리픽스를 붙이지 않고 같은 query만 실행한다 (프리픽스 재계산 비용을 "
        "분리하는 대조군). math500/kmmlu처럼 프리픽스가 없는 데이터셋에는 효과 없음.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="[실험적, 현재 라이브러리 버그로 크래시함] StaticCache + torch.compile을 추가로 켠다. "
        "기본(이 플래그 없이)은 이미 flash_attention_2 + 수동 prefill/decode 루프(Variant A, "
        "generate() 안 씀)이며, 여기에 compile까지 얹으려는 시도인데 Transformers 5.12.1의 "
        "Qwen3 forward가 dynamo 그래프 재개 지점에서 깨지는 버그로 4가지 방식 모두 실패했다 "
        "(engine/transformers_baseline_engine.py 모듈 docstring 참고). 재현/추가 조사용으로만 남겨둠.",
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="로드할 모델 경로")
    add_gpu_daemon_args(parser)
    add_output_args(parser)
    return parser.parse_args()


def run_profiling(args: argparse.Namespace, run_tag: str) -> tuple[list[RequestMetrics], list, list]:
    with gpu_profiling(interval_s=args.gpu_poll_interval_s, maxlen=args.gpu_history_maxlen) as gpu_daemon:
        print(
            "HF Transformers 모델 로딩 중 ("
            + ("StaticCache + torch.compile (실험적, 현재 라이브러리 버그로 크래시) + flash_attention_2" if args.compile else "flash_attention_2, 수동 prefill/decode 루프 [Variant A]")
            + ")..."
        )
        t0 = time.perf_counter()
        engine = TransformersBaselineEngine(model_path=args.model_path, use_compile=args.compile)
        print(f"모델 로딩 완료 ({time.perf_counter() - t0:.1f}s)")

        prompts, prefix_text = load_scenario(
            args.dataset, args.num_requests, args.seed, engine.tokenizer,
            target_prefix_tokens=args.target_prefix_tokens,
            min_math_level=args.min_math_level,
            num_categories=args.num_categories,
            squad_title=args.squad_title,
        )

        if args.query_only:
            if prefix_text is None:
                print(f"[{args.dataset}] --query-only는 프리픽스가 있는 데이터셋(tool_catalog/squad)에만 의미가 있어 무시합니다.")
            else:
                # 프리픽스 텍스트만 앞에서 잘라내 같은 request_id가 같은 query를
                # 가리키게 한다 — 프리픽스 유무만 다른 대조군.
                prompts = [p[len(prefix_text):] for p in prompts]
                print(f"[{args.dataset}] --query-only: 프리픽스를 제거하고 query만 실행")

        fixed_prompt_len = None
        if args.compile:
            # torch.compile(reduce-overhead)이 캡처한 CUDA Graph를 매 요청 재사용하려면
            # 입력 shape가 항상 같아야 한다 — 이번 run에서 가장 긴 프롬프트에 맞춰
            # 전부 좌측 패딩한다.
            fixed_prompt_len = max(len(engine.tokenizer.encode(p)) for p in prompts)
            print(f"[--compile] 모든 프롬프트를 {fixed_prompt_len}토큰으로 좌측 패딩해 그래프 shape 고정")

        print("GPU 커널 워밍업 중..." + (" (첫 컴파일/그래프 캡처 포함 — 시간이 걸릴 수 있음)" if args.compile else ""))
        engine.warmup(fixed_prompt_len=fixed_prompt_len)
        print("워밍업 완료\n")

        on_result = None
        gpu_snapshots_by_index: list = []
        if args.push_metrics:
            on_result, gpu_snapshots_by_index = make_live_pusher(
                run_tag, cache_enabled=False, cuda_graph_enabled=args.compile, continuous_batching=False,
                pushgateway_url=args.pushgateway_url, gpu_daemon=gpu_daemon,
            )

        print(f"{len(prompts)}건을 하나씩 순차 처리 중 (prefill+decode 수동 루프, 요청 간 KV 캐시 재사용 없음)...")
        results = engine.run_workload_sequential(
            prompts, max_output_tokens=args.max_output_tokens, on_result=on_result, fixed_prompt_len=fixed_prompt_len,
        )
        gpu_history = gpu_daemon.history()

    return results, gpu_history, gpu_snapshots_by_index


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

    print("=== [transformers baseline, 순차] 요청별 프로파일링 결과 (실측, 실제 데이터셋) ===")
    print(df.to_string(index=False))

    total_output_tokens = sum(r.output_tokens for r in results)
    wall_clock_s = max(r.arrival_time + r.latency_ms / 1000 for r in results) - min(
        r.arrival_time for r in results
    )
    aggregate_tps = total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    print(f"\n평균 TTFT: {df['ttft_ms'].mean():.2f} ms   평균 latency: {df['latency_ms'].mean():.1f} ms")
    print(f"집계 처리량: {total_output_tokens} tok / {wall_clock_s:.2f}s ≈ {aggregate_tps:.1f} tok/s")

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
    run_tag = args.tag or f"transformers_baseline_{args.dataset}_{int(time.time())}"
    results, gpu_history, gpu_snapshots_by_index = run_profiling(args, run_tag)
    print()
    summarize(results, gpu_history)

    finalize_output(
        results, gpu_history, run_tag, args,
        cache_enabled=False, cuda_graph_enabled=args.compile, continuous_batching=False,
        gpu_snapshots_by_index=gpu_snapshots_by_index,
    )


if __name__ == "__main__":
    main()
