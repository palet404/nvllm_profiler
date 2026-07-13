"""
scripts/profile_run.py
실제 nano-vLLM 엔진(engine/nanovllm_engine.py)을 실제 GPU에 로드해서 워크로드를
흘려보내고, TTFT/TPS/Latency 실측치와 nvidia-smi GPU 실측치를 함께 확인하는
테스트 스크립트. 더미/시뮬레이션 없이 전부 실제 추론 결과다.

nano-vLLM의 LLMEngine은 동기식(step()이 GPU 연산이 끝날 때까지 블로킹)이라
asyncio가 필요 없다 — 요청 도착 스케줄링과 GPU 프로파일링 데몬(별도 스레드)만
동시에 돌아가고, 추론 자체는 메인 스레드에서 순차적으로 step()을 반복한다.

--enforce-eager / --disable-prefix-cache / --sequential 플래그로 nano-vLLM의 세 가지
최적화(CUDA Graph / Prefix Caching / Continuous Batching)를 각각 개별적으로 끌 수
있다 — 어떤 최적화가 얼마나 기여하는지 ablation으로 확인하기 위함
(scripts/run_comparison_suite.py가 이 플래그들을 조합해 자동으로 비교표를 만든다).

실행 예:
    python scripts/profile_run.py --num-requests 20 --duplicate-ratio 0.4
    python scripts/profile_run.py --num-requests 8 --enforce-eager        # CUDA Graph ablation
    python scripts/profile_run.py --num-requests 8 --disable-prefix-cache # Prefix Caching ablation
    python scripts/profile_run.py --num-requests 8 --sequential           # Continuous Batching ablation
"""
import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가해 config/workload/engine을 절대 임포트로 사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from engine.gpu_metrics import GPUProfilerDaemon
from engine.nanovllm_engine import NanoVLLMProfilerEngine, RequestMetrics
from workload.generator import generate_prompts

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-vLLM 실측 프로파일링 파이프라인 테스트 실행기")
    parser.add_argument("--num-requests", type=int, default=20, help="생성할 이메일 분석 요청 수")
    parser.add_argument("--seed", type=int, default=42, help="워크로드 생성 시드")
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        default=0.3,
        help="이미 생성된 이메일을 그대로 재전송할 확률 (전체 프롬프트가 100%% 캐시 히트하는 케이스 포함)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=48, help="요청당 생성 토큰 수")
    parser.add_argument(
        "--long-prefix",
        action="store_true",
        help="config.LONG_SYSTEM_PREFIX(~1000토큰, 캐시 블록 3개)를 사용 "
        "— 짧은 프리픽스에서 고정 오버헤드에 묻혔던 Prefix Caching TTFT 효과를 재검증",
    )
    parser.add_argument(
        "--arrival-interval-s",
        type=float,
        default=0.05,
        help="요청 간 평균 도착 간격(초). 0이면 전부 동시에 제출(burst).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="CUDA Graph ablation: 캡처를 생략하고 eager 모드로 실행",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Prefix Caching ablation: 블록 해시 재사용을 강제로 끔 (매 요청 전량 재계산)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Continuous Batching ablation: 이전 배치가 완전히 끝나야 다음 요청을 admit함",
    )
    parser.add_argument("--save-csv", action="store_true", help="results/ 에 CSV로 결과 저장")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="지정하면 results/<tag>.csv로 고정 파일명 저장 (비교 스위트가 모드를 식별하는 데 사용)",
    )
    return parser.parse_args()


def run_profiling(args: argparse.Namespace) -> tuple[list[RequestMetrics], list]:
    system_prefix = config.LONG_SYSTEM_PREFIX if args.long_prefix else config.SYSTEM_PREFIX
    prompts = generate_prompts(
        args.num_requests, seed=args.seed, duplicate_ratio=args.duplicate_ratio, system_prefix=system_prefix
    )

    gpu_daemon = GPUProfilerDaemon()
    gpu_daemon.start()

    print("nano-vLLM 엔진 로딩 중 (가중치 로드 + KV 캐시 프로파일링"
          + ("" if args.enforce_eager else " + CUDA Graph 캡처") + ")...")
    t0 = time.perf_counter()
    engine = NanoVLLMProfilerEngine(enforce_eager=args.enforce_eager)
    print(f"엔진 로딩 완료 ({time.perf_counter() - t0:.1f}s)")

    print("GPU 커널 워밍업 중 (측정 구간에서 콜드스타트 비용을 제외하기 위함)...")
    engine.warmup()
    print("워밍업 완료\n")

    results = engine.run_workload(
        prompts,
        max_output_tokens=args.max_output_tokens,
        arrival_interval_s=args.arrival_interval_s,
        seed=args.seed,
        continuous_batching=not args.sequential,
        disable_prefix_cache=args.disable_prefix_cache,
    )

    gpu_daemon.stop()
    return results, gpu_daemon.history()


def summarize(results: list[RequestMetrics], gpu_history: list) -> None:
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

    print("=== 요청별 프로파일링 결과 (실측) ===")
    print(df.to_string(index=False))

    hit_df = df[df["cache_hit"]]
    miss_df = df[~df["cache_hit"]]

    print("\n=== Prefix Caching 효과 요약 (실측) ===")
    print(f"  캐시 히트 요청 수 : {len(hit_df)} / {len(df)}")
    if not hit_df.empty:
        print(f"  캐시 히트 평균 TTFT : {hit_df['ttft_ms'].mean():.2f} ms")
    if not miss_df.empty:
        print(f"  캐시 미스 평균 TTFT : {miss_df['ttft_ms'].mean():.2f} ms")
    if not hit_df.empty and not miss_df.empty and hit_df["ttft_ms"].mean() > 0:
        speedup = miss_df["ttft_ms"].mean() / hit_df["ttft_ms"].mean()
        print(f"  TTFT 단축 배수 : {speedup:.1f}x")

    total_output_tokens = sum(r.output_tokens for r in results)
    wall_clock_s = (
        max(r.arrival_time + r.latency_ms / 1000 for r in results)
        - min(r.arrival_time for r in results)
    )
    aggregate_tps = total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    print("\n=== Continuous Batching 요약 (실측) ===")
    print(f"  동시 처리된 최대 배치 크기 : {df['batch_at_admit'].max() + 1}")
    print(f"  전체 처리량(집계 TPS) : {total_output_tokens} tok / {wall_clock_s:.2f}s ≈ {aggregate_tps:.1f} tok/s")

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
    filename = f"{tag}.csv" if tag else f"profile_run_{int(time.time())}.csv"
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


if __name__ == "__main__":
    main()
