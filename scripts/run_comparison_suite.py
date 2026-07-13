"""
scripts/run_comparison_suite.py
nano-vLLM의 세 가지 최적화(Prefix Caching / Continuous Batching / CUDA Graph) 각각의
기여도와, nano-vLLM 전체 스택 vs HF Transformers baseline의 차이를 실측해 비교표로
뽑아내는 오케스트레이터.

두 개의 독립된 스위트로 나뉜다 (섞으면 안 되는 이유는 아래 "설계 노트" 참고):

  [A] TTFT 순수 비교 스위트 (전부 --sequential, 요청을 하나씩 순차 도착시킴)
      요청 1개의 TTFT/latency만 비교하고 싶을 때, 여러 요청이 우연히 한 스텝에
      배치로 묶이면 그 병렬성 이득이 섞여 들어가 ablation 결과가 왜곡된다.
      그래서 Prefix Caching / CUDA Graph ablation은 반드시 순차 도착으로 재서
      "배치 구성"이라는 변수를 없앤다.
        1) nanovllm_full_ttft            — 전부 ON, 순차 도착
        2) nanovllm_no_cuda_graph_ttft    — CUDA Graph만 OFF, 순차 도착
        3) nanovllm_no_prefix_cache_ttft  — Prefix Caching만 OFF, 순차 도착
        4) baseline_transformers_ttft     — HF Transformers, 요청 1개만 (배치 크기 1)

  [B] 처리량 비교 스위트 (동시 도착, Continuous Batching의 효과를 보기 위함)
        1) nanovllm_full_throughput                  — 전부 ON, 동시 도착
        2) nanovllm_no_continuous_batching_throughput — Continuous Batching만 OFF
           (= --sequential. "한 번에 하나씩"이 그 자체로 이 ablation의 정의다)
        3) baseline_transformers_throughput          — HF Transformers, 정적 배치 전체

설계 노트 (왜 나뉘었는가):
  처음에는 5개 모드를 전부 동일한 동시-도착 워크로드로 돌렸는데, prefix caching을
  꺼도 두 번째 요청부터 급격히 빨라지는 이상한 결과가 나왔다. 원인은 캐시가 없어
  첫 요청의 prefill이 오래 걸리는 동안(수백 ms) 나머지 요청이 전부 도착해 버려서,
  스케줄러가 그것들을 한 스텝에 몰아 배치로 처리했기 때문이다 — "캐시 없음의 비용"이
  아니라 "배치로 묶였을 때의 GPU 병렬성 이득"이 측정된 것이다. 그래서 TTFT를 순수하게
  재는 ablation(prefix cache, CUDA Graph)은 반드시 순차 도착으로 분리해야 한다.
  반대로 continuous batching의 효과는 애초에 "동시에 도착한 요청들을 얼마나 잘
  겹쳐서 처리하는가"이므로 동시 도착 워크로드가 맞다.

nano-vLLM과 transformers를 같은 프로세스에서 동시에 올리면(8GB GPU 기준) VRAM이
충돌하므로 매 모드를 별도 서브프로세스로 띄운다.

실행 예:
    python scripts/run_comparison_suite.py --num-requests 8 --max-output-tokens 24
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PYTHON = sys.executable

TTFT_MODES = [
    {"tag": "nanovllm_full_ttft", "label": "① nano-vLLM 전체 (순차, TTFT 순수 측정)",
     "script": "profile_run.py", "extra_flags": ["--sequential"]},
    {"tag": "nanovllm_no_cuda_graph_ttft", "label": "② CUDA Graph OFF (순차)",
     "script": "profile_run.py", "extra_flags": ["--sequential", "--enforce-eager"]},
    {"tag": "nanovllm_no_prefix_cache_ttft", "label": "③ Prefix Caching OFF (순차)",
     "script": "profile_run.py", "extra_flags": ["--sequential", "--disable-prefix-cache"]},
    {"tag": "baseline_transformers_ttft", "label": "④ HF Transformers baseline (요청 1개)",
     "script": "baseline_run.py", "extra_flags": [], "num_requests_override": 1},
]

THROUGHPUT_MODES = [
    {"tag": "nanovllm_full_throughput", "label": "① nano-vLLM 전체 (동시 도착)",
     "script": "profile_run.py", "extra_flags": []},
    {"tag": "nanovllm_no_continuous_batching_throughput", "label": "② Continuous Batching OFF",
     "script": "profile_run.py", "extra_flags": ["--sequential"]},
    {"tag": "baseline_transformers_throughput", "label": "③ HF Transformers baseline (정적 배치 전체)",
     "script": "baseline_run.py", "extra_flags": []},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-vLLM ablation + transformers baseline 비교 스위트")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-ratio", type=float, default=0.3)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--skip-baseline", action="store_true", help="HF Transformers baseline 모드들을 생략")
    return parser.parse_args()


def run_mode(mode: dict, args: argparse.Namespace) -> bool:
    num_requests = mode.get("num_requests_override", args.num_requests)
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "scripts" / mode["script"]),
        "--num-requests", str(num_requests),
        "--seed", str(args.seed),
        "--duplicate-ratio", str(args.duplicate_ratio),
        "--max-output-tokens", str(args.max_output_tokens),
        "--tag", mode["tag"],
        "--save-csv",
        *mode["extra_flags"],
    ]
    print(f"\n{'='*70}\n{mode['label']}  ({mode['tag']})\n{'='*70}")
    print(" ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"[{mode['tag']}] 종료 (elapsed {time.perf_counter() - t0:.1f}s, returncode={result.returncode})")
    return result.returncode == 0


def summarize(mode_tag: str) -> dict | None:
    csv_path = RESULTS_DIR / f"{mode_tag}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    wall_clock_s = (df["arrival_time"] + df["latency_ms"] / 1000).max() - df["arrival_time"].min()
    total_output_tokens = df["output_tokens"].sum()
    return {
        "tag": mode_tag,
        "avg_ttft_ms": df["ttft_ms"].mean(),
        "avg_latency_ms": df["latency_ms"].mean(),
        "cache_hit_rate": df["prefix_cache_hit"].mean() if "prefix_cache_hit" in df else 0.0,
        "aggregate_tps": total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0,
    }


def print_comparison_table(title: str, rows: list[dict], baseline_tag: str) -> None:
    if not rows:
        print(f"\n[{title}] 비교할 결과가 없습니다.")
        return

    baseline_row = next((r for r in rows if r["tag"] == baseline_tag), rows[0])

    print(f"\n{'='*95}")
    print(f"=== {title} ({baseline_tag} 기준 배수) ===")
    print(f"{'='*95}")
    header = f"{'모드':45} {'평균 TTFT(ms)':>15} {'평균 Latency(ms)':>18} {'집계 TPS':>12} {'TTFT 배수':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ttft_ratio = r["avg_ttft_ms"] / baseline_row["avg_ttft_ms"] if baseline_row["avg_ttft_ms"] > 0 else float("nan")
        print(
            f"{r['tag']:45} {r['avg_ttft_ms']:15.2f} {r['avg_latency_ms']:18.2f} "
            f"{r['aggregate_tps']:12.1f} {ttft_ratio:9.2f}x"
        )
    print(
        f"\n※ TTFT 배수 = 해당 모드 평균 TTFT / {baseline_tag} 평균 TTFT "
        "(1.0x보다 크면 그만큼 느려짐 = 그 최적화가 꺼졌을 때의 손실)"
    )


def run_suite(modes: list[dict], args: argparse.Namespace) -> list[dict]:
    modes = [m for m in modes if not (args.skip_baseline and "baseline" in m["tag"])]
    for mode in modes:
        ok = run_mode(mode, args)
        if not ok:
            print(f"[경고] {mode['tag']} 실행이 실패했습니다 (returncode != 0). 비교표에서 제외합니다.")
    return [s for m in modes if (s := summarize(m["tag"])) is not None]


def main() -> None:
    args = parse_args()

    ttft_rows = run_suite(TTFT_MODES, args)
    throughput_rows = run_suite(THROUGHPUT_MODES, args)

    print_comparison_table("TTFT 순수 비교 (순차 도착)", ttft_rows, baseline_tag="nanovllm_full_ttft")
    print_comparison_table("처리량 비교 (동시 도착)", throughput_rows, baseline_tag="nanovllm_full_throughput")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(ttft_rows).to_csv(RESULTS_DIR / "comparison_ttft_summary.csv", index=False)
    pd.DataFrame(throughput_rows).to_csv(RESULTS_DIR / "comparison_throughput_summary.csv", index=False)
    print(f"\n요약 CSV 저장: {RESULTS_DIR / 'comparison_ttft_summary.csv'}")
    print(f"요약 CSV 저장: {RESULTS_DIR / 'comparison_throughput_summary.csv'}")


if __name__ == "__main__":
    main()
