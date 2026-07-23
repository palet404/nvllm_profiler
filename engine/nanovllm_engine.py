"""
engine/nanovllm_engine.py
실제 nano-vLLM 엔진(nanovllm.LLM)을 로드해서 진짜 GPU 추론을 수행하는 서빙 엔진 래퍼.
Bench_server의 nanovllm_bench.py와 동일하게 실제 모델·실제 GPU 연산을 사용한다.
더미 산술 시뮬레이션은 전혀 없다 — 모든 지표는 실측값이다.

왜 LLMEngine.step()을 그대로 쓰지 않는가:
  nanovllm.engine.llm_engine.LLMEngine.step()은 아래 세 단계를 수행한다.
    1) Scheduler.schedule()      — continuous batching: prefill 우선, 없으면 decode 배치 구성
    2) ModelRunner.call("run")   — 실제 GPU forward (flash-attn prefill/decode, CUDA Graph)
    3) Scheduler.postprocess()   — 생성 토큰 반영 + prefix cache 블록 해시 등록
  하지만 step()의 반환값(outputs)에는 '이번 스텝에 완료된' 시퀀스만 담기고,
  '이번 스텝에 첫 토큰을 받은'(TTFT) 시퀀스는 알 수 없다. 그래서 이 세 단계를
  직접 호출해 매 스텝 스케줄된 시퀀스 전체를 들여다보고, 각 요청의 첫 토큰
  시각(TTFT)과 종료 시각을 실측한다.

프리픽스 캐시 히트 수치의 함정:
  Scheduler.postprocess()는 seq.num_cached_tokens를 "prefix 캐시로 재사용한
  토큰 수"로 초기화한 뒤(BlockManager.allocate), 매 스텝 seq.num_scheduled_tokens
  만큼 계속 누적시켜 "지금까지 계산된 토큰 수" 카운터로 재사용한다. 따라서 생성이
  끝날 무렵에는 이 값이 전체 토큰 수와 같아져 더 이상 '캐시 히트량'을 의미하지
  않는다. 그래서 반드시 schedule() 직후, postprocess()가 값을 덮어쓰기 전
  '이 시퀀스가 배치에 처음 등장한 시점'에 스냅샷을 떠 둬야 진짜 캐시 히트
  토큰 수를 얻을 수 있다.
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

DEFAULT_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")


@dataclass
class RequestMetrics:
    """단일 요청에 대한 실측 프로파일링 결과."""

    request_id: int
    prompt_preview: str
    output_text: str
    prompt_tokens: int
    cached_tokens: int  # 실제 BlockManager가 재사용한 KV 블록의 토큰 수
    new_prefill_tokens: int
    output_tokens: int
    prefix_cache_hit: bool
    batch_size_at_admit: int  # 요청 도착 시점에 이미 대기/실행 중이던 시퀀스 수
    ttft_ms: float
    tps: float
    latency_ms: float
    arrival_time: float


class NanoVLLMProfilerEngine:
    """
    실제 nano-vLLM(LLM)을 한 번 로드해 재사용하는 동기식 프로파일링 래퍼.
    모델 로딩(가중치 로드 + KV 캐시 프로파일링 + CUDA Graph 캡처)은
    프로세스당 1회만 수행되며 수십 초가 걸릴 수 있다.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        enforce_eager: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
    ):
        self.llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def warmup(self, max_output_tokens: int = 8) -> None:
        """
        측정 시작 전 GPU 커널을 데운다.

        엔진 생성 시점의 CUDA Graph 캡처는 decode 배치 크기별로만 이뤄지고, prefill은
        flash_attn_varlen_func를 실제 요청이 들어와야 처음 호출한다 — 이 첫 호출에서
        Triton/flash-attn 커널 JIT, cuBLAS 알고리즘 탐색 같은 1회성 초기화 비용이
        발생한다. 이걸 실제 측정 구간 안에서 흡수하면 "첫 요청만 압도적으로 느린"
        현상이 나타나 prefix caching 효과(캐시 히트 vs 미스)를 완전히 가려버린다.
        실제 워크로드의 프리픽스와 무관한 더미 프롬프트를 써서, 워밍업이 실제 워크로드의
        prefix cache 상태(어떤 블록이 이미 캐시됐는지)를 오염시키지 않게 한다.
        """
        dummy_prompt = "GPU 커널 워밍업을 위한 더미 프롬프트입니다. " * 20
        seq = self._admit(dummy_prompt, max_output_tokens)
        scheduler = self.llm.scheduler
        while not seq.is_finished:
            seqs, is_prefill = scheduler.schedule()
            token_ids = self.llm.model_runner.call("run", seqs, is_prefill)
            scheduler.postprocess(seqs, token_ids, is_prefill)

    def set_prefix_cache_enabled(self, enabled: bool) -> None:
        """
        Prefix Caching ablation 스위치.

        BlockManager.can_allocate는 해시 테이블(hash_to_block_id)을 조회해 재사용
        가능한 블록 수를 돌려준다. 이 메서드를 비활성화하면, 그 조회만 건너뛰고
        (항상 0을 반환) 여유 블록 수 검사(OOM 방지)는 원본과 동일하게 유지한
        패치 함수로 인스턴스 메서드를 갈아끼운다 — "캐시 재사용이 아예 없다면"을
        나머지 조건(블록 할당/해제, continuous batching)은 그대로 둔 채로 재현한다.
        enabled=True로 다시 부르면 원본 구현으로 복원된다.
        """
        block_manager = self.llm.scheduler.block_manager
        if not enabled:

            def _can_allocate_no_cache(seq):
                if len(block_manager.free_block_ids) < seq.num_blocks:
                    return -1
                return 0

            block_manager.can_allocate = _can_allocate_no_cache
        elif "can_allocate" in vars(block_manager):
            del block_manager.can_allocate  # 인스턴스 오버라이드 제거 → 클래스 원본 메서드로 복원

    def _admit(self, prompt: str, max_output_tokens: int) -> Sequence:
        """
        LLMEngine.add_request()는 seq_id를 돌려주지 않으므로, add_request와
        완전히 동일한 동작(Sequence 생성 후 대기열 삽입)을 직접 수행해 seq_id를 즉시 얻는다.
        ignore_eos=True: baseline(HF transformers)도 min_new_tokens로 항상 고정
        길이를 생성하도록 맞췄으므로, 조기 종료로 인한 출력 토큰 수 차이가 처리량
        비교를 왜곡하지 않도록 두 엔진 모두 고정 길이로 생성한다.
        """
        token_ids = self.llm.tokenizer.encode(prompt)
        sampling_params = SamplingParams(temperature=0.6, max_tokens=max_output_tokens, ignore_eos=True)
        seq = Sequence(token_ids, sampling_params)
        self.llm.scheduler.add(seq)
        return seq

    def _finalize_metrics(self, seq_id: int, m: dict) -> RequestMetrics:
        ttft_ms = (m["ttft_time"] - m["arrival_time"]) * 1000
        latency_ms = (m["finish_time"] - m["arrival_time"]) * 1000
        tps = m["output_tokens"] / (latency_ms / 1000) if latency_ms > 0 else 0.0
        prompt = m["prompt"]
        prompt_tokens = len(self.llm.tokenizer.encode(prompt))
        return RequestMetrics(
            request_id=seq_id,
            prompt_preview=(prompt[:40] + "…") if len(prompt) > 40 else prompt,
            output_text=m["output_text"],
            prompt_tokens=prompt_tokens,
            cached_tokens=m["cached_tokens"],
            new_prefill_tokens=prompt_tokens - m["cached_tokens"],
            output_tokens=m["output_tokens"],
            prefix_cache_hit=m["cached_tokens"] > 0,
            batch_size_at_admit=m["batch_size_at_admit"],
            ttft_ms=ttft_ms,
            tps=tps,
            latency_ms=latency_ms,
            arrival_time=m["arrival_time"],
        )

    def run_workload(
        self,
        prompts: list[str],
        max_output_tokens: int = 64,
        arrival_interval_s: float = 0.0,
        seed: int = 0,
        continuous_batching: bool = True,
        disable_prefix_cache: bool = False,
        on_result: Optional[Callable[[int, RequestMetrics], None]] = None,
    ) -> list[RequestMetrics]: # 파라미터 기본값 설정
        """
        prompts를 arrival_interval_s 간격(지터 포함)으로 실제 시간차를 두고 도착시키면서,
        실제 GPU에서 처리한다. arrival_interval_s=0이면 전부 동시(burst)에 도착한
        것으로 간주한다. 모든 요청이 끝날 때까지 동기 블로킹한다.

        continuous_batching=False: Continuous Batching ablation. 새 요청이 도착해도
        스케줄러가 완전히 빌 때까지(현재 배치가 전부 끝날 때까지) 대기열에만 쌓아두고
        admit하지 않는다 — "매 스텝 빈 슬롯을 즉시 채우는" 대신 "한 배치가 끝나야
        다음 배치를 받는" 정적 스케줄링을 재현한다. nano-vLLM 내부 구현(스케줄러,
        블록 매니저, CUDA Graph)은 전혀 건드리지 않고 admit 타이밍만 바꾼다.

        on_result: 요청이 하나 끝날 때마다(run 전체가 끝나길 기다리지 않고) 즉시
        (arrival_index, RequestMetrics)로 호출된다. Prometheus에 실시간으로
        push해서 Grafana를 라이브 데모처럼 채우는 용도(engine/metrics_exporter.py
        의 push_single_request 참고).
        """
        self.set_prefix_cache_enabled(not disable_prefix_cache)

        rng = random.Random(seed)
        arrival_offsets = []
        t = 0.0
        for _ in prompts:
            arrival_offsets.append(t)
            t += rng.uniform(0, 2 * arrival_interval_s) if arrival_interval_s > 0 else 0.0
        order = sorted(range(len(prompts)), key=lambda i: arrival_offsets[i])

        scheduler = self.llm.scheduler
        meta: dict[int, dict] = {}
        run_start = time.perf_counter()
        next_pos = 0

        while (
            next_pos < len(order)
            or scheduler.waiting
            or scheduler.running
        ):
            now = time.perf_counter() - run_start
            # continuous_batching=False일 때는 admit할 때마다 매번 다시 검사해야 한다.
            # 이 조건을 while 밖에서 한 번만 계산해 두면, 첫 admit 직후 스케줄러가
            # 비어있지 않게 됐는데도 같은 tick 안에서 계속 다음 요청을 admit해버려
            # "순차 도착"이 무의미해진다(전부 한꺼번에 들어가 버림).
            while (
                next_pos < len(order)
                and arrival_offsets[order[next_pos]] <= now
                and (continuous_batching or not (scheduler.waiting or scheduler.running))
            ):
                i = order[next_pos]
                batch_size_at_admit = len(scheduler.waiting) + len(scheduler.running) # "이 새 요청이 들어올 때, 시스템(스케줄러)이 이미 떠안고 있던 시퀀스 수" — 즉 GPU 자원을 두고 경쟁하는 총 동시 요청 수
                seq = self._admit(prompts[i], max_output_tokens)
                meta[seq.seq_id] = {
                    "arrival_index": next_pos,
                    "prompt": prompts[i],
                    "arrival_time": time.perf_counter(),
                    "batch_size_at_admit": batch_size_at_admit,
                    "cached_tokens": None,
                    "ttft_time": None,
                    "finish_time": None,
                    "output_tokens": 0,
                    "output_text": "",
                }
                next_pos += 1

            if not (scheduler.waiting or scheduler.running):
                continue  # 다음 요청 도착 대기 (burst가 아닐 때만 일시적으로 발생)

            # ── LLMEngine.step()과 동일한 3단계를 직접 수행 ──
            seqs, is_prefill = scheduler.schedule()

            # postprocess가 num_cached_tokens를 덮어쓰기 전, 이번에 처음 등장한
            # 시퀀스의 '진짜' 프리픽스 캐시 히트 토큰 수를 스냅샷으로 남긴다.
            for seq in seqs:
                m = meta[seq.seq_id] # seq_id로 이 요청의 "메타데이터 딕셔너리" 조회
                if m["cached_tokens"] is None: # 아직 스냅샷 안 떴으면 (=이 시퀀스가 배치에 처음 등장)
                    m["cached_tokens"] = seq.num_cached_tokens  # BlockManager.allocate()가 방금 세팅한 값을 기록

            token_ids = self.llm.model_runner.call("run", seqs, is_prefill)
            scheduler.postprocess(seqs, token_ids, is_prefill)
            step_end = time.perf_counter()

            for seq in seqs:
                m = meta[seq.seq_id] # 딕셔너리임 {int, dict(value)}
                if m["ttft_time"] is None and seq.num_completion_tokens >= 1:
                    m["ttft_time"] = step_end
                if seq.is_finished:
                    m["finish_time"] = step_end
                    m["output_tokens"] = seq.num_completion_tokens
                    m["output_text"] = self.llm.tokenizer.decode(seq.completion_token_ids)
                    if on_result is not None:
                        on_result(m["arrival_index"], self._finalize_metrics(seq.seq_id, m))

        return [self._finalize_metrics(seq_id, m) for seq_id, m in meta.items()]
