"""
engine/transformers_baseline_engine.py
HF Transformers로 "nano-vLLM의 최적화들이 없으면 어떻게 되는가"를 보여주는 대조군
엔진. nano-vLLM 프로세스와 VRAM을 동시에 점유하면 충돌하므로 반드시 별도 프로세스
(scripts/profile_transformers_baseline.py)로 실행한다.

기본 실행 방식 — "Variant A": sdpa attention + 수동 prefill/decode 루프
  원래는 eager attention + model.generate() 호출이었지만, torch.compile+StaticCache로
  CUDA Graph급 최적화를 시도하는 과정(아래 use_compile 설명)에서 (1) generate() 호출
  오버헤드를 없애고 nano-vLLM처럼 prefill 1회 + decode를 직접 루프 도는 방식, (2) eager
  대신 PyTorch 내장 fused 커널인 sdpa로 바꾸는 것만으로 — **compile 없이, 크래시 위험
  없이** — latency가 확연히 줄어드는 걸 실측으로 확인했다(2,000토큰대 프리픽스 기준
  1468ms → 1085ms대). "그래도 안전하게 낼 수 있는 최선의 Transformers"와 nano-vLLM을
  비교하는 게 "아무것도 안 건드린 eager baseline"보다 더 정확한 대조군이라고 판단해
  이걸 기본값으로 삼았다.

  generate()를 안 쓰므로 LogitsProcessor 훅 없이 TTFT를 직접 잰다 — prefill 호출
  (model(input_ids=..., use_cache=True))이 끝난 시점이 곧 첫 토큰이 나온 시점이다.

nano-vLLM과 남아있는 근본적인 차이 (Variant A로 바꿔도 그대로):
  - KV 캐시   : 요청마다 새로 시작하는 DynamicCache → 블록 재사용 불가
  - Prefix Caching : 없음. 공통 프리픽스라도 요청마다 처음부터 다시 prefill
  - Continuous Batching : 없음. 항상 배치 크기 1로 순차 처리
  - CUDA Graph : 없음. sdpa는 fused 커널이지만 그래프로 캡처/재생하지는 않는다

use_compile=True (실험적, 현재 깨져 있음 — 기본 baseline과 무관):
  HF 공식 가이드는 torch.compile(mode="reduce-overhead", 내부적으로 CUDA Graph 사용)
  + StaticCache + sdpa 조합을 권장하지만, 이 환경(PyTorch + Transformers 5.12.1 +
  Qwen3)에서는 4가지 방식(generate()+reduce-overhead, generate()+기본모드,
  model 전체 compile, 수동 루프+기본모드) 전부 dynamo가 그래프를 끊었다 재개하는
  지점에서 hidden_states를 잃어버리는 동일한 라이브러리 버그로 크래시하는 걸
  실측으로 확인했다(TypeError: 'NoneType' object is not subscriptable,
  modeling_qwen3.py의 torch_dynamo_resume_in_forward_at_492). 라이브러리 버그라
  이 프로젝트 코드로 우회할 방법이 없어 미해결로 남겨둔 실험적 옵션이다.
"""
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from engine.nanovllm_engine import RequestMetrics

DEFAULT_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")


class _TTFTRecorder(LogitsProcessor):
    """generate()가 새 토큰을 만들 때마다 호출되는 훅. 첫 호출 시각 = 배치 전체 TTFT."""

    def __init__(self):
        self.first_call_time: Optional[float] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.first_call_time is None:
            self.first_call_time = time.perf_counter()
        return scores


class TransformersBaselineEngine:
    """HF Transformers 표준 KV 캐시 기반 정적 배치 엔진 (nano-vLLM의 baseline 대조군)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, use_compile: bool = False, static_cache_max_len: int = 8192):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_compile = use_compile
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",  # nano-vLLM과 동일한 FlashAttention 커널 (Variant A 기본값)
        ).cuda().eval()

        if use_compile:
            # StaticCache + torch.compile로 decode 스텝을 그래프화한다 — nano-vLLM의
            # capture_cudagraph()와 같은 종류의 최적화를 Transformers 쪽에도 적용한 것.
            # 입력 shape가 매번 같아야 그래프가 재사용되므로, 호출부(run_workload_sequential)가
            # fixed_prompt_len으로 항상 동일한 길이로 패딩해야 한다.
            #
            # mode="reduce-overhead"(CUDA Graph)는 이 환경(PyTorch + Transformers 5.12.1 +
            # Qwen3)에서 실측 확인됨 — StaticCache의 cumulative_length 버퍼가 CPU에 남아
            # "skipping cudagraphs due to cpu device" 경고가 뜨고, 이어서 lm_head 단계에서
            # hidden_states가 None이 되는 TypeError로 크래시했다. fullgraph=True도 Transformers
            # 5.x의 forward 데코레이터(co_varnames 동적 검사)를 dynamo가 못 추적해 실패했다.
            # 그래서 기본 모드(Inductor, CUDA Graph 없음)만 쓴다 — Python/디스패치 오버헤드
            # 감소는 기대할 수 있지만 nano-vLLM 수준의 CUDA Graph replay는 아니다.
            #
            # model.forward = torch.compile(model.forward)도 동일한 지점(lm_head 앞
            # hidden_states=None)에서 mode 무관하게 크래시했다 — model.forward를 감싸는
            # 대신 모듈 전체를 감싸면(torch.compile(self.model)) dynamo 진입점이 달라져
            # 이 버그를 피해갈 수도 있어 대안으로 시도한다. OptimizedModule은 generate()
            # 등 원본 메서드를 _orig_mod로 위임하므로 겉보기 인터페이스는 동일하다.
            self.model.generation_config.cache_implementation = "static"
            self.model.generation_config.max_length = static_cache_max_len
            self.model = torch.compile(self.model)

    def _sample_token(self, logits: torch.Tensor, temperature: float = 0.6) -> torch.Tensor:
        """nano-vLLM _admit()과 동일한 샘플링 조건(temperature=0.6)으로 다음 토큰 하나를 뽑는다."""
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def warmup(self, fixed_prompt_len: Optional[int] = None) -> None:
        """
        측정 시작 전 GPU 커널을 데운다 (Bench_server/baseline.py와 동일한 목적).
        cuDNN/cuBLAS 알고리즘 탐색, 첫 커널 컴파일 비용을 실제 측정 구간 밖으로 밀어낸다.
        prefill 1회 + decode 1회만 돌려 sdpa 커널을 데운다(generate() 안 씀).

        use_compile=True일 때는 fixed_prompt_len을 실제 run에서 쓸 값과 동일하게 줘야
        한다 — 그래야 이 워밍업 호출에서 트리거된 컴파일이 실제 요청과 같은 shape라
        재사용되고, 측정 구간 첫 요청에서 또 재컴파일이 일어나지 않는다.
        """
        length = fixed_prompt_len if fixed_prompt_len is not None else 32
        dummy = torch.full((1, length), self.tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = self.model(input_ids=dummy, use_cache=True)
            next_id = self._sample_token(out.logits[:, -1, :])
            self.model(input_ids=next_id, past_key_values=out.past_key_values, use_cache=True)
        torch.cuda.synchronize()

    def run_workload_sequential(
        self,
        prompts: list[str],
        max_output_tokens: int = 64,
        on_result: Optional[Callable[[int, RequestMetrics], None]] = None,
        fixed_prompt_len: Optional[int] = None,
    ) -> list[RequestMetrics]:
        """
        prompts를 하나씩 순서대로 처리한다(배치 크기 1) — model.generate()를 거치지
        않고 prefill 1회 + decode를 직접 루프 도는 "Variant A" 방식(모듈 docstring
        참고). past_key_values는 매 요청 새로 시작하므로 공통 프리픽스라도 요청마다
        처음부터 다시 prefill한다 — Prefix Caching이 없는 서빙 루프를 그대로
        재현한 것이다. nano-vLLM의 Prefix Caching ablation(순차 도착)과 요청 단위로
        1:1 비교하기 위한 메서드.

        on_result: 요청이 하나 끝날 때마다 즉시 (request_index, RequestMetrics)로
        호출된다. nano-vLLM 쪽과 동일하게 Prometheus 실시간 push용
        (engine/metrics_exporter.py의 push_single_request 참고).

        fixed_prompt_len: use_compile=True일 때만 의미가 있다(현재 컴파일 자체가
        라이브러리 버그로 깨져 있어 실질적으로 미사용). 모든 프롬프트를 이 길이로
        좌측 패딩해 매 호출의 입력 shape를 동일하게 고정한다. prompt_tokens 메트릭은
        패딩 전 실제 토큰 수(real_len)를 기록해 다른 run과 비교 가능하게 한다.
        """
        results = []
        for i, prompt in enumerate(prompts):
            token_ids = self.tokenizer.encode(prompt)
            real_len = len(token_ids)
            if fixed_prompt_len is not None:
                if real_len > fixed_prompt_len:
                    raise ValueError(f"prompt({real_len}tok)이 fixed_prompt_len({fixed_prompt_len})보다 깁니다.")
                pad_id = self.tokenizer.pad_token_id
                token_ids = [pad_id] * (fixed_prompt_len - real_len) + token_ids

            input_ids = torch.tensor([token_ids], dtype=torch.long).cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.model(input_ids=input_ids, use_cache=True)
                torch.cuda.synchronize()
                ttft_time = time.perf_counter()

                next_id = self._sample_token(out.logits[:, -1, :])
                past_key_values = out.past_key_values
                output_ids = [next_id.item()]

                for _ in range(max_output_tokens - 1):
                    out = self.model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    next_id = self._sample_token(out.logits[:, -1, :])
                    output_ids.append(next_id.item())
            torch.cuda.synchronize()
            t_finish = time.perf_counter()

            ttft_ms = (ttft_time - t0) * 1000
            latency_ms = (t_finish - t0) * 1000
            tps = max_output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

            result = RequestMetrics(
                request_id=i,
                prompt_preview=(prompt[:40] + "…") if len(prompt) > 40 else prompt,
                output_text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
                prompt_tokens=real_len,
                cached_tokens=0,  # baseline은 요청 간 prefix caching이 없음 — 항상 전량 재계산
                new_prefill_tokens=real_len,
                output_tokens=max_output_tokens,
                prefix_cache_hit=False,
                batch_size_at_admit=0,  # 항상 순차 처리 — 동시에 실행 중인 다른 요청 없음
                ttft_ms=ttft_ms,
                tps=tps,
                latency_ms=latency_ms,
                arrival_time=t0,
            )
            results.append(result)
            if on_result is not None:
                on_result(i, result)
        return results

    def run_workload(self, prompts: list[str], max_output_tokens: int = 64) -> list[RequestMetrics]:
        """
        모든 prompts를 좌측 패딩해 하나의 정적 배치로 묶고 model.generate()를 1회
        호출한다. continuous batching이 없으므로 '요청 도착 간격' 개념 자체가 없다 —
        전부 동시에 도착해서 동시에 배치로 묶인다고 가정한다.
        """
        encodings = [self.tokenizer.encode(p) for p in prompts]
        prompt_lens = [len(ids) for ids in encodings]
        pad_id = self.tokenizer.pad_token_id

        max_len = max(prompt_lens)
        input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
        for i, ids in enumerate(encodings):
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, max_len - len(ids):] = 1
        input_ids = input_ids.cuda()
        attn_mask = attn_mask.cuda()

        recorder = _TTFTRecorder()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_output_tokens,
                min_new_tokens=max_output_tokens,  # nano-vLLM의 ignore_eos=True와 동일하게 고정 길이로 맞춤
                do_sample=False,
                use_cache=True,
                logits_processor=LogitsProcessorList([recorder]),
            )
        torch.cuda.synchronize()
        t_finish = time.perf_counter()

        ttft_ms = (recorder.first_call_time - t0) * 1000
        latency_ms = (t_finish - t0) * 1000
        tps = max_output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
        batch_size = len(prompts)

        results = []
        for i, prompt in enumerate(prompts):
            output_ids = output[i, max_len:].tolist()
            results.append(
                RequestMetrics(
                    request_id=i,
                    prompt_preview=(prompt[:40] + "…") if len(prompt) > 40 else prompt,
                    output_text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
                    prompt_tokens=prompt_lens[i],
                    cached_tokens=0,  # baseline은 prefix caching이 없음 — 항상 전량 재계산
                    new_prefill_tokens=prompt_lens[i],
                    output_tokens=max_output_tokens,
                    prefix_cache_hit=False,
                    batch_size_at_admit=batch_size - 1,  # 정적 배치라 전원이 동시에 도착
                    ttft_ms=ttft_ms,
                    tps=tps,
                    latency_ms=latency_ms,
                    arrival_time=t0,
                )
            )
        return results
