"""
engine/transformers_baseline_engine.py
Bench_server/baseline.py와 동일한 방식(HF Transformers, non-paged KV 캐시, 좌측 패딩
정적 배치, eager attention)을 이 프로젝트의 워크로드(공통 SYSTEM_PREFIX 이메일)에 맞춰
재구성한 baseline 엔진. "nano-vLLM의 최적화들이 없으면 어떻게 되는가"를 보여주는
대조군이다. nano-vLLM 프로세스와 VRAM을 동시에 점유하면 충돌하므로 반드시 별도
프로세스(scripts/baseline_run.py)로 실행한다.

nano-vLLM과의 근본적인 차이:
  - KV 캐시   : 배치 전체를 하나의 연속 텐서(past_key_values)로 할당 → 블록 재사용 불가
  - Prefix Caching : 없음. 공통 SYSTEM_PREFIX라도 요청마다 처음부터 다시 prefill
  - Continuous Batching : 없음. 모든 요청을 한 번에 정적 배치로 묶어 generate() 1회 호출.
    늦게 끝나는 시퀀스가 있어도 고정 shape 텐서라 먼저 끝난 시퀀스의 연산을
    조기에 반환할 수 없다(패딩만 하고 계속 연산에 참여).
  - CUDA Graph : 없음. 매 스텝 Python + eager PyTorch 오버헤드가 그대로 노출된다.

TTFT 측정 방법:
  model.generate()는 배치 전체가 끝나야 반환되므로 완료 시각만으로는 TTFT를 알 수
  없다. 대신 LogitsProcessor 훅을 쓴다 — generate()는 새 토큰을 만들 때마다(첫
  번째 포함) logits_processor를 정확히 한 번씩 호출하므로, 그 첫 호출 시각이 곧
  "배치 전체의 첫 토큰이 나온 시각"(TTFT)이다. 배치 전체가 하나의 프리필 스텝을
  공유하므로, 이 워크로드의 모든 요청은 동일한 TTFT/latency를 갖는다 — 이것 자체가
  정적 배치의 한계를 보여주는 지표다.
"""
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from config import MODEL_PATH
from engine.nanovllm_engine import RequestMetrics


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

    def __init__(self, model_path: str = MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # flash-attn 없는 표준 scaled-dot-product
        ).cuda().eval()

    def warmup(self) -> None:
        """
        측정 시작 전 GPU 커널을 데운다 (Bench_server/baseline.py와 동일한 목적).
        cuDNN/cuBLAS 알고리즘 탐색, 첫 커널 컴파일 비용을 실제 측정 구간 밖으로 밀어낸다.
        """
        dummy = torch.full((1, 32), self.tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        mask = torch.ones_like(dummy)
        with torch.inference_mode():
            self.model.generate(dummy, attention_mask=mask, max_new_tokens=8, do_sample=False, use_cache=True)
        torch.cuda.synchronize()

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
