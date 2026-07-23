"""
workload/dataset_loaders.py
nano-vLLM의 세 가지 최적화(Prefix Caching/CUDA Graph/Continuous Batching)가
가장 뚜렷하게 드러나는 세 가지 실제 시나리오를, 공개 벤치마크 데이터셋에서
그대로 가져와 재현한다.

  - CUDA Graph (긴 decode 길이)
        HuggingFaceH4/MATH-500 — 풀이가 긴 수학 문제. decode 길이 자체는
        엔진이 ignore_eos=True로 고정하므로 --max-output-tokens가 결정하지만,
        "실제로 긴 추론을 요구하는 프롬프트"여야 그 워크로드가 현실적이다.

  - Prefix Caching (긴 시스템 프롬프트)
        두 가지 변형을 제공한다. 둘 다 "공통 프리픽스 + 가변 요청" 구조는 동일하지만,
        프리픽스의 성격이 다르다.

        1) glaiveai/glaive-function-calling-v2 (load_prefix_cache_prompts) — row마다
           딸려 있는 "system"(함수 스키마 목록) 하나를 고정 프리픽스로 쓰고, 그 뒤에
           서로 다른 row의 첫 USER 발화(query)를 이어붙인다. "모든 요청에 항상 똑같이
           붙는 tool/function 카탈로그 시스템 프롬프트"(OpenAI/Anthropic 스타일
           function-calling API) 시나리오를 재현한다. RAG는 아니다 — 검색된 컨텍스트가
           아니라 정적으로 나열된 함수 정의 목록이다.

        2) rajpurkar/squad (load_squad_context_prompts) — 같은 Wikipedia 문서(title)의
           문단(context)들을 이어붙여 "검색해서 가져온 문서" 역할의 고정 프리픽스로
           쓰고, 그 문서에 실제로 달려 있는 서로 다른 question들을 가변 요청으로
           붙인다. "문서를 한 번 검색해 컨텍스트로 고정해두고, 같은 세션에서 그
           문서에 대해 여러 질문을 연달아 받는" RAG 세션을 재현한다 — 진짜 RAG처럼
           질의마다 top-k 검색 결과가 달라지면 프리픽스가 요청마다 바뀌어 애초에
           캐시가 재사용될 수 없으므로, "동일 문서 재사용"이 성립하는 세션형 RAG로
           범위를 좁힌 것이다.

  - Continuous Batching (다양한 길이의 동시 요청)
        HAERAE-HUB/KMMLU — 여러 과목(config)에서 무작위로 섞어 뽑아, 실제
        서비스처럼 도메인/질문 길이가 제각각인 요청들을 대량으로 흘려보낸다.

셋 다 HuggingFace `datasets` 패키지가 필요하다 (pip install datasets).
"""
import json
import random
import re
from collections import defaultdict
from typing import Optional


def _require_datasets():
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "이 로더는 HuggingFace `datasets` 패키지가 필요합니다. `pip install datasets`로 설치하세요."
        ) from e
    return datasets


def load_cuda_graph_prompts(n: int, seed: int = 42, min_level: int = 3) -> list[str]:
    """
    CUDA Graph(긴 decode) 검증용: HuggingFaceH4/MATH-500 수학 문제.

    min_level(1~5) 이상만 골라 실제로 풀이가 길어질 문제 위주로 뽑는다.
    """
    datasets = _require_datasets()
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.filter(lambda ex: ex["level"] >= min_level)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if len(indices) < n:
        raise ValueError(f"level>={min_level} 문제가 {len(indices)}개뿐이라 n={n}개를 채울 수 없습니다.")

    prompts = []
    for i in indices[:n]:
        problem = ds[i]["problem"]
        prompts.append(
            f"다음 수학 문제를 단계별로 풀이 과정을 모두 보이며 풀어라.\n\n문제: {problem}\n\n풀이:"
        )
    return prompts


_USER_TURN_RE = re.compile(r"USER:\s*(.*?)\s*\n\n\nASSISTANT:", re.DOTALL)
_CATALOG_HEADER = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"


def _extract_user_query(chat: str) -> str | None:
    """glaive-function-calling-v2의 'chat' 필드에서 첫 USER 발화만 뽑아낸다."""
    m = _USER_TURN_RE.search(chat)
    return m.group(1).strip() if m else None


def _extract_functions(system_text: str) -> list[dict]:
    """
    'system' 필드 안에 박혀 있는 함수 정의 JSON 객체들을 중괄호 매칭으로 추출한다.
    (row 하나에는 보통 함수 1~2개뿐이라, 여러 row를 훑어 모아야 카탈로그가 커진다.)
    """
    functions = []
    pos = system_text.find("{")
    while pos != -1:
        depth = 0
        end = None
        for j in range(pos, len(system_text)):
            if system_text[j] == "{":
                depth += 1
            elif system_text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        try:
            functions.append(json.loads(system_text[pos : end + 1]))
        except json.JSONDecodeError:
            pass
        pos = system_text.find("{", end + 1)
    return functions


def _build_tool_catalog(ds, target_tokens: int, tokenizer, rng: random.Random, max_scan: int = 3000) -> str:
    """
    여러 row에서 서로 다른 함수 정의를 이름 기준으로 중복 없이 모아, target_tokens
    토큰 분량이 될 때까지 누적한 고정 카탈로그 텍스트를 만든다. 이 텍스트가 모든
    요청에 토큰 단위로 완전히 동일하게 붙는 prefix가 된다.
    """
    seen_names: set[str] = set()
    catalog: list[dict] = []
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    text = _CATALOG_HEADER
    for i in indices[:max_scan]:
        for fn in _extract_functions(ds[i]["system"]):
            name = fn.get("name")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            catalog.append(fn)

        text = _CATALOG_HEADER + "\n".join(json.dumps(f, ensure_ascii=False) for f in catalog) + "\n"
        n_tokens = len(tokenizer.encode(text)) if tokenizer else len(text) // 2
        if n_tokens >= target_tokens:
            break

    return text


def load_prefix_cache_prompts(
    n: int, seed: int = 42, target_prefix_tokens: int = 1200, tokenizer=None
) -> tuple[list[str], str]:
    """
    Prefix Caching(긴 시스템 프롬프트) 검증용.

    glaiveai/glaive-function-calling-v2의 'system' 필드에는 row마다 함수 정의가
    1~2개씩 박혀 있다. 여러 row를 훑어 서로 다른 함수를 모아 target_prefix_tokens
    분량의 고정 "도구 카탈로그" 프리픽스를 만들고(단일 row의 system은 최대
    ~500토큰뿐이라 RAG 수준 프리픽스를 만들려면 여러 row를 합쳐야 한다), 그 뒤에
    서로 다른 row의 첫 USER 발화(query)를 이어 붙인다 — "공통 프리픽스 + 가변 요청"
    구조다.

    tokenizer를 넘기면(예: nano-vLLM이 실제 로드한 토크나이저) 프리픽스 길이를
    정확히 잰다. 넘기지 않으면 문자 수 기반으로 대략 추정한다.

    반환: (prompts, prefix_text) — prefix_text는 실제 프리픽스 토큰 수 확인/로깅용.
    """
    datasets = _require_datasets()
    ds = datasets.load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    rng = random.Random(seed)
    prefix_text = _build_tool_catalog(ds, target_prefix_tokens, tokenizer, rng)
    prefix_text += "\n[사용자 요청]\n"

    query_indices = list(range(len(ds)))
    rng.shuffle(query_indices)

    prompts = []
    for i in query_indices:
        if len(prompts) >= n:
            break
        query = _extract_user_query(ds[i]["chat"])
        if query:
            prompts.append(f"{prefix_text}{query}")

    if len(prompts) < n:
        raise ValueError(f"USER 발화를 추출할 수 있는 row가 {len(prompts)}개뿐이라 n={n}개를 채울 수 없습니다.")

    return prompts, prefix_text


def load_continuous_batching_prompts(n: int, seed: int = 42, num_categories: int = 6) -> list[str]:
    """
    Continuous Batching(다양한 길이의 동시 요청) 검증용.

    HAERAE-HUB/KMMLU에서 과목(config) 목록을 실행 시점에 조회해 무작위로
    num_categories개를 고르고, 그 안의 문제들을 섞어 뽑는다 — 실제 서비스처럼
    도메인/질문 길이가 제각각인 요청들을 재현하기 위함.
    """
    datasets = _require_datasets()
    rng = random.Random(seed)

    all_categories = datasets.get_dataset_config_names("HAERAE-HUB/KMMLU")
    chosen_categories = rng.sample(all_categories, min(num_categories, len(all_categories)))

    rows = []
    for category in chosen_categories:
        ds = datasets.load_dataset("HAERAE-HUB/KMMLU", category, split="test")
        rows.extend((category, ds[i]) for i in range(len(ds)))

    rng.shuffle(rows)
    if len(rows) < n:
        raise ValueError(
            f"선택한 {len(chosen_categories)}개 과목({chosen_categories})에서 {len(rows)}문항뿐이라 "
            f"n={n}개를 채울 수 없습니다."
        )

    prompts = []
    for category, row in rows[:n]:
        prompts.append(
            f"[{category}] 다음 문제의 정답을 A/B/C/D 중 하나로 고르세요.\n\n"
            f"질문: {row['question']}\n"
            f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\n\n정답:"
        )
    return prompts


_SQUAD_HEADER = "당신은 아래 문서를 참고해서 질문에 정확하게 답하는 어시스턴트입니다.\n\n[문서: {title}]\n"
_SQUAD_FOOTER = "\n[질문]\n"


def load_squad_context_prompts(
    n: int, seed: int = 42, target_prefix_tokens: int = 1200, tokenizer=None, max_titles_scan: int = 200,
    title: Optional[str] = None,
) -> tuple[list[str], str]:
    """
    Prefix Caching(세션형 RAG 유사 시나리오) 검증용: rajpurkar/squad.

    SQuAD는 같은 Wikipedia 문서(title) 아래 여러 문단(context)이 있고, 문단마다
    서로 다른 question이 5개 안팎 딸려 있다. title을 지정하지 않으면 무작위로 고른
    title 하나의 문단들을, 지정하면 그 title의 문단들을 처음부터 순서대로 이어붙여
    target_prefix_tokens 분량이 될 때까지 누적한 뒤, "그 안에 실제로 포함된 문단"에
    딸린 question들만 모아 가변 요청으로 삼는다 (포함되지 않은 문단의 question은
    프리픽스만 봐서는 답할 수 없으므로 제외한다).

    "검색 엔진이 문서 하나를 통째로 가져와 컨텍스트로 고정해두고, 같은 세션에서
    그 문서에 대해 여러 질문을 연달아 받는" RAG 시나리오를 재현한다. glaive 기반
    load_prefix_cache_prompts()와 반환 형식(prompts, prefix_text)이 동일해
    그대로 맞바꿔 쓸 수 있다.

    tokenizer를 넘기면(예: nano-vLLM이 실제 로드한 토크나이저) 프리픽스 길이를
    정확히 잰다. 넘기지 않으면 문자 수 기반으로 대략 추정한다.

    title: 특정 문서로 고정하고 싶을 때(예: 시연 영상에서 매번 같은 문서가 나오게)
    Wikipedia title(언더스코어 포함, 예: "Sexual_orientation")을 넘긴다. 그 title이
    없거나 조건(target_prefix_tokens 도달 + question n개 이상)을 못 채우면 즉시
    ValueError — 무작위 title로 자동 대체하지 않는다(시연 재현성이 title 고정의
    목적이므로, 조용히 다른 문서로 넘어가면 안 됨).
    """
    datasets = _require_datasets()
    ds = datasets.load_dataset("rajpurkar/squad", split="train")

    title_to_indices: dict[str, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        title_to_indices[ds[i]["title"]].append(i)

    rng = random.Random(seed)
    if title is not None:
        if title not in title_to_indices:
            raise ValueError(f"title '{title}'이 rajpurkar/squad에 없습니다.")
        candidate_titles = [title]
    else:
        candidate_titles = list(title_to_indices)
        rng.shuffle(candidate_titles)
        candidate_titles = candidate_titles[:max_titles_scan]

    for candidate_title in candidate_titles:
        rows = [ds[i] for i in title_to_indices[candidate_title]]

        unique_contexts = []
        seen_contexts = set()
        for row in rows:
            if row["context"] not in seen_contexts:
                seen_contexts.add(row["context"])
                unique_contexts.append(row["context"])

        header = _SQUAD_HEADER.format(title=candidate_title.replace("_", " "))
        body = ""
        included_contexts: set[str] = set()
        for context in unique_contexts:
            candidate_body = body + context + "\n\n"
            n_tokens = len(tokenizer.encode(header + candidate_body)) if tokenizer else len(header + candidate_body) // 2
            body = candidate_body
            included_contexts.add(context)
            if n_tokens >= target_prefix_tokens:
                break

        candidate_questions = list({row["question"] for row in rows if row["context"] in included_contexts})
        rng.shuffle(candidate_questions)
        if len(candidate_questions) < n:
            continue

        prefix_text = header + body + _SQUAD_FOOTER
        prompts = [f"{prefix_text}{q}" for q in candidate_questions[:n]]
        return prompts, prefix_text

    if title is not None:
        raise ValueError(
            f"title '{title}'은 target_prefix_tokens={target_prefix_tokens}에 도달하면서 동시에 서로 다른 "
            f"question이 {n}개 이상인 조건을 만족하지 못합니다. target_prefix_tokens를 낮추거나 n을 줄여보세요."
        )
    raise ValueError(
        f"title {max_titles_scan}개를 훑었지만 target_prefix_tokens={target_prefix_tokens}에 도달하면서 "
        f"동시에 서로 다른 question이 {n}개 이상인 문서를 찾지 못했습니다. max_titles_scan을 늘려보세요."
    )
