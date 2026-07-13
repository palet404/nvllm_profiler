"""
workload/generator.py
사내 업무 이메일을 흉내 낸 가상 워크로드 생성기.

Prefix Caching 효과를 관찰하려면 모든 요청의 앞부분이 '토큰 단위로
완전히 동일'해야 한다. 그래서 SYSTEM_PREFIX를 항상 프롬프트 맨 앞에
그대로 이어붙이고, 이메일 본문(가변 길이·가변 내용)만 뒤에 붙인다.
nano-vLLM의 BlockManager는 이 공통 접두사에 해당하는 KV 블록들을
요청마다 다시 계산하지 않고 참조 카운트만 올려 재사용한다.
"""
import random
from dataclasses import dataclass

from config import SYSTEM_PREFIX

_SENDERS = ["인사팀", "재무팀", "IT지원팀", "총무팀", "대외협력팀", "감사실", "교육기획팀"]

_TASKS = [
    ("2분기 실적 보고서", "이번 주 금요일 18:00"),
    ("정보보안 서약서", "다음 주 월요일 오전 9시"),
    ("법인카드 사용 내역 정산", "이번 달 25일 자정"),
    ("연차 사용 계획서", "다음 주 수요일까지"),
    ("사내 교육 이수 확인증", "이번 달 말일"),
    ("협력사 계약서 검토 의견", "내일 오후 3시"),
    ("출장 경비 영수증", "복귀 후 3영업일 이내"),
    ("개인정보 처리 현황 점검표", "다음 주 금요일 17:00"),
]

_GREETINGS = [
    "안녕하세요, 항상 업무 협조에 감사드립니다.",
    "안녕하십니까, 아래와 같이 안내드립니다.",
    "수고 많으십니다. 관련하여 요청드립니다.",
]

_FILLERS = [
    "최근 사내 시스템 점검으로 일부 서비스 접속이 원활하지 않을 수 있는 점 양해 부탁드립니다.",
    "관련 문의사항은 내선 1234로 연락 주시기 바랍니다.",
    "첨부된 양식을 다운로드하여 작성 후 회신해 주시기 바랍니다.",
    "부서장 결재가 필요한 항목이니 사전에 일정을 확인해 주세요.",
    "본 안내는 전 직원 대상 공통 공지사항입니다.",
]

_CLOSINGS = [
    "감사합니다.",
    "협조에 감사드립니다.",
    "문의사항은 언제든 연락 주세요.",
]


@dataclass
class MockEmail:
    email_id: str
    sender: str
    subject: str
    body: str


def _make_body(rng: random.Random) -> str:
    task, deadline = rng.choice(_TASKS)
    greeting = rng.choice(_GREETINGS)
    filler = rng.choice(_FILLERS)
    closing = rng.choice(_CLOSINGS)
    return (
        f"{greeting}\n\n"
        f"{task} 제출 관련하여 안내드립니다. "
        f"제출 기한은 {deadline}까지이며, {filler}\n\n{closing}"
    )


def generate_mock_emails(n: int, seed: int = 42) -> list[MockEmail]:
    """가상의 사내 이메일 n건을 생성한다 (본문 내용은 매번 달라짐)."""
    rng = random.Random(seed)
    emails = []
    for i in range(n):
        sender = rng.choice(_SENDERS)
        task, _ = rng.choice(_TASKS)
        emails.append(
            MockEmail(
                email_id=f"mail-{i:04d}",
                sender=sender,
                subject=f"[{sender}] {task} 제출 안내",
                body=_make_body(rng),
            )
        )
    return emails


def build_prompt(email: MockEmail, system_prefix: str = SYSTEM_PREFIX) -> str:
    """system_prefix + 이메일 본문을 이어붙여 최종 LLM 입력 프롬프트를 만든다."""
    return f"{system_prefix}{email.body}"


def generate_prompts(
    n: int, seed: int = 42, duplicate_ratio: float = 0.0, system_prefix: str = SYSTEM_PREFIX
) -> list[str]:
    """
    분석 요청용 프롬프트 n개를 생성한다.

    duplicate_ratio: 이미 만든 이메일 중 하나를 그대로 재전송할 확률(0~1).
        같은 메일이 두 번 들어오면 system_prefix뿐 아니라 본문까지 KV 블록이
        100% 캐시 히트하므로, Prefix Caching의 TTFT 단축 효과를 더 극적으로
        보여주는 데모 시나리오로 쓸 수 있다.
    system_prefix: config.SYSTEM_PREFIX(기본, ~370토큰) 또는
        config.LONG_SYSTEM_PREFIX(~1000토큰, few-shot 스트레스 테스트용)를 넘길 수 있다.
    """
    rng = random.Random(seed)
    emails = generate_mock_emails(n, seed=seed)
    prompts: list[str] = []
    seen: list[MockEmail] = []

    for email in emails:
        if seen and rng.random() < duplicate_ratio:
            chosen = rng.choice(seen)
        else:
            chosen = email
            seen.append(email)
        prompts.append(build_prompt(chosen, system_prefix=system_prefix))

    return prompts
