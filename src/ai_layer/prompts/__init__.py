"""
프롬프트 템플릿 로더.
"""

from pathlib import Path
from typing import Dict, Any

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """프롬프트 템플릿 파일 로드."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 템플릿을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs) -> str:
    """프롬프트 템플릿 로드 후 변수 치환."""
    template = load_prompt(name)
    return template.format(**kwargs)


# 프롬프트 템플릿 이름 상수
STEP1_DESTINATION = "step1_destination"
STEP2_CATEGORY = "step2_category"
STEP3_STORE = "step3_store"
STEP4_EVALUATE = "step4_evaluate"
STEP5_NEXT_ACTION = "step5_next_action"
