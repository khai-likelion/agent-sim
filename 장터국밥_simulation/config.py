"""
장터국밥 시뮬레이션 설정
"""
import os
from pathlib import Path

# .env 파일 로드 (python-dotenv 설치 시)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv 없으면 환경변수만 사용

# OpenAI GPT API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-3.5-turbo 등

# Groq API 설정
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Hugging Face API 설정
HF_API_KEY = os.getenv("HF_API_KEY", "your-hf-api-key-here")
HF_BASE_URL = "https://api-inference.huggingface.co/models"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Ollama 설정 (로컬 CPU 실행용)
OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"  # ollama run qwen2.5:7b

# 사용 API Provider 선택 ("openai", "groq", "hf", "ollama")
API_PROVIDER = "openai"  # 기본값을 OpenAI로 변경

# 시뮬레이션 설정
TIMESLOTS = ["06-11", "11-14", "17-21", "00-06"]  # 아침, 점심, 저녁, 야식

# 장터국밥 위치
JANGTER_LOCATION = {
    "name": "장터국밥",
    "x": 126.905530330727,
    "y": 37.5585189929025,
    "address": "서울 마포구 망원동 479-66"
}

# 주변 매장 검색 반경 (km)
NEARBY_RADIUS = 0.5
