"""
장터국밥 리포트 기반 전/후 비교 시뮬레이터
"""

import json
import time
import sys
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any
import httpx
import pandas as pd
from pathlib import Path

import config

# 로깅 설정
def setup_logging():
    """UTF-8 인코딩을 지원하는 로깅 설정 (실시간 업데이트)"""
    # 타임스탬프가 포함된 로그 파일명
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamped_log = log_dir / f"simulation_{timestamp}.log"
    latest_log = Path(__file__).parent / "latest_run.log"

    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()

    # 파일 핸들러 1: 타임스탬프 로그 (실시간 flush)
    file_handler1 = logging.FileHandler(timestamped_log, encoding='utf-8', mode='w')
    file_handler1.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    file_handler1.setLevel(logging.INFO)
    logger.addHandler(file_handler1)

    # 파일 핸들러 2: latest_run.log (실시간 flush)
    file_handler2 = logging.FileHandler(latest_log, encoding='utf-8', mode='w')
    file_handler2.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    file_handler2.setLevel(logging.INFO)
    logger.addHandler(file_handler2)

    # 콘솔 핸들러 (시스템 인코딩, 실시간 flush)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # 실시간 flush를 위한 커스텀 로깅
    class FlushLogger:
        def __init__(self, logger):
            self._logger = logger

        def info(self, msg):
            self._logger.info(msg)
            for handler in self._logger.handlers:
                handler.flush()

        def warning(self, msg):
            self._logger.warning(msg)
            for handler in self._logger.handlers:
                handler.flush()

        def error(self, msg):
            self._logger.error(msg)
            for handler in self._logger.handlers:
                handler.flush()

        def debug(self, msg):
            self._logger.debug(msg)
            for handler in self._logger.handlers:
                handler.flush()

    return FlushLogger(logger)

logger = setup_logging()


@dataclass
class CustomerAgent:
    """고객 에이전트 (18개 traits + 성별)"""

    id: int
    segment: str  # "주변_거주자_20대_남성_1인가구"
    age_group: str  # "20~30대"
    gender: str  # "남성", "여성"

    # 기본 특성 (6개)
    price_sensitivity: float = 0.5
    waiting_tolerance: float = 0.5
    visual_importance: float = 0.5
    brand_loyalty: float = 0.5
    novelty_seeking: float = 0.5
    social_influence: float = 0.5

    # 맛 선호도 (5개)
    spicy_preference: float = 0.5
    sweet_preference: float = 0.5
    mild_preference: float = 0.5
    rich_preference: float = 0.5
    authentic_preference: float = 0.5

    # 메뉴 도전 성향 (3개)
    new_menu_adventurous: float = 0.5
    cuisine_diversity: float = 0.5
    signature_focus: float = 0.5

    # 환경 민감도 (4개)
    time_sensitivity: float = 0.5
    ambiance_importance: float = 0.5
    noise_tolerance: float = 0.5
    child_friendly: float = 0.5

    def to_prompt_description(self) -> str:
        """LLM 프롬프트용 자연어 변환"""
        desc = []

        # 기본 특성
        if self.price_sensitivity > 0.7:
            desc.append(f"가격에 매우 민감합니다 (민감도 {self.price_sensitivity:.1f})")
        elif self.price_sensitivity < 0.3:
            desc.append(f"가격보다 품질을 우선합니다 (민감도 {self.price_sensitivity:.1f})")

        if self.visual_importance > 0.7:
            desc.append(f"음식의 비주얼과 플레이팅을 매우 중요하게 생각합니다 ({self.visual_importance:.1f})")

        if self.brand_loyalty > 0.7:
            desc.append(f"익숙한 단골집을 선호합니다 ({self.brand_loyalty:.1f})")
        elif self.brand_loyalty < 0.3:
            desc.append(f"항상 새로운 맛집을 탐방합니다 ({self.brand_loyalty:.1f})")

        if self.novelty_seeking > 0.7:
            desc.append(f"새로운 경험을 적극적으로 추구합니다 ({self.novelty_seeking:.1f})")

        if self.social_influence > 0.7:
            desc.append(f"SNS 핫플레이스를 자주 찾아다닙니다 ({self.social_influence:.1f})")

        # 맛 선호도
        taste_prefs = []
        if self.spicy_preference > 0.7:
            taste_prefs.append(f"매운맛 ({self.spicy_preference:.1f})")
        if self.sweet_preference > 0.7:
            taste_prefs.append(f"단맛 ({self.sweet_preference:.1f})")
        if self.mild_preference > 0.7:
            taste_prefs.append(f"담백한맛 ({self.mild_preference:.1f})")
        if self.rich_preference > 0.7:
            taste_prefs.append(f"진한맛 ({self.rich_preference:.1f})")
        if self.authentic_preference > 0.7:
            taste_prefs.append(f"정통맛 ({self.authentic_preference:.1f})")

        if taste_prefs:
            desc.append(f"{', '.join(taste_prefs)}을 선호합니다")

        # 메뉴 도전 성향
        if self.new_menu_adventurous > 0.7:
            desc.append(f"신메뉴를 적극적으로 시도하는 모험적인 성향입니다 ({self.new_menu_adventurous:.1f})")
        elif self.new_menu_adventurous < 0.3:
            desc.append(f"익숙한 메뉴를 선호하는 안정적인 성향입니다 ({self.new_menu_adventurous:.1f})")

        if self.cuisine_diversity > 0.7:
            desc.append(f"다양한 나라의 음식을 즐깁니다 ({self.cuisine_diversity:.1f})")

        if self.signature_focus > 0.7:
            desc.append(f"그 집의 시그니처 메뉴를 중시합니다 ({self.signature_focus:.1f})")

        # 환경 민감도
        if self.time_sensitivity > 0.7:
            desc.append(f"시간이 부족하여 빠른 서비스가 필요합니다 ({self.time_sensitivity:.1f})")

        if self.waiting_tolerance < 0.3:
            desc.append(f"웨이팅을 거의 하지 않습니다 ({self.waiting_tolerance:.1f})")
        elif self.waiting_tolerance > 0.7:
            desc.append(f"웨이팅도 기꺼이 합니다 ({self.waiting_tolerance:.1f})")

        if self.ambiance_importance > 0.7:
            desc.append(f"매장 분위기와 인테리어를 매우 중요하게 생각합니다 ({self.ambiance_importance:.1f})")

        if self.noise_tolerance < 0.3:
            desc.append(f"조용한 환경을 선호합니다 ({self.noise_tolerance:.1f})")
        elif self.noise_tolerance > 0.7:
            desc.append(f"시끌벅적한 분위기도 괜찮습니다 ({self.noise_tolerance:.1f})")

        if self.child_friendly > 0.7:
            desc.append(f"아이 동반이 편한 곳을 선호합니다 ({self.child_friendly:.1f})")

        return ". ".join(desc) if desc else "일반적인 성향을 가지고 있습니다"


@dataclass
class JangterStrategy:
    """장터국밥 개선 전략"""

    name: str
    description: str
    menu_changes: Dict[str, str] = field(default_factory=dict)
    target_segments: List[str] = field(default_factory=list)


class SimpleGroqClient:
    """Groq API 클라이언트"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = config.GROQ_MODEL
        self.base_url = config.GROQ_BASE_URL

    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """동기 생성 (Rate Limit Handling 포함)"""
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 512
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⚠️ [Rate Limit] API 한도 도달. 60초 대기 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(60)  # Rate Limit Reset 대기
                else:
                    logger.error(f"[ERROR] API 호출 상태 에러: {e}")
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[WARN] API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

class SimpleHFClient:
    """Hugging Face API Client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = config.HF_MODEL
        self.base_url = config.HF_BASE_URL

    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """Generation using HF Inference API"""
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.base_url}/{self.model}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 512,
                            "temperature": 0.7,
                            "return_full_text": False
                        }
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                # HF returns list of dicts with 'generated_text'
                return response.json()[0]["generated_text"]

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[WARN] HF API Call Failed (Attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise e


class SimpleOllamaClient:
    """Ollama Local API Client"""

    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.base_url = config.OLLAMA_BASE_URL

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generation using Ollama Local API"""
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 512
                        }
                    },
                    timeout=120.0  # CPU 실행은 오래 걸릴 수 있으므로 타임아웃 넉넉하게
                )
                response.raise_for_status()
                return response.json()["message"]["content"]

            except Exception as e:
                # Ollama가 안 켜져 있을 때 안내
                if "ConnectError" in str(e):
                    logger.error(f"❌ [Ollama 접속 실패] Ollama가 실행 중인지 확인해주세요.")
                    logger.error(f"   터미널에서 'ollama serve'를 실행하거나, 데스크탑 앱을 켜주세요.")
                    raise e

                if attempt < max_retries - 1:
                    logger.warning(f"[WARN] Ollama API Call Failed (Attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    raise e


class SimpleOpenAIClient:
    """OpenAI GPT API Client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = config.OPENAI_MODEL
        self.base_url = config.OPENAI_BASE_URL

    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """Generation using OpenAI API"""
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 512
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⚠️ [Rate Limit] OpenAI API 한도 도달. 60초 대기 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(60)
                elif e.response.status_code == 401:
                    logger.error(f"❌ [인증 실패] OpenAI API 키가 유효하지 않습니다. config.py에서 OPENAI_API_KEY를 확인해주세요.")
                    raise e
                else:
                    logger.error(f"[ERROR] OpenAI API 호출 상태 에러: {e}")
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[WARN] OpenAI API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise e


class JangterGukbapSimulator:
    """장터국밥 전/후 시뮬레이션"""

    def __init__(self):
        if hasattr(config, "API_PROVIDER"):
            if config.API_PROVIDER == "openai":
                self.client = SimpleOpenAIClient(config.OPENAI_API_KEY)
                logger.info(f"[INIT] Using OpenAI API (Model: {config.OPENAI_MODEL})")
            elif config.API_PROVIDER == "hf":
                self.client = SimpleHFClient(config.HF_API_KEY)
                logger.info(f"[INIT] Using Hugging Face API (Model: {config.HF_MODEL})")
            elif config.API_PROVIDER == "ollama":
                self.client = SimpleOllamaClient()
                logger.info(f"[INIT] Using Ollama Local API (Model: {config.OLLAMA_MODEL})")
                logger.info(f"   Create by Antigravity: CPU 모드로 실행 중입니다. 속도가 느릴 수 있습니다.")
            else:
                self.client = SimpleGroqClient(config.GROQ_API_KEY)
                logger.info(f"[INIT] Using Groq API (Model: {config.GROQ_MODEL})")
        else:
            self.client = SimpleGroqClient(config.GROQ_API_KEY)
            logger.info(f"[INIT] Using Groq API (Model: {config.GROQ_MODEL})")

        self.stores = self._load_nearby_stores()
        self.strategies = self._load_strategies()
        logger.info(f"[OK] 시뮬레이터 초기화 완료: 주변 매장 {len(self.stores)}개 로드됨")

    def _load_nearby_stores(self) -> List[Dict[str, Any]]:
        """장터국밥 주변 국밥/한식 매장 로드"""
        stores_path = Path(__file__).parent.parent / "data" / "raw" / "stores.csv"

        try:
            df = pd.read_csv(stores_path, encoding='cp949')
        except:
            df = pd.read_csv(stores_path, encoding='utf-8')

        # 장터국밥 위치
        jangter_x = config.JANGTER_LOCATION["x"]
        jangter_y = config.JANGTER_LOCATION["y"]

        # 거리 계산 (간단한 유클리드 거리)
        df['distance'] = ((df['x'] - jangter_x) ** 2 + (df['y'] - jangter_y) ** 2) ** 0.5

        # 필터링: 장터국밥 자신 제외하고, 너무 먼 곳(0.01 이상) 제외
        # 사용자 요청: 카테고리 제한 없이 제일 가까운 매장 선정
        df_filtered = df[
            (df['distance'] < 0.01)  # 약 1km 이내
        ].copy()

        # 거리순 정렬
        df_filtered = df_filtered.sort_values('distance')

        # 장터국밥을 첫 번째로
        df_jangter = df_filtered[df_filtered['장소명'] == '장터국밥']
        
        # 경쟁 매장 5개 (장터국밥 제외하고 가장 가까운 순)
        df_others = df_filtered[df_filtered['장소명'] != '장터국밥'].head(5)
        
        df_final = pd.concat([df_jangter, df_others])
        
        # 검증용 로그 출력
        store_names = [f"{row['장소명']}({row['카테고리']})" for _, row in df_final.iterrows()]
        logger.info(f"[CHECK] [데이터 검증] 선정된 경쟁 식당(거리순): {', '.join(store_names)}")

        return df_final.to_dict('records')

    def _load_strategies(self) -> List[JangterStrategy]:
        """리포트 기반 전략 로드"""
        return [
            JangterStrategy(
                name="양념장 가이드 표준화",
                description="테이블/벽면/메뉴판에 '양념장으로 간 맞추면 진국 맛 완성' 안내",
                menu_changes={"가이드": "양념장 사용법 안내문 비치"},
                target_segments=["관광객", "외지인", "초회_방문자"]
            ),
            JangterStrategy(
                name="세트 메뉴 구성",
                description="순대국 + 공기밥 + 미니수육 세트 (12,000원)",
                menu_changes={
                    "신메뉴": "순대국 세트 (12,000원)",
                    "구성": "순대국 + 공기밥 + 미니수육"
                },
                target_segments=["직장인", "가족", "가성비_중시"]
            ),
            JangterStrategy(
                name="포장 강화",
                description="포장 용기에 재가열 팁, 양념 비율 안내 제공",
                menu_changes={"포장": "재가열 팁 + 양념 비율 안내 스티커"},
                target_segments=["1인가구", "직장인", "배달_포장"]
            )
        ]

    def create_agents_by_timeslot(self, timeslot: str) -> List[CustomerAgent]:
        """시간대별 에이전트 생성 (남/여 구분)"""
        agents = []
        agent_id = 1

        # 주변 거주자 (항상 존재, 남/여 구분)
        # 1. 20~30대 1인가구 남성
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_20대_남성_1인가구", age_group="20~30대", gender="남성",
            price_sensitivity=0.6, waiting_tolerance=0.5, visual_importance=0.6,
            brand_loyalty=0.3, novelty_seeking=0.7, social_influence=0.6,
            spicy_preference=0.7, sweet_preference=0.3, mild_preference=0.4,
            rich_preference=0.8, authentic_preference=0.4,
            new_menu_adventurous=0.7, cuisine_diversity=0.7, signature_focus=0.5,
            time_sensitivity=0.4, ambiance_importance=0.4, noise_tolerance=0.7, child_friendly=0.1
        ))
        agent_id += 1

        # 2. 20~30대 1인가구 여성
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_20대_여성_1인가구", age_group="20~30대", gender="여성",
            price_sensitivity=0.6, waiting_tolerance=0.6, visual_importance=0.9,
            brand_loyalty=0.3, novelty_seeking=0.8, social_influence=0.9,
            spicy_preference=0.6, sweet_preference=0.6, mild_preference=0.5,
            rich_preference=0.5, authentic_preference=0.5,
            new_menu_adventurous=0.8, cuisine_diversity=0.8, signature_focus=0.6,
            time_sensitivity=0.4, ambiance_importance=0.8, noise_tolerance=0.6, child_friendly=0.1
        ))
        agent_id += 1

        # 3. 40~50대 가족 남성 (가장)
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_40대_남성_가족", age_group="40~50대", gender="남성",
            price_sensitivity=0.7, waiting_tolerance=0.4, visual_importance=0.3,
            brand_loyalty=0.8, novelty_seeking=0.3, social_influence=0.2,
            spicy_preference=0.5, sweet_preference=0.3, mild_preference=0.6,
            rich_preference=0.7, authentic_preference=0.8,
            new_menu_adventurous=0.3, cuisine_diversity=0.4, signature_focus=0.7,
            time_sensitivity=0.5, ambiance_importance=0.4, noise_tolerance=0.7, child_friendly=0.9
        ))
        agent_id += 1

        # 4. 40~50대 가족 여성 (주부)
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_40대_여성_가족", age_group="40~50대", gender="여성",
            price_sensitivity=0.8, waiting_tolerance=0.4, visual_importance=0.5,
            brand_loyalty=0.8, novelty_seeking=0.3, social_influence=0.3,
            spicy_preference=0.3, sweet_preference=0.4, mild_preference=0.8,
            rich_preference=0.4, authentic_preference=0.8,
            new_menu_adventurous=0.2, cuisine_diversity=0.4, signature_focus=0.6,
            time_sensitivity=0.6, ambiance_importance=0.5, noise_tolerance=0.6, child_friendly=0.95
        ))
        agent_id += 1

        # 5. 60대+ 시니어 남성
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_60대_남성_시니어", age_group="60대+", gender="남성",
            price_sensitivity=0.7, waiting_tolerance=0.6, visual_importance=0.2,
            brand_loyalty=0.9, novelty_seeking=0.2, social_influence=0.1,
            spicy_preference=0.3, sweet_preference=0.3, mild_preference=0.8,
            rich_preference=0.6, authentic_preference=0.9,
            new_menu_adventurous=0.2, cuisine_diversity=0.2, signature_focus=0.8,
            time_sensitivity=0.3, ambiance_importance=0.3, noise_tolerance=0.5, child_friendly=0.5
        ))
        agent_id += 1

        # 6. 60대+ 시니어 여성
        agents.append(CustomerAgent(
            id=agent_id, segment="주변_거주자_60대_여성_시니어", age_group="60대+", gender="여성",
            price_sensitivity=0.8, waiting_tolerance=0.7, visual_importance=0.2,
            brand_loyalty=0.9, novelty_seeking=0.2, social_influence=0.1,
            spicy_preference=0.2, sweet_preference=0.4, mild_preference=0.9,
            rich_preference=0.4, authentic_preference=0.9,
            new_menu_adventurous=0.1, cuisine_diversity=0.2, signature_focus=0.7,
            time_sensitivity=0.3, ambiance_importance=0.4, noise_tolerance=0.4, child_friendly=0.5
        ))
        agent_id += 1

        # 외부 유동인구 (시간대별, 남/여 구분)
        if timeslot == "11-14":  # 점심
            # 직장인 남성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_직장인_남성_점심", age_group="30~40대", gender="남성",
                price_sensitivity=0.6, waiting_tolerance=0.2, visual_importance=0.3,
                brand_loyalty=0.5, novelty_seeking=0.4, social_influence=0.3,
                spicy_preference=0.6, sweet_preference=0.3, mild_preference=0.5,
                rich_preference=0.7, authentic_preference=0.6,
                new_menu_adventurous=0.4, cuisine_diversity=0.5, signature_focus=0.6,
                time_sensitivity=0.9, ambiance_importance=0.2, noise_tolerance=0.8, child_friendly=0.1
            ))
            agent_id += 1

            # 직장인 여성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_직장인_여성_점심", age_group="30~40대", gender="여성",
                price_sensitivity=0.6, waiting_tolerance=0.3, visual_importance=0.6,
                brand_loyalty=0.5, novelty_seeking=0.5, social_influence=0.6,
                spicy_preference=0.4, sweet_preference=0.5, mild_preference=0.6,
                rich_preference=0.5, authentic_preference=0.5,
                new_menu_adventurous=0.5, cuisine_diversity=0.6, signature_focus=0.5,
                time_sensitivity=0.9, ambiance_importance=0.5, noise_tolerance=0.7, child_friendly=0.1
            ))
            agent_id += 1

        elif timeslot == "17-21":  # 저녁
            # 데이트 커플 남성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_데이트_남성_저녁", age_group="20~30대", gender="남성",
                price_sensitivity=0.4, waiting_tolerance=0.6, visual_importance=0.8,
                brand_loyalty=0.3, novelty_seeking=0.7, social_influence=0.7,
                spicy_preference=0.6, sweet_preference=0.4, mild_preference=0.4,
                rich_preference=0.6, authentic_preference=0.5,
                new_menu_adventurous=0.7, cuisine_diversity=0.7, signature_focus=0.6,
                time_sensitivity=0.3, ambiance_importance=0.9, noise_tolerance=0.3, child_friendly=0.1
            ))
            agent_id += 1

            # 데이트 커플 여성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_데이트_여성_저녁", age_group="20~30대", gender="여성",
                price_sensitivity=0.4, waiting_tolerance=0.7, visual_importance=0.95,
                brand_loyalty=0.3, novelty_seeking=0.8, social_influence=0.9,
                spicy_preference=0.5, sweet_preference=0.7, mild_preference=0.5,
                rich_preference=0.5, authentic_preference=0.5,
                new_menu_adventurous=0.8, cuisine_diversity=0.8, signature_focus=0.7,
                time_sensitivity=0.3, ambiance_importance=0.95, noise_tolerance=0.2, child_friendly=0.1
            ))
            agent_id += 1

            # 친구 모임 남성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_친구_남성_저녁", age_group="30~40대", gender="남성",
                price_sensitivity=0.5, waiting_tolerance=0.6, visual_importance=0.4,
                brand_loyalty=0.4, novelty_seeking=0.6, social_influence=0.6,
                spicy_preference=0.7, sweet_preference=0.3, mild_preference=0.4,
                rich_preference=0.8, authentic_preference=0.6,
                new_menu_adventurous=0.6, cuisine_diversity=0.7, signature_focus=0.5,
                time_sensitivity=0.3, ambiance_importance=0.4, noise_tolerance=0.9, child_friendly=0.2
            ))
            agent_id += 1

            # 친구 모임 여성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_친구_여성_저녁", age_group="30~40대", gender="여성",
                price_sensitivity=0.5, waiting_tolerance=0.7, visual_importance=0.7,
                brand_loyalty=0.4, novelty_seeking=0.7, social_influence=0.8,
                spicy_preference=0.5, sweet_preference=0.5, mild_preference=0.5,
                rich_preference=0.6, authentic_preference=0.5,
                new_menu_adventurous=0.7, cuisine_diversity=0.8, signature_focus=0.5,
                time_sensitivity=0.3, ambiance_importance=0.6, noise_tolerance=0.8, child_friendly=0.2
            ))
            agent_id += 1

        elif timeslot == "00-06":  # 야식
            # 관광객 남성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_관광객_남성_야식", age_group="20~40대", gender="남성",
                price_sensitivity=0.5, waiting_tolerance=0.7, visual_importance=0.7,
                brand_loyalty=0.2, novelty_seeking=0.9, social_influence=0.9,
                spicy_preference=0.6, sweet_preference=0.4, mild_preference=0.5,
                rich_preference=0.7, authentic_preference=0.8,
                new_menu_adventurous=0.8, cuisine_diversity=0.8, signature_focus=0.9,
                time_sensitivity=0.4, ambiance_importance=0.6, noise_tolerance=0.7, child_friendly=0.2
            ))
            agent_id += 1

            # 관광객 여성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_관광객_여성_야식", age_group="20~40대", gender="여성",
                price_sensitivity=0.5, waiting_tolerance=0.8, visual_importance=0.9,
                brand_loyalty=0.2, novelty_seeking=0.9, social_influence=0.95,
                spicy_preference=0.5, sweet_preference=0.6, mild_preference=0.5,
                rich_preference=0.6, authentic_preference=0.8,
                new_menu_adventurous=0.8, cuisine_diversity=0.8, signature_focus=0.9,
                time_sensitivity=0.4, ambiance_importance=0.7, noise_tolerance=0.6, child_friendly=0.2
            ))
            agent_id += 1

            # 혼술족 남성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_혼술_남성_야식", age_group="30~50대", gender="남성",
                price_sensitivity=0.6, waiting_tolerance=0.6, visual_importance=0.3,
                brand_loyalty=0.7, novelty_seeking=0.4, social_influence=0.3,
                spicy_preference=0.6, sweet_preference=0.3, mild_preference=0.5,
                rich_preference=0.8, authentic_preference=0.8,
                new_menu_adventurous=0.4, cuisine_diversity=0.5, signature_focus=0.7,
                time_sensitivity=0.3, ambiance_importance=0.5, noise_tolerance=0.6, child_friendly=0.1
            ))
            agent_id += 1

            # 혼술족 여성
            agents.append(CustomerAgent(
                id=agent_id, segment="유동_혼술_여성_야식", age_group="30~50대", gender="여성",
                price_sensitivity=0.6, waiting_tolerance=0.7, visual_importance=0.5,
                brand_loyalty=0.7, novelty_seeking=0.5, social_influence=0.5,
                spicy_preference=0.4, sweet_preference=0.4, mild_preference=0.6,
                rich_preference=0.6, authentic_preference=0.7,
                new_menu_adventurous=0.4, cuisine_diversity=0.5, signature_focus=0.6,
                time_sensitivity=0.3, ambiance_importance=0.6, noise_tolerance=0.5, child_friendly=0.1
            ))
            agent_id += 1

        return agents

    def _build_decision_prompt(
        self,
        agent: CustomerAgent,
        timeslot: str,
        with_strategy: bool
    ) -> str:
        """LLM 프롬프트 생성"""

        agent_desc = agent.to_prompt_description()

        stores_text = "\n".join([
            f"- {s['장소명']} ({s['카테고리']})"
            for s in self.stores
        ])

        timeslot_desc = {
            "06-11": "아침 (해장 시간대)",
            "11-14": "점심 (식사 시간대)",
            "17-21": "저녁 (외식/모임 시간대)",
            "00-06": "야식/새벽 (해장 시간대)"
        }.get(timeslot, timeslot)

        jangter_info = f"""
**장터국밥**
- 위치: 망원월드컵시장
- 메뉴: 순대국 (9,000원), 국밥 (8,000원)
- 특징: 진한 국물, 친절한 사장님, 포장 가능
- 리뷰 평가: 맛(0.85), 청결(0.80), 서비스(0.70), 가격가치(0.60)
- 대표 키워드: 순대국, 진국, 아날로그 감성, 망원시장, 포장 판매
"""

        if with_strategy:
            # 전략 적용 후
            jangter_info += f"""
✨ **신규 개선사항**:
  - 양념장 사용 가이드 제공 (테이블/벽면 안내문)
    -> "장터국밥은 맑고 시원한 베이스입니다. 양념장/새우젓으로 간 맞추면 진국 맛이 완성돼요"
  - 세트 메뉴 추가: 순대국 + 공기밥 + 미니수육 세트 (12,000원)
    -> 가성비 체감 개선
  - 포장 용기에 재가열 팁 제공
    -> 포장 고객을 위한 친절한 안내
"""
        else:
            # 전략 적용 전
            jangter_info += f"""
[WARN] **일부 고객 의견**:
  - "국물이 시원하지만 맹물 맛이 날 수 있음. 양념장 사용이 필수"
  - 가격가치 평가가 보통 수준 (0.60)
"""

        prompt = f"""당신은 망원동에서 식사할 곳을 찾는 소비자입니다.

### 당신의 정보
- 세그먼트: {agent.segment}
- 성별: {agent.gender}
- 연령대: {agent.age_group}

### 당신의 특성
{agent_desc}

### 현재 상황
- 시간대: {timeslot} ({timeslot_desc})
- 주변 매장 목록:
{stores_text}

{jangter_info}

### 임무 (2단계)
**1단계: 방문 결정**
위 매장 중 **어디를 방문할지** 결정하세요.

고려사항:
- 당신의 특성 (맛 선호도, 가격 민감도, 시간 민감도 등)을 반영하세요
- 시간대를 고려하세요 (해장, 점심, 외식, 야식)
- 장터국밥의 특징과 개선사항을 고려하세요
- 경쟁 매장과 비교하세요

**2단계: 방문 후 리뷰 작성**
방문한 매장에 대해 **별점과 리뷰**를 남기세요.

별점 기준 (1~5점):
- 5점: 매우 만족, 재방문 의사 강함
- 4점: 만족, 추천할 만함
- 3점: 보통, 기대에 부합
- 2점: 불만족, 개선 필요
- 1점: 매우 불만족, 재방문 안 함

리뷰 작성 시:
- 당신의 특성(맛 선호도, 가격 민감도 등)을 반영하세요
- 구체적으로 무엇이 좋았거나 아쉬웠는지 설명하세요
- 1-2문장으로 작성하세요

**반드시 JSON 형식으로만** 답하세요 (다른 설명 없이):
{{
  "store_name": "방문할 매장명",
  "reasoning": "방문 결정 이유 (1-2문장)",
  "rating": 별점 (1~5 정수),
  "review": "방문 후 리뷰 (1-2문장)",
  "rating_reason": "별점의 근거 (1문장)"
}}
"""
        return prompt

    def simulate_visit_decision(
        self,
        agent: CustomerAgent,
        timeslot: str,
        with_strategy: bool = False
    ) -> dict:
        """LLM 기반 방문 결정"""

        prompt = self._build_decision_prompt(agent, timeslot, with_strategy)

        try:
            # client.generate 사용 (Generic)
            response = self.client.generate(prompt)

            # JSON 파싱
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("JSON not found in response")

            return {
                "agent_id": agent.id,
                "segment": agent.segment,
                "gender": agent.gender,
                "timeslot": timeslot,
                "visited_store": result.get("store_name", "알 수 없음"),
                "reasoning": result.get("reasoning", ""),
                "rating": result.get("rating", 3),
                "review": result.get("review", ""),
                "rating_reason": result.get("rating_reason", ""),
                "jangter_visited": result.get("store_name") == "장터국밥"
            }

        except Exception as e:
            logger.error(f"[ERROR] LLM 응답 파싱 실패 (Agent {agent.id}): {e}")
            # Fallback
            return {
                "agent_id": agent.id,
                "segment": agent.segment,
                "gender": agent.gender,
                "timeslot": timeslot,
                "visited_store": "파싱_실패",
                "reasoning": f"LLM 응답 파싱 실패: {str(e)[:100]}",
                "rating": 3,
                "review": "파싱 실패로 리뷰 없음",
                "rating_reason": "LLM 응답 오류",
                "jangter_visited": False
            }

    def run_simulation(self) -> dict:
        """전체 시뮬레이션 실행"""

        results = {
            "before_strategy": {},
            "after_strategy": {},
            "summary": {}
        }

        logger.info("\n" + "="*60)
        logger.info("[START] 장터국밥 전/후 비교 시뮬레이션 시작")
        logger.info("="*60)

        for timeslot in config.TIMESLOTS:
            timeslot_name = {
                "06-11": "아침 (해장)",
                "11-14": "점심",
                "17-21": "저녁",
                "00-06": "야식/새벽"
            }.get(timeslot, timeslot)

            logger.info(f"\n{'='*60}")
            logger.info(f"[TIME] 시간대: {timeslot} - {timeslot_name}")
            logger.info(f"{'='*60}")

            agents = self.create_agents_by_timeslot(timeslot)
            logger.info(f"[AGENTS] 생성된 에이전트: {len(agents)}명")

            # 전략 적용 전
            logger.info(f"\n[DATA] [전략 적용 전] 시뮬레이션 중...")
            before_results = []
            for i, agent in enumerate(agents, 1):
                logger.info(f"\n  [{i}/{len(agents)}] {agent.segment} ({agent.gender}) 처리 중...")
                result = self.simulate_visit_decision(agent, timeslot, with_strategy=False)
                before_results.append(result)

                # 상세 정보 출력
                logger.info(f"  ✓ 선택: {result['visited_store']}")
                logger.info(f"    이유: {result['reasoning']}")
                logger.info(f"    별점: {result['rating']}/5 - {result['rating_reason']}")
                logger.info(f"    리뷰: {result['review']}")

                time.sleep(5.0)  # Rate limiting (5초로 증가)

            # 전략 적용 후
            logger.info(f"\n[DATA] [전략 적용 후] 시뮬레이션 중...")
            after_results = []
            for i, agent in enumerate(agents, 1):
                logger.info(f"\n  [{i}/{len(agents)}] {agent.segment} ({agent.gender}) 처리 중...")
                result = self.simulate_visit_decision(agent, timeslot, with_strategy=True)
                after_results.append(result)

                # 상세 정보 출력
                logger.info(f"  ✓ 선택: {result['visited_store']}")
                logger.info(f"    이유: {result['reasoning']}")
                logger.info(f"    별점: {result['rating']}/5 - {result['rating_reason']}")
                logger.info(f"    리뷰: {result['review']}")

                time.sleep(5.0)  # Rate limiting (5초로 증가)

            results["before_strategy"][timeslot] = before_results
            results["after_strategy"][timeslot] = after_results

            # 통계
            before_visits = sum(1 for r in before_results if r["jangter_visited"])
            after_visits = sum(1 for r in after_results if r["jangter_visited"])

            logger.info(f"\n[RESULT] 결과:")
            logger.info(f"  전략 전: {before_visits}/{len(agents)}명 ({before_visits/len(agents)*100:.1f}%)")
            logger.info(f"  전략 후: {after_visits}/{len(agents)}명 ({after_visits/len(agents)*100:.1f}%)")
            logger.info(f"  변화: {after_visits - before_visits:+d}명 ({(after_visits - before_visits)/len(agents)*100:+.1f}%p)")

            results["summary"][timeslot] = {
                "total_agents": len(agents),
                "before_visits": before_visits,
                "after_visits": after_visits,
                "change": after_visits - before_visits,
                "change_percent": (after_visits - before_visits) / len(agents) * 100
            }

        # 전체 요약
        logger.info(f"\n{'='*60}")
        logger.info("[DATA] 전체 요약")
        logger.info(f"{'='*60}")

        total_agents = sum(s["total_agents"] for s in results["summary"].values())
        total_before = sum(s["before_visits"] for s in results["summary"].values())
        total_after = sum(s["after_visits"] for s in results["summary"].values())

        if total_agents > 0:
            logger.info(f"총 에이전트: {total_agents}명")
            logger.info(f"전략 전 방문: {total_before}명 ({total_before/total_agents*100:.1f}%)")
            logger.info(f"전략 후 방문: {total_after}명 ({total_after/total_agents*100:.1f}%)")
            logger.info(f"전체 변화: {total_after - total_before:+d}명 ({(total_after - total_before)/total_agents*100:+.1f}%p)")

        return results


def main():
    """메인 실행"""

    # 시작 시간 기록
    start_time = datetime.now()
    logger.info(f"[START] 시뮬레이션 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 시뮬레이터 생성
    simulator = JangterGukbapSimulator()

    # 시뮬레이션 실행
    results = simulator.run_simulation()

    # 종료 시간 기록
    end_time = datetime.now()
    duration = end_time - start_time

    # 실행 시간 정보 추가
    results["execution_info"] = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "api_provider": config.API_PROVIDER,
        "model": getattr(config, f"{config.API_PROVIDER.upper()}_MODEL", "unknown")
    }

    # 결과 저장 (타임스탬프 포함)
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 타임스탬프가 있는 파일명
    timestamped_path = output_dir / f"simulation_results_{timestamp}.json"
    # 최신 결과를 가리키는 파일 (덮어쓰기)
    latest_path = output_dir / "simulation_results_latest.json"

    # 두 곳에 저장
    for path in [timestamped_path, latest_path]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n[SAVE] 결과 저장 완료:")
    logger.info(f"  - 타임스탬프 파일: {timestamped_path}")
    logger.info(f"  - 최신 파일: {latest_path}")
    logger.info(f"\n[TIME] 실행 시간: {duration.total_seconds():.1f}초 ({duration.total_seconds()/60:.1f}분)")
    logger.info(f"\n[OK] 시뮬레이션 완료!")

    return results


if __name__ == "__main__":
    main()
