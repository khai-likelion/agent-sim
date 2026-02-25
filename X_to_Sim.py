"""
utils_strategy_bridge_v2.py

망원동 상권 에이전트 시뮬레이션용 전략 적용 모듈
X-Report 전략을 기존 store JSON에 반영하여 새로운 상태 생성

[v2] update_text_fields 최적화 - Patch 방식으로 출력 토큰 감소
"""

import os
import sys
import io
import json
import re
import copy
import asyncio
from pathlib import Path
from difflib import SequenceMatcher

# Windows cp949 이모지 인코딩 에러 방지
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class StrategyBridge:
    """X-Report 전략을 store JSON에 반영하는 브리지"""

    # feature 매핑 (전략 키워드 → feature 이름)
    FEATURE_MAPPING = {
        "맛": "taste",
        "짠맛": "taste",
        "염도": "taste",
        "면발": "taste",
        "가성비": "price_value",
        "가격": "price_value",
        "객단가": "price_value",
        "청결": "cleanliness",
        "위생": "cleanliness",
        "서비스": "service",
        "친절": "service",
        "응대": "service",
        "회전율": "turnover",
        "웨이팅": "turnover",
        "대기": "turnover",
        "분위기": "atmosphere",
        "인테리어": "atmosphere",
        "감성": "atmosphere"
    }

    # 델타 정책
    MAX_DELTA_PER_FEATURE = 0.20
    MAX_DELTA_PER_STRATEGY = 0.10
    DEFAULT_DELTA = 0.05

    # 금지 표현 - 시계열/변화 암시 표현
    FORBIDDEN_PATTERNS = [
        r'개선되었', r'개선됐', r'개선돼', r'개선했',
        r'향상되었', r'향상됐', r'향상돼', r'향상했',
        r'변경되었', r'변경됐', r'변경돼', r'변경했',
        r'도입했', r'도입하여', r'도입되었', r'도입됐',
        r'이전에', r'과거에', r'예전에',
        r'더\s*나아졌', r'좋아졌',
        r'줄어들었', r'감소했', r'감소됐',
        r'증가했', r'늘어났', r'늘었',
        r'최근', r'새롭게',
    ]

    # 금지 표현 (메타 설명 - 괄호 안 설명)
    META_DESCRIPTION_PATTERNS = [
        r'\([^)]*불만[^)]*\)',
        r'\([^)]*개선[^)]*\)',
        r'\([^)]*해소[^)]*\)',
        r'\([^)]*보완[^)]*\)',
        r'\([^)]*배려[^)]*\)',
        r'\([^)]*대응[^)]*\)',
        r'\([^)]*위해[^)]*\)',
    ]

    def __init__(self, api_key: str, base_url: str = None, model_name: str = None):
        """
        Args:
            api_key: API 키 (OpenAI or Gemini)
            base_url: API 엔드포인트 (Gemini 사용 시 자동 감지)
            model_name: 모델 이름 (None이면 환경변수 또는 기본값)
        """
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        if base_url is None and provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        if model_name is None:
            model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite")

        self.model_name = model_name

        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

    async def _parse_json_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Any:
        """
        LLM 응답을 JSON으로 파싱하고, 실패 시 최대 2회 재시도
        response_format={"type": "json_object"} 적용으로 JSON 응답 강제
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )

                text = response.choices[0].message.content.strip()

                # JSON 추출 (안전장치)
                if text.startswith("```json"):
                    text = text.split("```json")[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                if text.startswith("```"):
                    text = text.split("```", 1)[1]

                return json.loads(text.strip())

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    print(f"    ⚠️ JSON 파싱 실패 (재시도 {attempt + 1}/{max_retries})")
                    messages = messages + [
                        {"role": "assistant", "content": text if 'text' in dir() else ""},
                        {"role": "user", "content": "설명 없이 JSON만 다시 출력하세요."}
                    ]
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"    ⚠️ API 오류 (재시도 {attempt + 1}/{max_retries}): {e}")
                    continue

        raise last_error

    @staticmethod
    def _cleanup_punctuation(text: str) -> str:
        """문장 대체 후 발생하는 이중 구두점·공백을 정리."""
        import re as _re
        # 마침표+쉼표, 쉼표+마침표 → 마침표
        text = _re.sub(r'\.\s*,', '. ', text)
        text = _re.sub(r',\s*\.', '. ', text)
        # 마침표 중복
        text = _re.sub(r'\.{2,}', '.', text)
        # 쉼표 중복
        text = _re.sub(r',{2,}', ',', text)
        # 공백 앞 구두점
        text = _re.sub(r'\s+([.,])', r'\1', text)
        # 다중 공백
        text = _re.sub(r' {2,}', ' ', text)
        return text.strip()

    def find_similar_segment(self, text: str, target: str, threshold: float = 0.6) -> Tuple[str, float]:
        """
        텍스트에서 target과 유사한 부분을 찾음

        Returns:
            (찾은 세그먼트, 유사도 점수) - 찾지 못하면 ("", 0.0)
        """
        # 문장 단위로 분리
        sentences = re.split(r'(?<=[.!?다])\s+', text)

        best_match = ""
        best_score = 0.0

        for sentence in sentences:
            score = SequenceMatcher(None, sentence, target).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = sentence

        # 부분 문자열 매칭도 시도
        target_words = target.split()[:5]
        for i in range(len(text) - 20):
            segment = text[i:i+len(target)+50]
            if any(word in segment for word in target_words):
                score = SequenceMatcher(None, segment[:len(target)], target).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    end_idx = segment.find('.')
                    if end_idx > 0:
                        best_match = text[i:i+end_idx+1]

        return (best_match, best_score)

    def extract_issues(self, x_report_text: str) -> List[Dict[str, Any]]:
        """X-Report에서 핵심 이슈(키워드) 추출"""
        issues = []
        # 'TOP X' 또는 '전략 카테고리 X' 지원 (더 유연한 매칭)
        sections = re.split(r'(?m)^(?=(?:#+\s*)?🔹\s*(?:\*\*|)?(?:TOP|전략 카테고리)\s*\d+)', x_report_text)

        for section in sections:
            if not section.strip():
                continue

            # 타이틀 매칭 (번호 추출)
            top_match = re.search(r'🔹\s*(?:\*\*|)?(?:TOP|전략 카테고리)\s*(\d+)', section)
            if not top_match:
                continue

            top_num = int(top_match.group(1))
            
            # 이슈 제목 추출
            title_match = re.search(r'🔹\s*(?:TOP|전략 카테고리)\s*\d+[:\.]?\s*(?:\*\*)?([^*|\n]+)(?:\*\*)?', section)
            issue_title = title_match.group(1).replace("[", "").replace("]", "").strip() if title_match else f"이슈{top_num}"

            issue_keyword = None
            for keyword, feature in self.FEATURE_MAPPING.items():
                if keyword in issue_title or keyword in section[:1000]:
                    issue_keyword = keyword
                    break

            if not issue_keyword:
                issue_keyword = issue_title.split()[0] if issue_title else f"이슈{top_num}"

            feature = self.FEATURE_MAPPING.get(issue_keyword, "turnover" if "회전" in issue_title or "대기" in issue_title else "taste")

            issues.append({
                "top_num": top_num,
                "issue_keyword": issue_keyword,
                "issue_title": issue_title,
                "feature": feature
            })

        return issues

    def extract_strategies(self, x_report_text: str) -> List[Dict[str, Any]]:
        """X-Report에서 전략 추출 (이슈 키워드 포함)"""
        issues = self.extract_issues(x_report_text)
        issue_map = {i['top_num']: i for i in issues}

        strategies = []
        sections = re.split(r'(?m)^(?=(?:#+\s*)?🔹\s*(?:\*\*|)?(?:TOP|전략 카테고리)\s*\d+)', x_report_text)

        for section in sections:
            if not section.strip():
                continue

            top_match = re.search(r'🔹\s*(?:\*\*|)?(?:TOP|전략 카테고리)\s*(\d+)', section)
            if not top_match:
                continue

            top_num = int(top_match.group(1))
            issue_info = issue_map.get(top_num, {})
            issue_keyword = issue_info.get('issue_keyword', '')

            # 전략 시작 단락들을 찾음 (숫자리스트나 - 불렛)
            # 예: "1) **제목**" 또는 "- **솔루션 A: 제목**"
            strategy_blocks = re.split(r'(?m)^(?=(?:\d+\)|-\s*\*\*))', section)
            
            for block in strategy_blocks:
                m = re.search(r'^(?:(?:\s*)?(\d+)\)|-)\s*\*\*([^*]+)\*\*', block)
                if m:
                    full_title = m.group(2).strip()
                    
                    # ID용 번호 추출
                    strategy_num = m.group(1) if m.group(1) else ""
                    if not strategy_num:
                        sol_match = re.search(r'(?:솔루션\s*)?([A-Z])', full_title)
                        strategy_num = sol_match.group(1) if sol_match else "1"

                    title = re.sub(r'^(?:솔루션\s*)?[A-Z][:\.]?\s*', '', full_title).strip()
                    
                    # 기대효과 분리
                    goal_text = ""
                    content_main = block.strip()
                    goal_match = re.search(r'(기대효과:|➜ 기대효과).*', block, re.DOTALL)
                    if goal_match:
                        goal_text = goal_match.group(0).strip()
                        content_main = block[:goal_match.start()].strip()

                    strategies.append({
                        "id": f"S{top_num}_{strategy_num}",
                        "title": title,
                        "content": content_main,
                        "goal": goal_text,
                        "issue_keyword": issue_keyword,
                        "top_num": top_num
                    })

        return strategies

    async def analyze_strategies_impact_batch(
        self,
        strategies: List[Dict[str, Any]],
        current_features: Dict[str, float]
    ) -> Dict[str, float]:
        """GPT로 모든 전략의 feature 영향을 한 번에 분석 (batch)"""
        feature_list = "\n".join([
            f"- {feat}: {score:.2f}"
            for feat, score in current_features.items()
        ])

        strategy_list = "\n\n".join([
            f"[{s['id']}] {s['title']}\n내용: {s['content']}\n기대효과: {s['goal']}"
            for s in strategies
        ])

        prompt = f"""당신은 전략과 평가 지표의 관계를 분석하는 전문가입니다.

# 전략 목록 (총 {len(strategies)}개)
{strategy_list}

# 현재 feature 점수
{feature_list}

# 작업
각 전략이 영향을 주는 feature와 점수 조정량을 분석하고, **feature별 총합**을 계산하세요.

## 규칙
1. 각 전략당 최대 delta: {self.MAX_DELTA_PER_STRATEGY}
2. 기대효과에 "10~20%p" 같은 수치 있으면 평균값 사용, 없으면 {self.DEFAULT_DELTA}
3. 확실한 관련성만 반영 (애매하면 제외)
4. 최종 출력은 **feature별 delta 합계**

## feature 종류
- taste: 맛
- price_value: 가성비
- cleanliness: 청결
- service: 서비스
- turnover: 회전율
- atmosphere: 분위기

# 출력 (JSON만, 설명 없이)
각 feature별 delta 총합을 출력하세요:
{{
  "taste": 합계값,
  "price_value": 합계값,
  "cleanliness": 합계값,
  "service": 합계값,
  "turnover": 합계값,
  "atmosphere": 합계값
}}

관련 없는 feature는 0으로 출력:"""

        messages = [
            {
                "role": "system",
                "content": "전략-feature 관계를 정확히 분석합니다. feature별 delta 합계를 JSON으로만 출력합니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            deltas = await self._parse_json_with_retry(messages, max_retries=2, temperature=0.3, max_tokens=500)

            valid_features = set(current_features.keys())
            result = {}
            for feat in valid_features:
                val = deltas.get(feat, 0)
                result[feat] = float(val) if val else 0.0

            return result

        except Exception as e:
            print(f"⚠️ GPT batch 분석 오류: {e}")
            return {feat: 0.0 for feat in current_features}

    def apply_deltas_to_features(
        self,
        total_deltas: Dict[str, float],
        current_features: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """델타를 feature_scores에 적용 (동기 함수)"""
        current_scores = {
            feat: data['score']
            for feat, data in current_features.items()
        }

        updated_scores = {}

        for feat, current_score in current_scores.items():
            delta = total_deltas.get(feat, 0.0)

            if delta > self.MAX_DELTA_PER_FEATURE:
                print(f"⚠️ {feat}: 델타 {delta:.2f} → {self.MAX_DELTA_PER_FEATURE:.2f} (상한 적용)")
                delta = self.MAX_DELTA_PER_FEATURE

            new_score = max(0.0, min(1.0, current_score + delta))
            updated_scores[feat] = round(new_score, 2)

            if delta > 0:
                print(f"{feat}: {current_score:.2f} → {new_score:.2f} (Δ{delta:+.2f})")

        return updated_scores

    async def extract_action_mappings(
        self,
        strategies: List[Dict[str, Any]],
        original_rag_context: str
    ) -> List[Dict[str, str]]:
        """GPT로 전략에서 '문제 표현 → 해결 후 표현' 매핑 추출"""
        strategy_details = "\n\n".join([
            f"[{s['id']}] {s['title']}\n이슈: {s.get('issue_keyword', '없음')}\n실행 내용: {s['content']}"
            for s in strategies
        ])

        prompt = f"""당신은 X-Report 전략을 rag_context에 적용하는 전문가입니다.

# 원본 rag_context
{original_rag_context}

# 적용할 전략들
{strategy_details}

# 작업
각 전략의 실행 내용을 기반으로, 원본 rag_context에서 수정이 필요한 부분을 찾아 매핑하세요.

## 규칙
1. 원본에서 **부정적/문제 표현**을 찾아 → 전략의 **실행 결과**로 대체할 표현 생성
2. 해결 후 표현은 **현재 상태**로 서술 (시간 흐름 표현 금지)
3. 전략과 관련 없는 부분은 매핑하지 않음
4. 구체적인 실행 내용을 반영 (예: "염도 조절 옵션" → "기본/덜짭게 선택 가능")

## 핵심 규칙
- **고객 경험 관점**으로만 서술 (운영/사장님 관점 금지)
- "~를 유도한다", "~를 공지한다", "~문구를 고정한다" 같은 운영 행위 금지
- 고객이 직접 인지/체험할 수 있는 팩트만 서술
- **반드시 자연스러운 손님 리뷰/후기 말투**로 작성 (공지문·정책문 금지)
  - ✅ 리뷰 말투: "깔끔한 편이에요", "청결하게 잘 관리되더라고요", "음식이 일관되게 맛있었어요"
  - ❌ 공지문 말투: "~관리되고 있다", "~확인할 수 있다", "~기준에 따라 관리되어", "~제공한다"

## 예시
- 원본: "짠맛이 강한 편으로 일부 고객에게는 적응하기 어려울 수 있습니다"
- 전략: "짠맛 이슈 대응 옵션화 - 주문 시 '기본/덜짭게' 선택"
- ❌ 잘못된 해결: "염도 조절 옵션을 제공한다" (공지문)
- ✅ 올바른 해결: "진한 맛이 특징인데, 주문할 때 덜짭게도 선택할 수 있어서 좋았어요" (리뷰 말투)

- 원본: "위생 상태와 청결함이 다소 떨어진다는 점을 지적하며"
- 전략: "청결 3포인트 표준(테이블·바닥·화장실) + 체크리스트 공개"
- ❌ 잘못된 해결: "테이블, 바닥, 화장실은 청결하게 관리되고, 청소 완료 상태를 확인할 수 있다" (공지문)
- ✅ 올바른 해결: "전에 위생 지적이 있었는데, 요즘은 테이블이랑 화장실 모두 꽤 깨끗한 편이에요" (리뷰 말투)

- 원본: "웨이팅 시간이 길 수 있으니 미리 예약하는 것이 좋습니다"
- 전략: "웨이팅 안내를 플레이스/인스타에 먼저 공지"
- ❌ 잘못된 해결: "웨이팅 안내를 플레이스에 공지하여 대기 방식을 안내한다" (운영 관점)
- ✅ 올바른 해결: "웨이팅이 있긴 한데 미리 대기 등록하고 가면 훨씬 수월해요" (리뷰 말투)

# 출력 (JSON만, 코드 블록 없이)
{{
  "mappings": [
    {{
      "issue_keyword": "이슈 키워드",
      "problem_expression": "원본에서 찾은 문제 표현 (정확히 복사)",
      "solution_expression": "전략 적용 후 대체할 표현",
      "source_strategy": "전략 ID"
    }}
  ]
}}

매핑이 없으면 {{"mappings": []}}"""

        messages = [
            {"role": "system", "content": "전략-텍스트 매핑 전문가입니다. JSON만 출력합니다."},
            {"role": "user", "content": prompt}
        ]

        try:
            result = await self._parse_json_with_retry(messages, max_retries=2, temperature=0.2, max_tokens=2000)
            return result.get("mappings", [])
        except Exception as e:
            print(f"⚠️ 액션 매핑 추출 오류: {e}")
            return []

    # ============================================================
    # [v2] update_text_fields - Patch 방식으로 최적화
    # ============================================================
    async def update_text_fields(
        self,
        strategies: List[Dict[str, Any]],
        original_comparison: str,
        original_rag_context: str,
        action_mappings: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        [v2] GPT로 수정이 필요한 문장만 반환받아 파이썬으로 치환 (Patch 방식)

        Args:
            action_mappings: 미리 추출된 액션 매핑 (병렬 처리 결과)

        Returns:
            {
                "updated_comparison": "업데이트된 텍스트",
                "updated_rag_context": "업데이트된 텍스트"
            }
        """
        # 1단계: action_mappings를 먼저 직접 적용
        working_comparison = original_comparison
        working_rag = original_rag_context

        if action_mappings:
            print(f"  ✓ {len(action_mappings)}개 매핑 직접 적용 중...")
            for m in action_mappings:
                problem = m.get('problem_expression', '')
                solution = m.get('solution_expression', '')
                if problem and solution:
                    replaced = False

                    # 정확한 매칭 시도
                    if problem in working_rag:
                        working_rag = working_rag.replace(problem, solution)
                        print(f"    ✓ rag_context 직접 대체: {m.get('issue_keyword')}")
                        replaced = True
                    elif problem in working_comparison:
                        working_comparison = working_comparison.replace(problem, solution)
                        print(f"    ✓ comparison 직접 대체: {m.get('issue_keyword')}")
                        replaced = True

                    # 유사 매칭 시도
                    if not replaced:
                        similar_rag, score_rag = self.find_similar_segment(working_rag, problem)
                        similar_comp, score_comp = self.find_similar_segment(working_comparison, problem)

                        if score_rag >= 0.5 and similar_rag:
                            working_rag = working_rag.replace(similar_rag, solution)
                            print(f"    ✓ rag_context 유사 대체 ({score_rag:.0%}): {m.get('issue_keyword')}")
                            replaced = True
                        elif score_comp >= 0.5 and similar_comp:
                            working_comparison = working_comparison.replace(similar_comp, solution)
                            print(f"    ✓ comparison 유사 대체 ({score_comp:.0%}): {m.get('issue_keyword')}")
                            replaced = True

                    if not replaced:
                        print(f"    ⚠️ 매칭 실패: {m.get('issue_keyword')}")
        else:
            print("  ⚠️ 매핑 없음")

        # 2단계: GPT에게 추가 수정이 필요한 문장만 Patch로 요청
        issue_keywords = list(set([s.get('issue_keyword', '') for s in strategies if s.get('issue_keyword')]))

        strategy_info = "\n".join([
            f"- {s['id']}: {s['title']} (이슈: {s.get('issue_keyword', '')})"
            for s in strategies
        ])

        issue_info = f"해결 대상 이슈 키워드: {', '.join(issue_keywords)}" if issue_keywords else ""

        # [v2] Patch 방식 프롬프트 - 수정할 문장 쌍만 반환
        prompt = f"""망원동 상권 시뮬레이션용 매장 정보를 검토하고, 규칙 위반 문장만 수정하세요.

# 적용된 전략
{strategy_info}

{issue_info}

# 현재 텍스트

## comparison
{working_comparison}

## rag_context
{working_rag}

# 절대 규칙

## 1. ❌ 금지 표현 (절대 사용 금지!)
- 개선되었다, 향상되었다, 변경되었다, 해결되었다, 보완되었다
- 도입했다, 도입하여, 도입되었다 (시계열 변화 암시)
- 이전에, 과거에, 예전에, 예전보다는
- 더 나아졌다, 좋아졌다, 맛이 좋아졌다
- 줄어들었다, 증가했다, 늘어났다
- 최근, 새롭게, 마련
- 그 외 모든 시계열적 변화/흐름 표현

## 2. ❌ 메타 설명 금지 (절대 포함 금지!)
- "(짠맛 불만 제거)", "(웨이팅 개선)", "(서비스 보완)" 같은 괄호 안 설명 금지
- "~를 해결하기 위해", "~를 개선하고자" 같은 의도 설명 금지
- 결과물에는 **현재 상태의 팩트만** 서술

## 3. ✅ 반드시 사용 (자연스러운 리뷰 말투)
- "~인 편이에요", "~더라고요", "~했어요", "~좋았어요", "~괜찮았어요"
- 구체적 팩트만, 새로운 근거 없는 수치 생성 금지
- ❌ 공지문/정책문 금지: "~관리되고 있다", "~확인할 수 있다", "~기준에 따라", "~제공한다", "~운영한다"

## 4. 🔍 핵심 특징 유지
- '맛(짠맛)', '면의 질감' 등 매장의 고유한 정체성은 삭제하지 말고, 부정적 표현을 자연스러운 리뷰 말투의 중립/긍정 표현으로 전환

# 작업
위 텍스트에서 **규칙을 위반하는 문장만** 찾아서, 수정된 문장으로 반환하세요.
- 전체 텍스트를 다시 생성하지 마세요
- 수정이 필요한 문장만 정확히 추출하여 반환
- 수정이 필요 없으면 빈 배열 반환

# 출력 (JSON만, 설명 없이)
{{
  "replacements": [
    {{
      "field": "comparison 또는 rag_context",
      "target_sentence": "원본에서 교체할 문장 (정확히 복사)",
      "new_sentence": "규칙이 적용된 새로운 문장"
    }}
  ]
}}

수정할 문장이 없으면 {{"replacements": []}}"""

        messages = [
            {
                "role": "system",
                "content": "시간 흐름 표현을 절대 사용하지 않습니다. 규칙 위반 문장만 찾아 수정합니다. JSON만 출력합니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            # GPT로부터 수정할 문장 쌍만 받음 (출력 토큰 최소화)
            result = await self._parse_json_with_retry(
                messages,
                max_retries=2,
                temperature=0.2,
                max_tokens=1000  # v1 대비 절반으로 감소
            )

            replacements = result.get("replacements", [])

            # 3단계: 파이썬으로 Patch 적용
            if replacements:
                print(f"  📝 GPT 추가 수정: {len(replacements)}개 문장")
                for r in replacements:
                    field = r.get("field", "")
                    target = r.get("target_sentence", "")
                    new_sent = r.get("new_sentence", "")

                    if not target or not new_sent:
                        continue

                    replaced = False

                    if field == "comparison":
                        # 정확한 매칭 시도
                        if target in working_comparison:
                            working_comparison = working_comparison.replace(target, new_sent)
                            print(f"    ✓ comparison 패치: \"{target[:30]}...\"")
                            replaced = True
                        else:
                            # 유사 매칭 시도
                            similar, score = self.find_similar_segment(working_comparison, target, threshold=0.5)
                            if similar and score >= 0.5:
                                working_comparison = working_comparison.replace(similar, new_sent)
                                print(f"    ✓ comparison 유사 패치 ({score:.0%}): \"{target[:30]}...\"")
                                replaced = True

                    elif field == "rag_context":
                        # 정확한 매칭 시도
                        if target in working_rag:
                            working_rag = working_rag.replace(target, new_sent)
                            print(f"    ✓ rag_context 패치: \"{target[:30]}...\"")
                            replaced = True
                        else:
                            # 유사 매칭 시도
                            similar, score = self.find_similar_segment(working_rag, target, threshold=0.5)
                            if similar and score >= 0.5:
                                working_rag = working_rag.replace(similar, new_sent)
                                print(f"    ✓ rag_context 유사 패치 ({score:.0%}): \"{target[:30]}...\"")
                                replaced = True

                    if not replaced:
                        print(f"    ⚠️ 패치 실패: \"{target[:30]}...\"")
            else:
                print("  ✅ GPT 추가 수정 없음 (규칙 준수)")

            return {
                "updated_comparison": self._cleanup_punctuation(working_comparison),
                "updated_rag_context": self._cleanup_punctuation(working_rag)
            }

        except Exception as e:
            print(f"⚠️ 텍스트 업데이트 오류: {e}")
            return {
                "updated_comparison": self._cleanup_punctuation(working_comparison),
                "updated_rag_context": self._cleanup_punctuation(working_rag)
            }

    def validate_no_time_expressions(self, text: str) -> tuple[bool, List[str]]:
        """시간 흐름 표현 및 메타 설명 검증"""
        violations = []

        for pattern in self.FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(f"[시간표현] {pattern} → {matches}")

        for pattern in self.META_DESCRIPTION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(f"[메타설명] {pattern} → {matches}")

        return (len(violations) == 0, violations)

    async def apply_strategies(
        self,
        store_json: Dict[str, Any],
        x_report_path: str,
        selected_strategy_ids: List[str]
    ) -> Dict[str, Any]:
        """전략을 store JSON에 적용 (비동기 병렬 처리)"""
        # 1. X-Report 로드 및 이슈/전략 추출
        print(f"📄 X-Report 로딩: {x_report_path}")
        with open(x_report_path, 'r', encoding='utf-8') as f:
            x_report_text = f.read()

        issues = self.extract_issues(x_report_text)
        print(f"\n📋 추출된 이슈 키워드:")
        for issue in issues:
            print(f"  TOP{issue['top_num']}: {issue['issue_keyword']} ({issue['issue_title']}) → {issue['feature']}")

        all_strategies = self.extract_strategies(x_report_text)
        print(f"\n✓ {len(all_strategies)}개 전략 추출됨\n")

        # 2. 선택된 전략 필터링
        selected_strategies = [
            s for s in all_strategies
            if s['id'] in selected_strategy_ids
        ]

        if not selected_strategies:
            print("⚠️ 선택된 전략이 없습니다.")
            return store_json

        print(f"🎯 적용할 전략: {len(selected_strategies)}개")
        for s in selected_strategies:
            issue_kw = s.get('issue_keyword', '')
            issue_str = f" (이슈: {issue_kw})" if issue_kw else ""
            print(f"  ✓ [{s['id']}] {s['title']}{issue_str}")
        print()

        # 3. 데이터 준비
        current_features = store_json['review_metrics']['feature_scores']
        current_scores = {feat: data['score'] for feat, data in current_features.items()}
        original_comparison = store_json['review_metrics']['overall_sentiment']['comparison']
        original_rag_context = store_json['rag_context']
        original_critical = store_json.get('critical_feedback', [])
        original_keywords = store_json.get('top_keywords', [])

        # ============================================================
        # 4. [병렬 처리] 4개의 독립적인 API 호출을 동시에 실행
        # ============================================================
        print("="*60)
        print("🚀 병렬 API 호출 시작...")
        print("="*60 + "\n")

        (
            total_deltas,
            action_mappings,
            updated_critical,
            updated_keywords
        ) = await asyncio.gather(
            self.analyze_strategies_impact_batch(selected_strategies, current_scores),
            self.extract_action_mappings(selected_strategies, original_rag_context),
            self.update_critical_feedback(selected_strategies, original_critical),
            self.update_top_keywords(selected_strategies, original_keywords)
        )

        print("✅ 병렬 API 호출 완료\n")

        # 5. feature_scores 델타 적용
        print("="*60)
        print("📊 전략 feature 영향 분석 결과")
        print("="*60 + "\n")

        print("전략별 feature 영향 합계:")
        for feat, delta in total_deltas.items():
            if delta > 0:
                print(f"  {feat}: +{delta:.2f}")
        print()

        print("="*60)
        print("📊 최종 feature_scores 업데이트")
        print("="*60 + "\n")

        updated_scores = self.apply_deltas_to_features(total_deltas, current_features)
        print()

        # ============================================================
        # 6. [v2] update_text_fields - Patch 방식 (순차 처리)
        # ============================================================
        print("="*60)
        print("🤖 GPT Patch 방식 텍스트 업데이트...")
        print("="*60 + "\n")

        text_updates = await self.update_text_fields(
            selected_strategies,
            original_comparison,
            original_rag_context,
            action_mappings
        )

        # 7. 검증
        print("\n" + "="*60)
        print("🔍 시간 흐름 표현 검증")
        print("="*60 + "\n")

        comp_valid, comp_violations = self.validate_no_time_expressions(
            text_updates.get('updated_comparison', '')
        )
        rag_valid, rag_violations = self.validate_no_time_expressions(
            text_updates.get('updated_rag_context', '')
        )

        if not comp_valid:
            print("⚠️ comparison에서 금지 표현 발견:")
            for v in comp_violations:
                print(f"  - {v}")

        if not rag_valid:
            print("⚠️ rag_context에서 금지 표현 발견:")
            for v in rag_violations:
                print(f"  - {v}")

        if comp_valid and rag_valid:
            print("✅ 검증 통과: 시간 흐름 표현 없음\n")
        else:
            print("⚠️ 재생성 권장\n")

        # 8. 원본 구조 복사 후 변경 부분만 업데이트
        result = copy.deepcopy(store_json)

        for feat, new_score in updated_scores.items():
            if feat in result['review_metrics']['feature_scores']:
                result['review_metrics']['feature_scores'][feat]['score'] = new_score

        result['review_metrics']['overall_sentiment']['comparison'] = text_updates.get(
            'updated_comparison', original_comparison
        )

        result['rag_context'] = text_updates.get('updated_rag_context', original_rag_context)

        result['critical_feedback'] = updated_critical

        result['top_keywords'] = updated_keywords

        return result

    async def update_critical_feedback(
        self,
        strategies: List[Dict[str, Any]],
        original_critical: List[str]
    ) -> List[str]:
        """전략 적용 후 critical_feedback 업데이트"""
        if not original_critical:
            return []

        issue_keywords = list(set([s.get('issue_keyword', '') for s in strategies if s.get('issue_keyword')]))

        strategy_info = "\n".join([
            f"- {s['id']}: {s['title']} (이슈: {s.get('issue_keyword', '')})"
            for s in strategies
        ])

        prompt = f"""critical_feedback 목록에서 전략으로 해결된 단점을 삭제하세요.

# 중요: critical_feedback의 역할
이 필드는 에이전트가 매장의 **'치명적인 단점'으로만 인식**해야 하는 태그(Tag) 영역입니다.
긍정적이거나 중립적인 팩트가 이 리스트에 포함되면 에이전트가 이를 단점으로 오독합니다.

# 적용된 전략
{strategy_info}

# 해결 대상 이슈 키워드
{', '.join(issue_keywords)}

# 현재 critical_feedback
{json.dumps(original_critical, ensure_ascii=False, indent=2)}

# 규칙 (엄격히 준수!)
1. **전략으로 해결된 문제점은 리스트에서 완전히 삭제(Drop)하세요.**
2. **어떠한 경우에도 긍정적이거나 중립적인 팩트(~가능함, ~제공됨, ~있음 등)를 생성하여 남겨두지 마세요.**
3. 전략과 무관하여 **아직 해결되지 않은 단점들만** 원본 텍스트 그대로 유지하세요.
4. 모든 문제가 해결되었다면 빈 배열을 반환하세요.

# 예시

## 예시 1
- 원본: "매우 짠 국물 맛으로 인해 일부 고객에게는 부적합할 수 있음"
- 전략: 염도 조절 옵션화
- ❌ 잘못된 결과: "진한 맛이 특징이며, 염도 조절이 가능함" (긍정/중립 팩트 변환 금지!)
- ✅ 올바른 결과: (해당 항목 완전 삭제)

## 예시 2
- 원본: "웨이팅 시간이 길 수 있음"
- 전략: 대기 안내 시스템, 원격 줄서기
- ✅ 올바른 결과: (해당 항목 완전 삭제)
  → "예상 대기시간 안내 및 원격 줄서기 이용 가능"은 팩트이지만,
     critical_feedback은 치명적 단점 전용 필드이므로 여기에는 넣지 않음.
     해당 팩트는 rag_context에서 서술됨.

## 예시 3
- 원본: "면의 질감이 일부 고객에게는 적합하지 않을 수 있음"
- 전략: (관련 전략 없음)
- ✅ 올바른 결과: "면의 질감이 일부 고객에게는 적합하지 않을 수 있음" (원본 그대로 유지)

# 출력 (JSON 객체, 코드 블록 없이)
- 유지할 단점이 있으면: {{"items": ["유지할_단점1", "유지할_단점2"]}}
- 모든 단점이 해결되었으면: {{"items": []}}

남길 단점만 반환:"""

        messages = [
            {"role": "system", "content": "critical_feedback에서 해결된 단점을 삭제합니다. 긍정/중립 팩트 생성 금지. JSON만 출력합니다."},
            {"role": "user", "content": prompt}
        ]

        try:
            result = await self._parse_json_with_retry(messages, max_retries=2, temperature=0.2, max_tokens=500)
            return result.get("items", [])
        except Exception as e:
            print(f"⚠️ critical_feedback 업데이트 오류: {e}")
            return original_critical

    async def update_top_keywords(
        self,
        strategies: List[Dict[str, Any]],
        original_keywords: List[str]
    ) -> List[str]:
        """GPT로 전략 기반 top_keywords 업데이트"""
        if not original_keywords:
            return []

        strategy_info = "\n".join([
            f"- {s['id']}: {s['title']} (이슈: {s.get('issue_keyword', '')})"
            for s in strategies
        ])

        prompt = f"""top_keywords 목록에서 전략으로 해결된 부정적 키워드를 제거하세요.

# 적용된 전략
{strategy_info}

# 현재 top_keywords
{json.dumps(original_keywords, ensure_ascii=False)}

# 규칙
1. 전략으로 **해결된 문제**와 관련된 **부정적 키워드**만 제거
2. 중립적/긍정적 키워드는 유지 (예: "맛", "분위기", "친절")
3. 전략과 무관한 키워드는 유지
4. 매장의 특징을 나타내는 중립 키워드는 유지 (예: "면", "숙주", "차슈")

# 예시
- 전략: "짠맛 대응 - 염도 조절 옵션화"
- 키워드 "짜다" → 제거 (부정적 + 해결됨)
- 키워드 "맛" → 유지 (중립적)

- 전략: "웨이팅 안내 시스템"
- 키워드 "웨이팅" → 제거 (부정적 + 해결됨)
- 키워드 "줄" → 제거 (부정적 + 해결됨)

# 출력 (JSON 객체, 설명 없이)
{{"keywords": ["유지할_키워드1", "유지할_키워드2", ...]}}"""

        messages = [
            {"role": "system", "content": "top_keywords를 분석하여 전략으로 해결된 부정적 키워드를 제거합니다. JSON만 출력합니다."},
            {"role": "user", "content": prompt}
        ]

        try:
            result = await self._parse_json_with_retry(messages, max_retries=2, temperature=0.2, max_tokens=300)
            updated = result.get("keywords", original_keywords)

            removed = set(original_keywords) - set(updated)
            if removed:
                print(f"  ✓ top_keywords에서 제거: {', '.join(removed)}")

            return updated

        except Exception as e:
            print(f"⚠️ top_keywords 업데이트 오류: {e}")
            return original_keywords


# ============================================================================
# Async wrapper 함수들
# ============================================================================

async def apply_x_report_strategy_async(
    store_json_path: str,
    x_report_path: str,
    selected_strategy_ids: List[str],
    api_key: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """전략 적용 메인 함수 (비동기 버전)"""
    with open(store_json_path, 'r', encoding='utf-8') as f:
        store_json = json.load(f)

    bridge = StrategyBridge(api_key=api_key)
    result = await bridge.apply_strategies(store_json, x_report_path, selected_strategy_ids)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 결과 저장: {output_path}\n")

    return result


def apply_x_report_strategy(
    store_json_path: str,
    x_report_path: str,
    selected_strategy_ids: List[str],
    api_key: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """전략 적용 메인 함수 (동기 wrapper - 기존 API 호환)"""
    return asyncio.run(apply_x_report_strategy_async(
        store_json_path,
        x_report_path,
        selected_strategy_ids,
        api_key,
        output_path
    ))


def get_xreport_issues(x_report_path: str, api_key: str = None) -> List[Dict[str, Any]]:
    """X-Report에서 이슈 목록만 추출 (전략 선택 전 조회용)"""
    bridge = StrategyBridge(api_key=api_key or "dummy")

    with open(x_report_path, 'r', encoding='utf-8') as f:
        x_report_text = f.read()

    return bridge.extract_issues(x_report_text)


def get_xreport_strategies(x_report_path: str, api_key: str = None) -> List[Dict[str, Any]]:
    """X-Report에서 전략 목록 추출 (이슈 키워드 포함)"""
    bridge = StrategyBridge(api_key=api_key or "dummy")

    with open(x_report_path, 'r', encoding='utf-8') as f:
        x_report_text = f.read()

    return bridge.extract_strategies(x_report_text)


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import time

    # 설정
    _PROJECT_ROOT = Path(__file__).resolve().parent
    API_KEY = os.getenv("LLM_API_KEY")
    STORE_NAME = "돼지야"
    STORE_JSON_PATH = str(_PROJECT_ROOT / "data" / "raw" / "split_by_store_id_ver5" / f"{STORE_NAME}.json")
    X_REPORT_PATH   = str(_PROJECT_ROOT / "data" / "raw" / "전략 md 파일들" / f"{STORE_NAME}_report.md")
    # 레버 3개만 선택 (S1_A, S2_A, S3_A)
    SELECTED_STRATEGY_IDS = ["S1_A", "S2_A", "S3_A"]

    # 실행 시간 측정
    start_time = time.time()

    # 실행
    result = apply_x_report_strategy(
        store_json_path=STORE_JSON_PATH,
        x_report_path=X_REPORT_PATH,
        selected_strategy_ids=SELECTED_STRATEGY_IDS,
        api_key=API_KEY,
        output_path=f"{STORE_NAME}_전략적용_v2.json"
    )

    elapsed_time = time.time() - start_time

    # 결과 확인
    print("\n" + "="*60)
    print("📊 최종 결과")
    print("="*60 + "\n")

    print("Feature Scores:")
    for feat, data in result['review_metrics']['feature_scores'].items():
        print(f"  {feat}: {data['score']}")

    print("\nComparison:")
    print(result['review_metrics']['overall_sentiment']['comparison'])

    print("\nRAG Context:")
    print(result['rag_context'][:300] + "...")

    print("\nCritical Feedback:")
    for item in result.get('critical_feedback', []):
        print(f"  - {item}")

    print("\nTop Keywords:")
    print(f"  {result.get('top_keywords', [])}")

    print(f"\n⏱️ 총 실행 시간: {elapsed_time:.2f}초")
