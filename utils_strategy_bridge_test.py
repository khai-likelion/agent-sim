"""
utils_strategy_bridge.py

망원동 상권 에이전트 시뮬레이션용 전략 적용 모듈
X-Report 전략을 기존 store JSON에 반영하여 새로운 상태 생성

[Async 리팩토링] - 병렬 처리로 속도 최적화
"""

import os
import json
import re
import copy
import asyncio
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()

# Provider별 OpenAI-compatible 엔드포인트
PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


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

    # [수정2] 금지 표현 - '도입' 단어 자체 제거, 시계열/변화 암시 표현만 금지
    FORBIDDEN_PATTERNS = [
        r'개선되었', r'개선됐', r'개선돼', r'개선했',
        r'향상되었', r'향상됐', r'향상돼', r'향상했',
        r'변경되었', r'변경됐', r'변경돼', r'변경했',
        r'도입했', r'도입하여', r'도입되었', r'도입됐',  # 도입 + 과거/변화 표현만 금지
        r'이전에', r'과거에', r'예전에',
        r'더\s*나아졌', r'좋아졌',
        r'줄어들었', r'감소했', r'감소됐',
        r'증가했', r'늘어났', r'늘었',
        r'최근', r'새롭게',
    ]

    # 금지 표현 (메타 설명 - 괄호 안 설명)
    META_DESCRIPTION_PATTERNS = [
        r'\([^)]*불만[^)]*\)',      # (짠맛 불만 제거), (서비스 불만 해소)
        r'\([^)]*개선[^)]*\)',      # (웨이팅 개선), (맛 개선)
        r'\([^)]*해소[^)]*\)',      # (불만 해소)
        r'\([^)]*보완[^)]*\)',      # (서비스 보완)
        r'\([^)]*배려[^)]*\)',      # (민감층 배려)
        r'\([^)]*대응[^)]*\)',      # (고객 대응)
        r'\([^)]*위해[^)]*\)',      # (개선을 위해)
    ]

    def __init__(self, api_key: str = None, provider: str = None, model_name: str = None):
        """
        Args:
            api_key: API 키 (기본값: LLM_API_KEY 환경변수)
            provider: LLM provider (기본값: LLM_PROVIDER 환경변수)
            model_name: 모델 이름 (기본값: LLM_MODEL_NAME 환경변수)
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
        self.base_url = PROVIDER_URLS.get(self.provider, PROVIDER_URLS["gemini"])

    # [Async] JSON 파싱 헬퍼 with 재시도 로직 + response_format 적용
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
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "response_format": {"type": "json_object"}
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    text = data["choices"][0]["message"]["content"].strip()

                # JSON 추출 (안전장치 - response_format 사용해도 마크다운 올 수 있음)
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
                    # 재시도 프롬프트 추가
                    messages = messages + [
                        {"role": "assistant", "content": text if 'text' in locals() else ""},
                        {"role": "user", "content": "설명 없이 JSON만 다시 출력하세요."}
                    ]
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"    ⚠️ API 오류 (재시도 {attempt + 1}/{max_retries}): {e}")
                    continue

        raise last_error

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
            # 문장 유사도 계산
            score = SequenceMatcher(None, sentence, target).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = sentence

        # 부분 문자열 매칭도 시도
        target_words = target.split()[:5]  # 처음 5단어
        for i in range(len(text) - 20):
            segment = text[i:i+len(target)+50]
            if any(word in segment for word in target_words):
                score = SequenceMatcher(None, segment[:len(target)], target).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    # 문장 경계까지 확장
                    end_idx = segment.find('.')
                    if end_idx > 0:
                        best_match = text[i:i+end_idx+1]

        return (best_match, best_score)

    def extract_issues(self, x_report_text: str) -> List[Dict[str, Any]]:
        """
        X-Report에서 핵심 이슈(키워드) 추출

        Returns:
            [
                {
                    "top_num": 1,
                    "issue_keyword": "짠맛",
                    "issue_title": "짠맛 민감층 이탈 가능성",
                    "feature": "taste"
                },
                ...
            ]
        """
        issues = []
        sections = re.split(r'(?=🔹\s*\*\*TOP\s*\d+)', x_report_text)

        for section in sections:
            if not section.strip():
                continue

            # TOP 번호와 이슈 제목 추출
            top_match = re.search(r'🔹\s*\*\*TOP\s*(\d+):\s*([^*\n]+)', section)
            if not top_match:
                continue

            top_num = int(top_match.group(1))
            issue_title = top_match.group(2).strip()

            # 핵심 키워드 추출 (이슈 제목에서)
            issue_keyword = None
            for keyword, feature in self.FEATURE_MAPPING.items():
                if keyword in issue_title or keyword in section[:500]:
                    issue_keyword = keyword
                    break

            if not issue_keyword:
                # 제목에서 첫 단어를 키워드로 사용
                issue_keyword = issue_title.split()[0] if issue_title else f"이슈{top_num}"

            # feature 매핑
            feature = self.FEATURE_MAPPING.get(issue_keyword, "taste")

            issues.append({
                "top_num": top_num,
                "issue_keyword": issue_keyword,
                "issue_title": issue_title,
                "feature": feature
            })

        return issues

    def extract_strategies(self, x_report_text: str) -> List[Dict[str, Any]]:
        """
        X-Report에서 전략 추출 (이슈 키워드 포함)

        Returns:
            [
                {
                    "id": "S1_1",
                    "title": "피크타임 병목 3분해 체크",
                    "content": "전체 실행 내용",
                    "goal": "기대효과 텍스트",
                    "issue_keyword": "웨이팅",
                    "top_num": 1
                },
                ...
            ]
        """
        # 먼저 이슈 추출
        issues = self.extract_issues(x_report_text)
        issue_map = {i['top_num']: i for i in issues}

        strategies = []
        sections = re.split(r'(?=🔹\s*\*\*TOP\s*\d+)', x_report_text)

        for section in sections:
            if not section.strip():
                continue

            top_match = re.search(r'🔹\s*\*\*TOP\s*(\d+):', section)
            if not top_match:
                continue

            top_num = top_match.group(1)

            exec_start = section.find('➜ 실행')
            goal_start = section.find('➜ 기대효과')

            if exec_start == -1:
                continue

            exec_end = goal_start if goal_start != -1 else len(section)
            exec_text = section[exec_start:exec_end]
            goal_text = section[goal_start:] if goal_start != -1 else ""

            # 실행 항목 추출 (1), 2), 3))
            actions = re.findall(r'(\d+)\)\s*\*\*([^*]+)\*\*:?\s*([^\n]+)', exec_text)

            # 이슈 정보 가져오기
            issue_info = issue_map.get(int(top_num), {})
            issue_keyword = issue_info.get('issue_keyword', '')

            for num, title, desc in actions:
                strategy_id = f"S{top_num}_{num}"

                # 해당 항목의 전체 내용 추출
                pattern = rf'{num}\)\s*\*\*{re.escape(title.strip())}\*\*.*?(?=\d+\)|\➜|$)'
                full_match = re.search(pattern, exec_text, re.DOTALL)
                full_content = full_match.group(0) if full_match else f"{title}: {desc}"

                strategies.append({
                    "id": strategy_id,
                    "title": title.strip(),
                    "content": full_content.strip(),
                    "goal": goal_text.strip(),
                    "issue_keyword": issue_keyword,
                    "top_num": int(top_num)
                })

        return strategies

    # [Async] batch 방식으로 변경 - 전략 전체를 한번에 분석
    async def analyze_strategies_impact_batch(
        self,
        strategies: List[Dict[str, Any]],
        current_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        GPT로 모든 전략의 feature 영향을 한 번에 분석 (batch)

        Args:
            strategies: 전략 리스트
            current_features: 현재 feature_scores (feature → score 매핑)

        Returns:
            {
                "taste": 0.08,
                "turnover": 0.15,
                ...
            }
        """
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

            # 유효한 feature만 필터링하고 float 변환
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
        """
        델타를 feature_scores에 적용 (동기 함수)

        Args:
            total_deltas: feature별 delta 합계
            current_features: 현재 feature_scores (원본 구조)

        Returns:
            업데이트된 feature_scores (feature → score만)
        """
        # 현재 점수 추출
        current_scores = {
            feat: data['score']
            for feat, data in current_features.items()
        }

        # 정책 상한 적용 및 최종 점수 계산
        updated_scores = {}

        for feat, current_score in current_scores.items():
            delta = total_deltas.get(feat, 0.0)

            # feature당 최대 델타 제한
            if delta > self.MAX_DELTA_PER_FEATURE:
                print(f"⚠️ {feat}: 델타 {delta:.2f} → {self.MAX_DELTA_PER_FEATURE:.2f} (상한 적용)")
                delta = self.MAX_DELTA_PER_FEATURE

            # 최종 점수 (0~1 clip)
            new_score = max(0.0, min(1.0, current_score + delta))
            updated_scores[feat] = round(new_score, 2)

            if delta > 0:
                print(f"{feat}: {current_score:.2f} → {new_score:.2f} (Δ{delta:+.2f})")

        return updated_scores

    # [Async] 액션 매핑 추출
    async def extract_action_mappings(
        self,
        strategies: List[Dict[str, Any]],
        original_rag_context: str
    ) -> List[Dict[str, str]]:
        """
        GPT로 전략에서 '문제 표현 → 해결 후 표현' 매핑 추출

        Returns:
            [
                {
                    "issue_keyword": "웨이팅",
                    "problem_expression": "웨이팅 시간이 길 수 있으니 미리 예약하는 것이 좋습니다",
                    "solution_expression": "대기 안내 시스템이 운영되며, 예상 대기시간이 안내된다",
                    "source_strategy": "S1_2"
                },
                ...
            ]
        """
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

## 예시
- 원본: "짠맛이 강한 편으로 일부 고객에게는 적응하기 어려울 수 있습니다"
- 전략: "짠맛 이슈 대응 옵션화 - 주문 시 '기본/덜짭게' 선택"
- ❌ 잘못된 해결: "짠맛 이슈 대응을 위해 염도 조절 옵션을 제공한다" (운영 관점)
- ✅ 올바른 해결: "진한 맛이 특징이며, 주문 시 염도 조절(기본/덜짭게)을 선택할 수 있다" (고객 경험)

- 원본: "웨이팅 시간이 길 수 있으니 미리 예약하는 것이 좋습니다"
- 전략: "웨이팅 안내를 플레이스/인스타에 먼저 공지"
- ❌ 잘못된 해결: "웨이팅 안내를 플레이스에 공지하여 대기 방식을 안내한다" (운영 관점)
- ✅ 올바른 해결: "예상 대기시간이 안내되며, 원격 줄서기 시스템을 이용할 수 있다" (고객 경험)

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

    # [Async] 텍스트 필드 업데이트 (extract_action_mappings 결과 필요)
    async def update_text_fields(
        self,
        strategies: List[Dict[str, Any]],
        original_comparison: str,
        original_rag_context: str,
        action_mappings: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        GPT로 comparison과 rag_context를 현재 상태로 업데이트

        Args:
            action_mappings: 미리 추출된 액션 매핑 (병렬 처리 결과)

        Returns:
            {
                "updated_comparison": "업데이트된 텍스트",
                "updated_rag_context": "업데이트된 텍스트"
            }
        """
        if action_mappings:
            print(f"  ✓ {len(action_mappings)}개 매핑 적용 중...")
            for m in action_mappings:
                prob = m.get('problem_expression', '')[:30]
                sol = m.get('solution_expression', '')[:30]
                print(f"    - [{m.get('source_strategy')}] {m.get('issue_keyword')}: \"{prob}...\" → \"{sol}...\"")
        else:
            print("  ⚠️ 매핑 없음")

        # 2단계: 매핑을 직접 적용하여 중간 결과 생성
        intermediate_rag = original_rag_context
        intermediate_comparison = original_comparison

        for mapping in action_mappings:
            problem = mapping.get('problem_expression', '')
            solution = mapping.get('solution_expression', '')
            if problem and solution:
                replaced = False

                # 1. 정확한 매칭 시도
                if problem in intermediate_rag:
                    intermediate_rag = intermediate_rag.replace(problem, solution)
                    print(f"    ✓ rag_context 직접 대체: {mapping.get('issue_keyword')}")
                    replaced = True
                elif problem in intermediate_comparison:
                    intermediate_comparison = intermediate_comparison.replace(problem, solution)
                    print(f"    ✓ comparison 직접 대체: {mapping.get('issue_keyword')}")
                    replaced = True

                # 2. 유사 문장 매칭 시도
                if not replaced:
                    similar_rag, score_rag = self.find_similar_segment(intermediate_rag, problem)
                    similar_comp, score_comp = self.find_similar_segment(intermediate_comparison, problem)

                    if score_rag >= 0.5 and similar_rag:
                        intermediate_rag = intermediate_rag.replace(similar_rag, solution)
                        print(f"    ✓ rag_context 유사 대체 ({score_rag:.0%}): {mapping.get('issue_keyword')}")
                        replaced = True
                    elif score_comp >= 0.5 and similar_comp:
                        intermediate_comparison = intermediate_comparison.replace(similar_comp, solution)
                        print(f"    ✓ comparison 유사 대체 ({score_comp:.0%}): {mapping.get('issue_keyword')}")
                        replaced = True

                if not replaced:
                    print(f"    ⚠️ 매칭 실패 (GPT 폴백): {mapping.get('issue_keyword')}")

        # 직접 대체가 안 된 경우 GPT로 유사 문장 찾아서 대체
        mapping_info = json.dumps(action_mappings, ensure_ascii=False, indent=2) if action_mappings else "매핑 없음"

        # 중간 결과가 변경되었으면 이를 기반으로 진행
        working_rag = intermediate_rag
        working_comparison = intermediate_comparison

        # 이슈 키워드 추출
        issue_keywords = list(set([s.get('issue_keyword', '') for s in strategies if s.get('issue_keyword')]))

        strategy_info = "\n\n".join([
            f"전략 {s['id']}: {s['title']}\n대상 이슈: {s.get('issue_keyword', '없음')}\n내용: {s['content']}"
            for s in strategies
        ])

        issue_info = f"해결 대상 이슈 키워드: {', '.join(issue_keywords)}" if issue_keywords else ""

        # [수정1] 근거 없는 숫자 예시(30분) 제거, 정성 문장 예시로 교체
        prompt = f"""망원동 상권 시뮬레이션용 매장 정보를 업데이트합니다.

# 적용된 전략
{strategy_info}

{issue_info}

# 문제 → 해결 매핑 (반드시 이 매핑대로 대체)
{mapping_info}

# 현재 텍스트

## comparison
{working_comparison}

## rag_context
{working_rag}

# 절대 규칙

## 0. 🎯 매핑 기반 대체 (최우선, 반드시 수행!)
위 "문제 → 해결 매핑"의 각 항목에서:
1. problem_expression과 **유사한 문장**을 원본에서 찾는다
2. 해당 문장을 solution_expression으로 **완전히 대체**한다
3. **절대로 원본 문장을 그대로 두지 않는다**
4. 매핑에 없는 부분만 원본 유지

⚠️ 예시:
- 매핑: "짠맛이 강한 편으로..." → "진한 맛이 특징이며, 염도 조절 가능"
- 원본: "짠맛이 강한 편으로 일부 고객에게는 적응하기 어려울 수 있습니다"
- 결과: "진한 맛이 특징이며, 주문 시 염도 조절(기본/덜짭게)을 선택할 수 있다"
- ❌ 원본 문장 유지 금지!

## 1. 🔍 부정적 팩트의 '대체' 및 '특징 유지' (핵심)
- 적용된 전략에 의해 해결된 문제점(예: 웨이팅 병목)은 전략의 결과물인 '현재의 팩트'로 담백하게 대체하세요.
- **핵심 특징 유지**: '맛(짠맛)', '면의 질감' 등 매장의 고유한 정체성은 **삭제하지 마세요.** 대신, "짜서 힘들다"는 부정적 표현을 전략에 근거하여 **"염도 조절 옵션을 제공한다", "강한 맛이 특징이다"**와 같은 중립적/시스템적 팩트로 전환하세요.
- 에이전트가 "가게가 노력한다"고 느끼게 하는 것이 아니라, **"이 가게는 원래 이런 특징과 시스템을 가졌다"**고 인지하도록 서술해야 합니다.

## 2. ❌ 금지 표현 (절대 사용 금지!)
- 개선되었다, 향상되었다, 변경되었다, 해결되었다, 보완되었다
- 도입했다, 도입하여, 도입되었다 (시계열 변화 암시)
- 이전에, 과거에, 예전에, 예전보다는
- 더 나아졌다, 좋아졌다, 맛이 좋아졌다
- 줄어들었다, 증가했다, 늘어났다
- 최근, 새롭게, 마련
- 그 외 모든 시계열적 변화/흐름 표현

## 3. ❌ 메타 설명 금지 (절대 포함 금지!)
- "(짠맛 불만 제거)", "(웨이팅 개선)", "(서비스 보완)" 같은 괄호 안 설명 금지
- "~를 해결하기 위해", "~를 개선하고자" 같은 의도 설명 금지
- 결과물에는 **현재 상태의 팩트만** 서술, 변경 의도나 배경 설명 없음

## 4. ✅ 반드시 사용 (현재 상태 서술)
- "~이다", "~가 있다", "~을 제공한다", "~이 특징이다"
- "대기 안내 시스템이 있다" 같은 현재형 서술은 허용
- 구체적 팩트만 (전략의 실행 내용을 기반으로)
- 새로운 근거 없는 수치 생성 금지

## 수정 예시
❌ "최근 웨이팅 시스템을 도입하여 대기시간이 개선되었다" (금지어 포함)
❌ "대기 안내 시스템이 운영되고 있다 (웨이팅 불만 해소)" (메타 설명 포함)
✅ "대기 안내 시스템이 있으며, 예상 대기시간이 안내된다" (현재 팩트만)

❌ "이전보다 짠맛이 줄어들어 먹기 편하다" (비교 표현 포함)
❌ "염도 조절 옵션을 제공한다 (짠맛 민감층 배려)" (메타 설명 포함)
✅ "주문 시 염도 조절 옵션(기본/덜짭게)을 선택할 수 있다" (현재의 기능만)

# 출력 (JSON만, 코드 블록 없이)
{{
  "updated_comparison": "현재 상태로 서술된 comparison",
  "updated_rag_context": "현재 상태로 서술된 rag_context"
}}

시간 흐름 표현 절대 금지. 괄호 안 메타 설명 절대 금지. 에이전트에게 전달될 '현재의 진실'만:"""

        messages = [
            {
                "role": "system",
                "content": "시간 흐름 표현을 절대 사용하지 않습니다. 오직 현재 상태만 서술합니다. JSON만 출력합니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            return await self._parse_json_with_retry(messages, max_retries=2, temperature=0.2, max_tokens=2000)
        except Exception as e:
            print(f"⚠️ 텍스트 업데이트 오류: {e}")
            return {
                "updated_comparison": original_comparison,
                "updated_rag_context": original_rag_context
            }

    def validate_no_time_expressions(self, text: str) -> tuple[bool, List[str]]:
        """시간 흐름 표현 및 메타 설명 검증"""
        violations = []

        # 시간 흐름 표현 검증
        for pattern in self.FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(f"[시간표현] {pattern} → {matches}")

        # 메타 설명 검증
        for pattern in self.META_DESCRIPTION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(f"[메타설명] {pattern} → {matches}")

        return (len(violations) == 0, violations)

    # [Async] 메인 전략 적용 함수 - 병렬 처리 적용
    async def apply_strategies(
        self,
        store_json: Dict[str, Any],
        x_report_path: str,
        selected_strategy_ids: List[str]
    ) -> Dict[str, Any]:
        """
        전략을 store JSON에 적용 (비동기 병렬 처리)

        Args:
            store_json: 기존 store JSON (원본)
            x_report_path: X-Report 파일 경로
            selected_strategy_ids: 선택된 전략 ID 리스트 (예: ["S1_1", "S2_2"])

        Returns:
            원본 구조를 유지하면서 변경된 부분만 업데이트된 전체 JSON
        """
        # 1. X-Report 로드 및 이슈/전략 추출
        print(f"📄 X-Report 로딩: {x_report_path}")
        with open(x_report_path, 'r', encoding='utf-8') as f:
            x_report_text = f.read()

        # 이슈 추출
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

        # asyncio.gather로 4개 메서드 병렬 실행
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

        # 5. feature_scores 델타 적용 (동기)
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
        # 6. [순차 처리] update_text_fields는 action_mappings 결과 필요
        # ============================================================
        print("="*60)
        print("🤖 GPT로 텍스트 업데이트 중...")
        print("="*60 + "\n")

        text_updates = await self.update_text_fields(
            selected_strategies,
            original_comparison,
            original_rag_context,
            action_mappings
        )

        # 7. 검증
        print("="*60)
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

        # feature_scores 업데이트 (score 값만 변경)
        for feat, new_score in updated_scores.items():
            if feat in result['review_metrics']['feature_scores']:
                result['review_metrics']['feature_scores'][feat]['score'] = new_score

        # comparison 업데이트
        result['review_metrics']['overall_sentiment']['comparison'] = text_updates.get(
            'updated_comparison', original_comparison
        )

        # rag_context 업데이트
        result['rag_context'] = text_updates.get('updated_rag_context', original_rag_context)

        # critical_feedback 업데이트
        result['critical_feedback'] = updated_critical

        # top_keywords 업데이트
        result['top_keywords'] = updated_keywords

        return result

    # [Async] critical_feedback 업데이트
    async def update_critical_feedback(
        self,
        strategies: List[Dict[str, Any]],
        original_critical: List[str]
    ) -> List[str]:
        """
        전략 적용 후 critical_feedback 업데이트

        - 해결된 문제는 제거하거나 중립적 표현으로 변경
        - 고객 경험 관점으로 서술
        """
        if not original_critical:
            return []

        # 이슈 키워드 추출
        issue_keywords = list(set([s.get('issue_keyword', '') for s in strategies if s.get('issue_keyword')]))

        strategy_info = "\n".join([
            f"- {s['id']}: {s['title']} (이슈: {s.get('issue_keyword', '')})"
            for s in strategies
        ])

        prompt = f"""critical_feedback 목록을 전략 적용 후 상태로 업데이트하세요.

# 적용된 전략
{strategy_info}

# 해결 대상 이슈 키워드
{', '.join(issue_keywords)}

# 현재 critical_feedback
{json.dumps(original_critical, ensure_ascii=False, indent=2)}

# 규칙
1. 전략으로 해결된 문제는 **제거**하거나 **중립적 팩트**로 변경
2. 고객 경험 관점으로 서술 (운영 관점 금지)
3. 시간 흐름 표현 금지 (개선되었다, 해결되었다 등)
4. 여전히 유효한 주의사항만 남김

# 예시
- 원본: "매우 짠 국물 맛으로 인해 일부 고객에게는 부적합할 수 있음"
- 전략: 염도 조절 옵션화
- 변경: "진한 맛이 특징이며, 염도 조절이 가능함"

- 원본: "웨이팅 시간이 길 수 있음"
- 전략: 대기 안내 시스템, 원격 줄서기
- 변경: "예상 대기시간 안내 및 원격 줄서기 이용 가능" 또는 제거

- 원본: "위생 상태에 대한 우려 제기"
- 전략: 청결 강점 리뷰 키워드 유도
- 변경: 제거 (전략으로 해결됨)

# 출력 (JSON 객체, 코드 블록 없이)
{{"items": ["항목1", "항목2", ...]}}

업데이트된 critical_feedback:"""

        messages = [
            {"role": "system", "content": "critical_feedback을 전략 적용 후 상태로 업데이트합니다. JSON만 출력합니다."},
            {"role": "user", "content": prompt}
        ]

        try:
            result = await self._parse_json_with_retry(messages, max_retries=2, temperature=0.2, max_tokens=500)
            return result.get("items", [])
        except Exception as e:
            print(f"⚠️ critical_feedback 업데이트 오류: {e}")
            return original_critical

    # [Async] top_keywords 업데이트
    async def update_top_keywords(
        self,
        strategies: List[Dict[str, Any]],
        original_keywords: List[str]
    ) -> List[str]:
        """
        GPT로 전략 기반 top_keywords 업데이트
        - 전략으로 해결된 부정적 키워드 제거
        - 남은 키워드만 반환

        Args:
            strategies: 적용된 전략 리스트
            original_keywords: 원본 top_keywords

        Returns:
            업데이트된 top_keywords
        """
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

            # 제거된 키워드 로그
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
    """
    전략 적용 메인 함수 (비동기 버전)

    Args:
        store_json_path: 기존 store JSON 파일 경로
        x_report_path: X-Report 파일 경로
        selected_strategy_ids: 선택된 전략 ID 리스트
        api_key: OpenAI API 키
        output_path: 결과 저장 경로 (None이면 저장 안 함)

    Returns:
        전략 적용 결과 dict
    """
    # 1. store JSON 로드
    with open(store_json_path, 'r', encoding='utf-8') as f:
        store_json = json.load(f)

    # 2. 브리지 초기화 및 적용
    bridge = StrategyBridge(api_key=api_key)
    result = await bridge.apply_strategies(store_json, x_report_path, selected_strategy_ids)

    # 3. 저장 (옵션)
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
    """
    전략 적용 메인 함수 (동기 wrapper - 기존 API 호환)

    Args:
        store_json_path: 기존 store JSON 파일 경로
        x_report_path: X-Report 파일 경로
        selected_strategy_ids: 선택된 전략 ID 리스트
        api_key: OpenAI API 키
        output_path: 결과 저장 경로 (None이면 저장 안 함)

    Returns:
        전략 적용 결과 dict
    """
    return asyncio.run(apply_x_report_strategy_async(
        store_json_path,
        x_report_path,
        selected_strategy_ids,
        api_key,
        output_path
    ))


def get_xreport_issues(x_report_path: str, api_key: str = None) -> List[Dict[str, Any]]:
    """
    X-Report에서 이슈 목록만 추출 (전략 선택 전 조회용)

    Args:
        x_report_path: X-Report 파일 경로
        api_key: OpenAI API 키 (필요 없음, 호환성용)

    Returns:
        [
            {
                "top_num": 1,
                "issue_keyword": "짠맛",
                "issue_title": "짠맛 민감층 이탈 가능성",
                "feature": "taste"
            },
            ...
        ]
    """
    bridge = StrategyBridge(api_key=api_key or "dummy")

    with open(x_report_path, 'r', encoding='utf-8') as f:
        x_report_text = f.read()

    return bridge.extract_issues(x_report_text)


def get_xreport_strategies(x_report_path: str, api_key: str = None) -> List[Dict[str, Any]]:
    """
    X-Report에서 전략 목록 추출 (이슈 키워드 포함)

    Args:
        x_report_path: X-Report 파일 경로
        api_key: OpenAI API 키 (필요 없음, 호환성용)

    Returns:
        [
            {
                "id": "S1_1",
                "title": "피크타임 병목 3분해 체크",
                "issue_keyword": "웨이팅",
                "top_num": 1,
                ...
            },
            ...
        ]
    """
    bridge = StrategyBridge(api_key=api_key or "dummy")

    with open(x_report_path, 'r', encoding='utf-8') as f:
        x_report_text = f.read()

    return bridge.extract_strategies(x_report_text)


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import time
    import sys
    from pathlib import Path

    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent

    # 설정 (명령줄 인자 또는 기본값)
    API_KEY = os.getenv("LLM_API_KEY")

    # 기본 파일 경로 (정드린치킨망원점)
    STORE_JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "정드린치킨_전략적용_v2.json")
    X_REPORT_PATH = sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / "X-reports" / "정드린치킨망원점_report.md")
    SELECTED_STRATEGY_IDS = ["S1_1", "S1_2", "S1_3", "S2_1", "S2_2", "S2_3", "S3_1", "S3_2", "S3_3"]

    # 실행 시간 측정
    start_time = time.time()

    # 사전 검증
    if not API_KEY:
        print("❌ 오류: LLM_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  .env 파일에 LLM_API_KEY=your_api_key 를 추가하세요.")
        sys.exit(1)

    if not Path(STORE_JSON_PATH).exists():
        print(f"❌ 오류: store JSON 파일을 찾을 수 없습니다: {STORE_JSON_PATH}")
        sys.exit(1)

    if not Path(X_REPORT_PATH).exists():
        print(f"❌ 오류: X-Report 파일을 찾을 수 없습니다: {X_REPORT_PATH}")
        sys.exit(1)

    print(f"📁 Store JSON: {STORE_JSON_PATH}")
    print(f"📁 X-Report: {X_REPORT_PATH}")
    print(f"🎯 선택된 전략: {SELECTED_STRATEGY_IDS}")
    print()

    # 출력 파일명 생성
    store_name = Path(STORE_JSON_PATH).stem.split("_")[0]
    output_path = f"{store_name}_전략적용_test.json"

    # 실행
    result = apply_x_report_strategy(
        store_json_path=STORE_JSON_PATH,
        x_report_path=X_REPORT_PATH,
        selected_strategy_ids=SELECTED_STRATEGY_IDS,
        api_key=API_KEY,
        output_path=output_path
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
