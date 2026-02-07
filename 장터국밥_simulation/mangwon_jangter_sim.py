"""
장터국밥 X_Report 기반 Before/After 시뮬레이션
- 상권X매출X리뷰.json에서 장터국밥 데이터 로드
- X_Report 개선사항 적용
- 장터국밥 방문자만 리뷰+평점 생성
- 타 식당 방문자는 로그만
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import config
from mangwon_pure_llm import (
    init_llm_client, create_demo_agents, load_restaurants_from_csv,
    run_pure_llm_simulation, SEGMENT_DATA, export_agent_profiles
)

# 로그 파일
LOG_FILE = Path(__file__).parent / "latest_run.log"


def log(message: str):
    """로그 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def load_jangter_data() -> Dict[str, Any]:
    """상권X매출X리뷰.json에서 장터국밥 데이터 로드"""
    data_path = Path(__file__).parent.parent / "상권X매출X리뷰.json"

    log(f"Loading 상권X매출X리뷰.json from {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 장터국밥 찾기 (망원장터국밥 또는 장터국밥)
    for item in data:
        if "analysis_reports" in item:
            for report in item["analysis_reports"]:
                store_name = report.get("store_name", "")
                if "장터국밥" in store_name:
                    log(f"Found {store_name} data!")
                    return report

    raise ValueError("장터국밥 데이터를 찾을 수 없습니다.")


def apply_x_report_improvements(original_data: Dict) -> Dict:
    """
    X_Report 개선사항 적용
    - price_value: 0.6 → 0.7 (세트 구성으로 가성비 체감 개선)
    - service: 0.7 → 0.8 (국물 안내 표준화로 서비스 개선)
    - critical_feedback 업데이트
    """
    improved_data = json.loads(json.dumps(original_data))  # Deep copy

    # Feature scores 개선
    improved_data["review_metrics"]["feature_scores"]["price_value"]["score"] = 0.7
    improved_data["review_metrics"]["feature_scores"]["price_value"]["label"] = "좋음"

    improved_data["review_metrics"]["feature_scores"]["service"]["score"] = 0.8
    improved_data["review_metrics"]["feature_scores"]["service"]["label"] = "매우 좋음"

    # Overall sentiment 향상
    improved_data["review_metrics"]["overall_sentiment"]["score"] = 0.80
    improved_data["review_metrics"]["overall_sentiment"]["label"] = "매우 긍정적"
    improved_data["review_metrics"]["overall_sentiment"]["comparison"] = "X_Report 개선 후 더욱 긍정적인 반응을 보임"

    # Critical feedback 업데이트
    improved_data["critical_feedback"] = [
        "양념장 안내 표준화로 국물 맹물 이슈 해결됨",
        "세트 구성 도입으로 가성비 체감 개선"
    ]

    # Top keywords에 개선 관련 키워드 추가
    if "세트 메뉴" not in improved_data["top_keywords"]:
        improved_data["top_keywords"].append("세트 메뉴")
    if "양념장 안내" not in improved_data["top_keywords"]:
        improved_data["top_keywords"].append("양념장 안내")

    # RAG context 업데이트
    improved_data["rag_context"] += " X_Report 개선 후: 양념장 사용 안내를 표준화하여 국물 맹물 이슈를 해결했으며, 세트 구성 도입으로 가성비 체감이 크게 개선되었습니다. 서비스와 가격가치 점수가 상승했습니다."

    log("Applied X_Report improvements:")
    log(f"  - price_value: 0.6 → 0.7")
    log(f"  - service: 0.7 → 0.8")
    log(f"  - overall_sentiment: 0.75 → 0.80")

    return improved_data


def run_jangter_simulation(jangter_data: Dict, n_agents: int = 30, n_days: int = 5,
                           seed: int = 42, apply_improvements: bool = False):
    """
    장터국밥 중심 시뮬레이션
    - 장터국밥 방문자: 리뷰 + 평점
    - 타 식당 방문자: 로그만
    """
    log(f"\n{'='*70}")
    log(f"Starting {'AFTER' if apply_improvements else 'BEFORE'} Simulation")
    log(f"{'='*70}\n")

    # API 클라이언트 초기화
    api_key = config.OPENAI_API_KEY
    if not api_key or "your_openai_api_key" in api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        log("ERROR: API key not found")
        return None

    init_llm_client(api_key)
    log("[OK] LLM client initialized")

    # 식당 데이터 로드
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "data" / "raw" / "stores.csv"

    log(f"Loading stores.csv from {csv_path}")

    # 망원장터국밥이 포함되도록 강제로 로드
    import pandas as pd
    df = pd.read_csv(str(csv_path))

    # 망원장터국밥을 포함하는 행 찾기
    jangter_rows = df[df['장소명'] == '망원장터국밥']

    # 나머지에서 샘플링
    other_rows = df[df['장소명'] != '망원장터국밥']
    sampled_rows = other_rows.sample(n=min(49, len(other_rows)), random_state=42)

    # 합치기
    final_df = pd.concat([jangter_rows, sampled_rows], ignore_index=True)

    # 임시 파일로 저장하고 로드
    temp_csv = script_dir / "temp_stores.csv"
    final_df.to_csv(temp_csv, index=False)

    restaurants = load_restaurants_from_csv(str(temp_csv), sample_size=None)
    temp_csv.unlink()  # 임시 파일 삭제

    log(f"[OK] Loaded {len(restaurants)} restaurants (including 망원장터국밥)")

    # 장터국밥 찾기 및 데이터 적용
    jangter_rest = None
    for r in restaurants:
        if "장터국밥" in r.name:  # 망원장터국밥 또는 장터국밥
            jangter_rest = r
            original_name = r.name  # 원래 이름 저장

            # 데이터 매핑
            r.hygiene_level = jangter_data["review_metrics"]["feature_scores"]["cleanliness"]["score"]
            r.visual_score = 0.65  # 시장 내 매장, 보통
            r.comfort_level = 0.65  # 아날로그 감성, 보통
            r.solo_accessibility = 0.75  # 혼밥 가능
            r.stimulation_level = jangter_data["review_metrics"]["feature_scores"]["taste"]["score"]

            # Review metrics
            r.review_count = 15  # RAG context에서 언급된 리뷰 수
            r.avg_review_score = jangter_data["review_metrics"]["overall_sentiment"]["score"]

            log(f"[OK] Found and configured {original_name}")
            log(f"  - hygiene: {r.hygiene_level:.2f}")
            log(f"  - taste: {r.stimulation_level:.2f}")
            log(f"  - overall_score: {r.avg_review_score:.2f}")
            break

    if not jangter_rest:
        log("WARNING: 장터국밥을 찾을 수 없습니다. 첫 번째 한식집을 사용하고 이름을 장터국밥으로 변경합니다.")
        for r in restaurants:
            if "한식" in r.업종:
                jangter_rest = r
                r.name = "장터국밥"
                # 데이터 매핑
                r.hygiene_level = jangter_data["review_metrics"]["feature_scores"]["cleanliness"]["score"]
                r.visual_score = 0.65
                r.comfort_level = 0.65
                r.solo_accessibility = 0.75
                r.stimulation_level = jangter_data["review_metrics"]["feature_scores"]["taste"]["score"]
                r.review_count = 15
                r.avg_review_score = jangter_data["review_metrics"]["overall_sentiment"]["score"]
                log(f"[OK] Using {r.name} as fallback")
                break

    # 에이전트 생성
    center_loc = (126.906, 37.556)  # 망원동 중심
    agents = create_demo_agents(n_agents, seed, center_loc)
    log(f"[OK] Created {len(agents)} agents")

    # 시뮬레이션 실행
    metrics = run_pure_llm_simulation(
        agents, restaurants, n_days, seed,
        strategy=None, verbose=False
    )

    # 장터국밥 방문 로그 및 리뷰 생성
    jangter_visits = []
    other_visits = []

    for decision_log in metrics["decision_logs"]:
        agent_id = decision_log["agent_id"]
        segment = decision_log["segment"]
        restaurant = decision_log["restaurant"]
        day = decision_log["day"]
        timeblock = decision_log["timeblock"]
        reasoning = decision_log["reasoning"]
        satisfaction = decision_log["satisfaction"]

        visit_info = {
            "day": day,
            "timeblock": timeblock,
            "agent_id": agent_id,
            "segment": segment,
            "restaurant": restaurant,
            "reasoning": reasoning,
            "satisfaction": satisfaction
        }

        if "장터국밥" in restaurant:
            jangter_visits.append(visit_info)
            log(f"Day {day} {timeblock}: Agent#{agent_id} ({segment}) → 장터국밥 | 만족도 {satisfaction:.2f}")
            log(f"  이유: {reasoning}")
        else:
            other_visits.append(visit_info)
            log(f"Day {day} {timeblock}: Agent#{agent_id} ({segment}) → {restaurant}")

    # 장터국밥 리뷰만 필터링
    jangter_reviews = [
        r for r in metrics["reviews_generated"]
        if "장터국밥" in r["restaurant_name"]
    ]

    log(f"\n{'='*70}")
    log(f"Simulation Results ({'AFTER' if apply_improvements else 'BEFORE'})")
    log(f"{'='*70}")
    log(f"Total visits: {metrics['total_visits']}")
    log(f"장터국밥 visits: {len(jangter_visits)}")
    log(f"Other restaurant visits: {len(other_visits)}")
    log(f"장터국밥 reviews generated: {len(jangter_reviews)}")

    if jangter_reviews:
        log(f"\n[장터국밥 리뷰 샘플]")
        for rev in jangter_reviews[:5]:
            log(f"  - Agent#{rev['agent_id']} ({rev['segment']})")
            log(f"    만족도: {rev['satisfaction']:.2f}")
            log(f"    리뷰: \"{rev['review_text']}\"")

    return {
        "metrics": metrics,
        "jangter_visits": jangter_visits,
        "other_visits": other_visits,
        "jangter_reviews": jangter_reviews,
        "jangter_data": jangter_data
    }


def main():
    # 로그 파일 초기화
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("="*70)
    log("장터국밥 X_Report 기반 Before/After 시뮬레이션")
    log("="*70)

    # 장터국밥 데이터 로드
    jangter_data_original = load_jangter_data()

    log("\n[BEFORE] Original 장터국밥 Data:")
    log(f"  - Overall Sentiment: {jangter_data_original['review_metrics']['overall_sentiment']['score']:.2f}")
    log(f"  - Taste: {jangter_data_original['review_metrics']['feature_scores']['taste']['score']:.2f}")
    log(f"  - Price Value: {jangter_data_original['review_metrics']['feature_scores']['price_value']['score']:.2f}")
    log(f"  - Cleanliness: {jangter_data_original['review_metrics']['feature_scores']['cleanliness']['score']:.2f}")
    log(f"  - Service: {jangter_data_original['review_metrics']['feature_scores']['service']['score']:.2f}")

    # X_Report 개선사항 적용
    jangter_data_improved = apply_x_report_improvements(jangter_data_original)

    log("\n[AFTER] Improved 장터국밥 Data:")
    log(f"  - Overall Sentiment: {jangter_data_improved['review_metrics']['overall_sentiment']['score']:.2f}")
    log(f"  - Taste: {jangter_data_improved['review_metrics']['feature_scores']['taste']['score']:.2f}")
    log(f"  - Price Value: {jangter_data_improved['review_metrics']['feature_scores']['price_value']['score']:.2f}")
    log(f"  - Cleanliness: {jangter_data_improved['review_metrics']['feature_scores']['cleanliness']['score']:.2f}")
    log(f"  - Service: {jangter_data_improved['review_metrics']['feature_scores']['service']['score']:.2f}")

    # BEFORE 시뮬레이션
    log("\n" + "="*70)
    log("Running BEFORE Simulation (Original Data)")
    log("="*70)
    results_before = run_jangter_simulation(
        jangter_data_original,
        n_agents=30,
        n_days=5,
        seed=42,
        apply_improvements=False
    )

    # AFTER 시뮬레이션
    log("\n" + "="*70)
    log("Running AFTER Simulation (Improved Data)")
    log("="*70)
    results_after = run_jangter_simulation(
        jangter_data_improved,
        n_agents=30,
        n_days=5,
        seed=43,  # 다른 seed로 변화 확인
        apply_improvements=True
    )

    # 결과 비교
    log("\n" + "="*70)
    log("BEFORE vs AFTER Comparison")
    log("="*70)

    before_visits = len(results_before["jangter_visits"])
    after_visits = len(results_after["jangter_visits"])
    before_reviews = len(results_before["jangter_reviews"])
    after_reviews = len(results_after["jangter_reviews"])

    log(f"\n[장터국밥 방문 수]")
    log(f"  BEFORE: {before_visits}회")
    log(f"  AFTER:  {after_visits}회")
    log(f"  변화:   {after_visits - before_visits:+}회 ({(after_visits/max(before_visits,1)-1)*100:+.1f}%)")

    log(f"\n[장터국밥 리뷰 수]")
    log(f"  BEFORE: {before_reviews}개")
    log(f"  AFTER:  {after_reviews}개")
    log(f"  변화:   {after_reviews - before_reviews:+}개")

    if results_before["jangter_reviews"]:
        avg_sat_before = sum(r["satisfaction"] for r in results_before["jangter_reviews"]) / len(results_before["jangter_reviews"])
        log(f"\n[BEFORE 평균 만족도]: {avg_sat_before:.2f}")

    if results_after["jangter_reviews"]:
        avg_sat_after = sum(r["satisfaction"] for r in results_after["jangter_reviews"]) / len(results_after["jangter_reviews"])
        log(f"[AFTER 평균 만족도]: {avg_sat_after:.2f}")

    log("\n" + "="*70)
    log("Simulation Complete! Check latest_run.log for full details.")
    log("="*70)


if __name__ == "__main__":
    main()
