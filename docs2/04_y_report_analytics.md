# 📊 Y-Report: 비교 분석 및 성과 지표

## 1. 개요
Y-Report는 전략 적용 전(Sim 1)과 후(Sim 2)의 시뮬레이션 결과를 비교하여, 제안된 솔루션이 매장에 미치는 영향을 다각도로 분석합니다.

## 2. 핵심 분석 지표 (11대 지표)
보고서는 다음의 수치 및 시각화 데이터를 제공합니다.

1.  **Overview**: 총 방문 수 변화(%) 및 상권 내 시장 점유율 데이터.
2.  **Rating Spread (KDE)**: 평점 평균을 넘어 분포의 변화(`맛/가성비/분위기`)를 커널 밀도 추정으로 시각화.
3.  **Hourly Traffic**: 시간대별 방문 트래픽 변화 (Area Chart 및 Peak Slot 추적).
4.  **Generation Impact**: Z대에서 시니어 세대까지, 세대별 고객 증감 비율 분석.
5.  **Purpose Analysis**: 방문 목적(생활베이스, 사적모임, 공적모임, 가족모임)별 비중 및 만족도.
6.  **Retention**: 기존 고객 유지율(재방문)과 신규 유입 및 이탈(Churn) 데이터.
7.  **Radar Chart**: 상권 내 경쟁 매장 대비 성과(방문수, 평점, 재방문, 리뷰수) 정규화 비교.
8.  **Agent Type**: 상주 인구 대비 유동 인구 유입 비율 변화.
9.  **Gender Composition**: 남성/여성/혼성 구성비 변화.
10. **Cross-tab Heatmap**: 세대와 방문 목적을 교차 분석하여 정밀한 고객 이동 파악.
11. **LLM Insight Summary**: 위 데이터를 종합하여 Gemini/GPT가 인간 언어로 정리한 최종 인사이트.

## 3. 솔루션 안전성 진단 (Safety Diagnosis)
전략이 매출에만 집중하여 고객 만족도나 특정 세대 이탈과 같은 **부작용(Trade-off)**을 초래하지 않는지 검증합니다.
- **Safety Score (0~100)**: 역효과 건수에 따른 위험도 산출.
- **Trade-off Analysis**: 얻은 것(Gain)과 잃은 것(Loss)을 명확히 대조하여 의사결정 보조.

## 4. 데이터 연동 (Strategy Bridge)
- `X_to_Sim.py` 모듈을 통해 X-Report에서 제안된 텍스트 기반 솔루션을 시뮬레이션 엔진이 이해할 수 있는 매장 속성(JSON)으로 자동 변환하여 반영합니다.

---
*Y-Report: 결과로 증명하는 분석 데이터*
