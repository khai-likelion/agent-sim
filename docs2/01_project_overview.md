# 🚀 Lovelop 프로젝트 개요 (Overview)

## 1. 개요
Lovelop은 소상공인을 위한 **AI 기반 비즈니스 솔루션 플랫폼**입니다. 매장의 데이터를 심층 분석하여 맞춤형 전략(X-Report)을 제안하고, 해당 전략을 적용했을 때의 결과를 시뮬레이션(Y-Report)하여 리스크를 최소화하고 매출을 극대화하는 것을 목표로 합니다.

## 2. 핵심 시스템 구조
본 프로젝트는 크게 세 가지 레이어로 구성됩니다.

### 🏛️ Frontend (Dashboard)
- **기술 스택**: React, Vite, Tailwind CSS, Lucide React
- **주요 기능**:
    - **Map View**: 카카오맵 API를 활용한 매장 위치 및 시뮬레이션 에이전트 동선 시각화 (Pydeck 활용).
    - **X-Report Modal**: GPT-5.2가 생성한 Markdown 리소스의 전문적인 렌더링 (`@tailwindcss/typography` 적용).
    - **Y-Report Dashboard**: 시뮬레이션 전(Sim 1) 후(Sim 2) 데이터를 비교하여 지표화 (차트 및 통계).

### 🧠 Analysis Engine (X-Report)
- **핵심 모듈**: `report_engine.py` (GPTReportGenerator)
- **기술 스택**: Python, OpenAI API (**GPT-5.2**)
- **주요 기능**:
    - 매장별 리뷰, 매출, 인구 데이터를 심층 분석하여 SWOT 분석 및 개선 솔루션 도출.
    - Markdown 형식의 분석 보고서 자동 생성 및 결과 JSON 저장.

### 🎢 Simulation Engine (Y-Report / Agent-Sim)
- **핵심 모듈**: `run_generative_simulation.py`, `action_algorithm.py`
- **기술 스택**: Python, Gemini 2.0 Flash/GPT-4o, Asyncio
- **주요 기능**:
    - **지능형 에이전트**: 페르소나별(Z1~S세대, 상주/유동) 독립적 의사결정 모델.
    - **Softmax 가중 샘플링**: 롱테일 매장 방문 기회를 확보한 현실적인 매장 선택 알고리즘.
    - **AB 테스트**: 전략 적용 전과 후의 가상 상권을 7일간 시뮬레이션하여 데이터 비교.

## 3. 주요 발전 사항 (Milestones)
- **v1.0 - v1.9**: 기본 시뮬레이션 엔진 구축 및 에이전트 고도화 (28개 항목 개선).
- **v2.0 (Current)**: 
    - **X-Report 고도화**: GPT-5.2 모델 정식 채택 및 리포트 가독성 완성 (Prose 스타일 렌더링).
    - **엔진 안정화**: 윈도우 환경 Unicode 인코딩 및 데이터 언패킹 오류 전면 해결.
    - **UI/UX 정밀화**: 브랜드 정체성을 반영한 로고 최적화 및 사이드바 레이아웃 개선.

---
*Lovelop: 데이터로 예측하고 전략으로 승부하는 소상공인 파트너*
