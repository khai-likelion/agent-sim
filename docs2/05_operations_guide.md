# 🛠️ 운영 및 개발 가이드 (Ops Guide)

## 1. 환경 설정
- **기본 환경**: Python 3.9+
- **필수 환경 변수 (`.env`)**:
    - `OPENAI_API_KEY`: X-Report 생성 및 분석용.
    - `OPENAI_MODEL_NAME`: `gpt-5.2` (최신 분석 모델).
    - `GEMINI_API_KEY`: 시뮬레이션 엔진용 (Gemini 2.0 Flash 권장).
    - `VITE_KAKAO_MAP_API_KEY`: 프론트엔드 지도 렌더링용.

## 2. 주요 실행 스크립트

### X-Report 생성
```bash
# 특정 매장 분석 리포트 생성
python report_engine.py
# (매장명 입력 예: 돼지야)
```

### 시뮬레이션 (전/후 비교)
```bash
# 특정 매장에 대한 전략 전/후 시뮬레이션 통합 실행
python scripts/run_before_after_sim.py --target-store 돼지야
```

### 배치 시뮬레이션 (다수 매장)
```bash
# 10개 핵심 타겟 매장에 대해 일괄 시뮬레이션 실행
python scripts/run_batch_10stores.py
```

### 대시보드 실행
```bash
# 프론트엔드 개발 서버 실행
cd lovelop-frontend
npm run dev
```

## 3. 운영상 유의사항 (Troubleshooting)

### 인코딩 이슈 (Windows)
- 윈도우 기본 터미널(CP949)에서는 이모지 등 특수 문자가 출력될 때 `UnicodeEncodeError`가 발생할 수 있습니다.
- `report_engine.py` 등 주요 엔진은 이를 방지하기 위해 텍스트 기반 표시자(`[OK]`, `[ERROR]`)를 사용하도록 안정화되었습니다.
- 가급적 유니코드를 지원하는 터미널(VS Code 내장 터미널, Git Bash 등) 사용을 권장합니다.

### API Rate Limit
- 다수의 에이전트가 동시에 LLM을 호출하므로 API 제한에 걸릴 수 있습니다.
- 시뮬레이션 설정의 `max_concurrent_llm_calls` 값을 조정하여 동시 호출 수를 제어하십시오.

## 4. 버전 관리 정책
- **v1.x**: 기능 구현 및 로직 고도화 단계.
- **v2.x**: 안정성 강화 및 UI/UX 시각화 완성 단계.
- 모든 주요 변경 사항은 `walkthrough.md`에 기록됩니다.

---
*Ops Guide: 안정적인 서비스 운영을 지원합니다.*
