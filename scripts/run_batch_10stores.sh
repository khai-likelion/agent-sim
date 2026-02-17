#!/bin/bash
# 10개 매장 전략 전/후 비교 시뮬레이션 배치 실행
# 160명 에이전트, 7일, 시드 42
# 네트워크 끊김 시 해당 매장부터 재시작 가능 (이미 완료된 매장은 건너뜀)

set -e
cd "$(dirname "$0")/.."

AGENTS=160
DAYS=7
SEED=42

# 매장명(JSON 파일명)과 report MD 매핑
# 형식: "JSON파일명|report파일명|출력폴더명"
STORES=(
  "돼지야|돼지야_report.md|돼지야"
  "망원부자부대찌개|망원부자부대찌개_report.md|망원부자부대찌개"
  "메가MGC커피 망원망리단길점|메가MGC커피망원망리단길점_report.md|메가MGC커피망원망리단길점"
  "반했닭옛날통닭 망원점|반했닭옛날통닭망원점_report.md|반했닭옛날통닭망원점"
  "오늘요거|오늘요거_report.md|오늘요거"
  "육장|육장_report.md|육장"
  "육회by유신 망원점|육회by유신망원점_report.md|육회by유신망원점"
  "전조|전조_report.md|전조"
  "정드린치킨 망원점|정드린치킨망원점_report.md|정드린치킨망원점"
  "크리머리|크리머리_report.md|크리머리"
)

TOTAL=${#STORES[@]}
COMPLETED=0
FAILED=0

echo "============================================================"
echo "10개 매장 전략 전/후 비교 시뮬레이션 배치"
echo "에이전트: ${AGENTS}명 / 기간: ${DAYS}일 / 시드: ${SEED}"
echo "============================================================"

for i in "${!STORES[@]}"; do
  IFS='|' read -r STORE_NAME REPORT_FILE OUTPUT_PREFIX <<< "${STORES[$i]}"
  IDX=$((i + 1))

  # 이미 완료된 매장 건너뛰기 (before/after 폴더 모두 존재하면)
  BEFORE_DIR="data/output/${OUTPUT_PREFIX}_before"
  AFTER_DIR="data/output/${OUTPUT_PREFIX}_after"

  if [ -f "${BEFORE_DIR}/visit_log.csv" ] && [ -f "${AFTER_DIR}/visit_log.csv" ]; then
    echo ""
    echo "[${IDX}/${TOTAL}] ${STORE_NAME} - 이미 완료됨, 건너뜀"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  echo ""
  echo "============================================================"
  echo "[${IDX}/${TOTAL}] ${STORE_NAME} 시뮬레이션 시작"
  echo "============================================================"

  # 최대 3회 재시도
  for RETRY in 1 2 3; do
    if python scripts/run_before_after_sim.py \
      --target-store "${STORE_NAME}" \
      --report "${REPORT_FILE}" \
      --agents ${AGENTS} \
      --days ${DAYS} \
      --seed ${SEED} \
      --output-prefix "${OUTPUT_PREFIX}" \
      -y; then
      COMPLETED=$((COMPLETED + 1))
      echo "[${IDX}/${TOTAL}] ${STORE_NAME} 완료!"
      break
    else
      echo "[${IDX}/${TOTAL}] ${STORE_NAME} 실패 (시도 ${RETRY}/3)"
      if [ $RETRY -eq 3 ]; then
        FAILED=$((FAILED + 1))
        echo "[${IDX}/${TOTAL}] ${STORE_NAME} 3회 실패, 다음 매장으로 이동"
      else
        echo "30초 후 재시도..."
        sleep 30
      fi
    fi
  done
done

echo ""
echo "============================================================"
echo "배치 완료: 성공 ${COMPLETED}/${TOTAL}, 실패 ${FAILED}/${TOTAL}"
echo "============================================================"
echo "결과 폴더: data/output/{매장명}_before, data/output/{매장명}_after"
