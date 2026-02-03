@echo off
REM Full simulation pipeline
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================
echo Mangwon-dong Agent Simulation
echo ========================================
echo.

cd /d %~dp0\..

echo [1/2] Running simulation...
python scripts/run_simulation.py
if %errorlevel% neq 0 (
    echo Error: Simulation failed
    pause
    exit /b 1
)
echo.

echo [2/2] Running analysis...
python scripts/run_analysis.py
if %errorlevel% neq 0 (
    echo Error: Analysis failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo All tasks complete!
echo ========================================
echo.
echo Output files:
echo - data/output/agents.json
echo - data/output/simulation_result.csv
echo - data/output/visit_log.csv
echo.

pause
