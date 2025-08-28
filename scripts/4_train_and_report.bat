@echo off
setlocal ENABLEDELAYEDEXPANSION
pushd %~dp0\..

echo === Queue Training via DVC (optional) or run directly ===
if "%1"=="queue" (
  dvc exp run --queue || goto :error
  dvc queue start || goto :error
) else (
  python src\train.py || goto :error
)

echo Training complete. Reports under reports\ and predictions_report.csv
popd
exit /b 0

:error
echo Failed. See output above.
popd
exit /b 1


