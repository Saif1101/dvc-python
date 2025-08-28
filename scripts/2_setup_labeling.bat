@echo off
setlocal ENABLEDELAYEDEXPANSION
pushd %~dp0\..

echo === Prepare Staging for Labeling ===
set FORMAT=%1
if "%FORMAT%"=="" (
  python src\labeling_setup.py || goto :error
) else (
  python src\labeling_setup.py --format %FORMAT% || goto :error
)

echo Staging ready under data\staging
popd
exit /b 0

:error
echo Failed. See output above.
popd
exit /b 1


