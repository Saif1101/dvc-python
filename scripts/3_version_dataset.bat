@echo off
setlocal ENABLEDELAYEDEXPANSION
pushd %~dp0\..

echo === Version Dataset (new or extend) ===
set MODE=%1
set SUFFIX=%2
if "%MODE%"=="" (
  python src\version_dataset.py || goto :error
) else if "%MODE%"=="extend" (
  if "%SUFFIX%"=="" (
    python src\version_dataset.py --mode extend || goto :error
  ) else (
    python src\version_dataset.py --mode extend --suffix %SUFFIX% || goto :error
  )
) else (
  python src\version_dataset.py --mode new || goto :error
)

popd
exit /b 0

:error
echo Failed. See output above.
popd
exit /b 1


