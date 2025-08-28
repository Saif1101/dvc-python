@echo off
setlocal ENABLEDELAYEDEXPANSION
pushd %~dp0\..

echo === Dataset Preparation: Validate PDFs ===
python src\prep_pdfs.py || goto :error
echo Report written to reports\pdfs_report.*

popd
exit /b 0

:error
echo Failed. See output above.
popd
exit /b 1


