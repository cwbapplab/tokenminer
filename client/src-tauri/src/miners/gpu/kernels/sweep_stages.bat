@echo off
REM Sweep the pipeline depth (POW_STAGES). The kernel is issue-bound at 1 block/SM, so the lever is
REM hiding the cp.async latency and the BAR.SYNC behind more multiply work. More stages costs shared
REM memory; at 16x12 with BK=64 one stage is (16+12)*16*80 = 35,328 B and sT is 12,288 B, so:
REM   s2 = 82,944   s3 = 118,272 (over the 101,376 B limit)
REM A deeper pipeline therefore needs a narrower rectangle to fit, which the sweep tries in pairs.
REM usage: sweep_stages.bat [TPR]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set TPR=%1
if "%TPR%"=="" set TPR=8192

call :t 16 12 2
call :t 12 12 2
call :t 16 12 1
call :t 16 10 2
call :t 12 8 2
call :t 16 8 2
exit /b 0

:t
set R=%~1
set C=%~2
set S=%~3
%NVCC% -O3 -arch=sm_120a -DPOW_BLOCK_ROWS=%R% -DPOW_BLOCK_COLS=%C% -DPOW_STAGES=%S% ^
  -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 -ccbin "%CCBIN%" ^
  -o bench_st.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo %R%x%C% stages=%S%   BUILD FAILED
  exit /b 0
)
set V=
for /f "tokens=2" %%t in ('bench_st.exe %TPR% ^| findstr /B /C:"single:"') do set V=%%t
if "%V%"=="" (echo %R%x%C% stages=%S%   no fit) else (echo %R%x%C% stages=%S%   !V! TH/s)
exit /b 0