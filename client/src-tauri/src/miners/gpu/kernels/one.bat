@echo off
REM Time the real search kernel at one geometry, reporting the single-launch TH/s.
REM usage: one.bat ROWS COLS STAGES [TILEWIDTH] [L2GROUP]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"

set R=%~1
set C=%~2
set S=%~3
set TPR=%~4
set G=%~5
if "%TPR%"=="" set TPR=8192
if "%G%"=="" set G=1

%NVCC% -O3 -arch=sm_120a -DPOW_BLOCK_ROWS=%R% -DPOW_BLOCK_COLS=%C% -DPOW_STAGES=%S% ^
  -DPOW_L2_GROUP=%G% -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 ^
  -ccbin "%CCBIN%" -o bench_one.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo %R%x%C% s%S% g%G%   BUILD FAILED
  exit /b 0
)
echo === %R%x%C% stages=%S% l2group=%G% tpr=%TPR% ===
bench_one.exe %TPR% 2>&1 | findstr /B /C:"single" /C:"  block rect" /C:"  regs=" /C:"smem"
exit /b 0