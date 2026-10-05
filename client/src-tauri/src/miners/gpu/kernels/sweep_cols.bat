@echo off
REM Sweep POW_BLOCK_COLS for the shipped 16-row geometry. Wider columns mean fewer passes over
REM the A operand, which is the dominant DRAM traffic (16x8 reads A 8x per block-strip sweep).
REM usage: sweep_cols.bat [TPR]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set TPR=%1
if "%TPR%"=="" set TPR=8192

for %%C in (4 8 10 12 14 16) do call :c %%C
exit /b 0

:c
set C=%~1
%NVCC% -O3 -arch=sm_120a -DPOW_BLOCK_ROWS=16 -DPOW_BLOCK_COLS=%C% -DPOW_STAGES=2 ^
  -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 -ccbin "%CCBIN%" ^
  -o bench_c.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo cols=%C%   BUILD FAILED
  exit /b 0
)
set V=
for /f "tokens=2" %%t in ('bench_c.exe %TPR% ^| findstr /B /C:"single:"') do set V=%%t
if "%V%"=="" (echo cols=%C%   does not fit) else (echo cols=%C%   !V! TH/s)
exit /b 0