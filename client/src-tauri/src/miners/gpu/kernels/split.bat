@echo off
REM Route 1: split a tile row's columns across POW_COL_SPLIT warps, so each warp's accumulator is
REM 8*POW_COLS_PER_WARP registers instead of 8*POW_BLOCK_COLS. At 12 columns and a split of 2 that is
REM 48 accumulator registers per lane instead of 96, which is what it takes to fit under the 64-register
REM ceiling that two blocks per SM requires.
REM
REM POW_COL_SPLIT must divide POW_BLOCK_COLS (each warp owns a whole number of tile columns) and
REM POW_COL_SPLIT * 4 must be <= 32 so a warp's four-lanes-per-tile hash still fits inside it.
REM usage: split.bat [TPR]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set TPR=%1
if "%TPR%"=="" set TPR=8192

echo --- 16x12, split sweep (baseline split=1) ---
call :s 16 12 1 1
call :s 16 12 2 1
call :s 16 12 3 1
call :s 16 12 4 1
echo.
echo --- with 2 blocks/SM demanded (split frees the registers) ---
call :s 16 12 2 2
call :s 16 12 3 2
call :s 16 8 2 2
call :s 16 8 1 2
exit /b 0

:s
set R=%~1
set C=%~2
set SP=%~3
set MB=%~4
%NVCC% -O3 -arch=sm_120a -DPOW_BLOCK_ROWS=%R% -DPOW_BLOCK_COLS=%C% -DPOW_COL_SPLIT=%SP% ^
  -DPOW_MIN_BLOCKS=%MB% -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 ^
  -ccbin "%CCBIN%" -o bench_sp.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo %R%x%C% split=%SP% minblk=%MB%   BUILD FAILED
  exit /b 0
)
set V=
for /f "tokens=2" %%t in ('bench_sp.exe %TPR% ^| findstr /B /C:"single:"') do set V=%%t
if "%V%"=="" (echo %R%x%C% split=%SP% minblk=%MB%   no fit) else (echo %R%x%C% split=%SP% minblk=%MB%   !V! TH/s)
exit /b 0