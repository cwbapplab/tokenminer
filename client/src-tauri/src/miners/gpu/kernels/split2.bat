@echo off
REM Route 1 follow-up. split=2 halves the accumulator (48 registers/lane at 12 columns) but duplicates
REM every A ldmatrix, which measured 115.6 against 137.9 at one block per SM -- a net loss. The idea
REM is only worth it if the freed registers buy a second resident block, so this pairs the split with
REM POW_MIN_BLOCKS=2 and with wider rectangles that only become buildable once the registers are freed.
REM
REM Split 3 and 4 cannot run at 16 rows: POW_THREADS = 16 * SPLIT * 32 exceeds the 1536 threads an SM
REM allows, so the launch is refused. Rectangles with fewer rows can split further.
REM usage: split2.bat [TPR]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set TPR=%1
if "%TPR%"=="" set TPR=8192

echo --- 1 block/SM: what the split alone costs ---
call :s 16 12 1 1
call :s 16 12 2 1
echo.
echo --- 2 blocks/SM demanded: split makes this buildable without heavy spilling ---
call :s 16 12 2 2
call :s 16 8 2 2
call :s 8 8 2 2
echo.
echo --- fewer rows allow a deeper split within the 1536-thread limit ---
call :s 8 12 2 2
call :s 8 12 3 2
call :s 8 16 2 2
call :s 4 12 4 2
echo.
echo --- wider rectangles at 1 block/SM, split 1, as the control ---
call :s 16 10 1 1
call :s 16 8 1 1
exit /b 0

:s
set R=%~1
set C=%~2
set SP=%~3
set MB=%~4
%NVCC% -O3 -arch=sm_120a -DPOW_BLOCK_ROWS=%R% -DPOW_BLOCK_COLS=%C% -DPOW_COL_SPLIT=%SP% ^
  -DPOW_MIN_BLOCKS=%MB% -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 ^
  -ccbin "%CCBIN%" -o bench_s2.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo %R%x%C% split=%SP% minblk=%MB%   BUILD FAILED
  exit /b 0
)
set V=
for /f "tokens=2" %%t in ('bench_s2.exe %TPR% ^| findstr /B /C:"single:"') do set V=%%t
if "%V%"=="" (echo %R%x%C% split=%SP% minblk=%MB%   no fit) else (echo %R%x%C% split=%SP% minblk=%MB%   !V! TH/s)
exit /b 0