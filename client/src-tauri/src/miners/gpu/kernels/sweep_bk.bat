@echo off
REM Sweep the staged k-width (POW_BK). The barrier count per unit of multiply-accumulates scales as
REM 1/POW_BK, which is why 32 -> 64 was historically worth +77%, so 128 is the next step on that axis.
REM usage: sweep_bk.bat [TPR]
setlocal enabledelayedexpansion
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set TPR=%1
if "%TPR%"=="" set TPR=8192

call :bk 32
call :bk 64
call :bk 128
exit /b 0

:bk
set K=%~1
rem the harness's DEV_BK must match the kernel's POW_BK or shared memory is under-requested
%NVCC% -O3 -arch=sm_120a -DPOW_BK=%K% -DDEV_BK=%K% -diag-suppress 550 -diag-suppress 177 ^
  -Xcompiler /wd4477 -ccbin "%CCBIN%" -o bench_bk.exe bench_real.cu >nul 2>&1
if errorlevel 1 (
  echo POW_BK=%K%   BUILD FAILED
  exit /b 0
)
set V=
set S=
for /f "tokens=2" %%t in ('bench_bk.exe %TPR% ^| findstr /B /C:"single:"') do set V=%%t
for /f "tokens=9" %%s in ('bench_bk.exe %TPR% ^| findstr /C:"smem: dynamic"') do set S=%%s
if "%V%"=="" (echo POW_BK=%K%   no fit) else (echo POW_BK=%K%   !V! TH/s  dyn+static=!S!)
exit /b 0