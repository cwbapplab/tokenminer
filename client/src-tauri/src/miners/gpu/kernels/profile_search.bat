@echo off
REM Profile tokenminer_search_grid with Nsight Compute. Needs an elevated (Administrator) prompt,
REM because Windows refuses GPU performance counters without it (ERR_NVGPUCTRPERM).
REM
REM   Right-click "Command Prompt" (or PowerShell) -> "Run as administrator", then:
REM
REM     cd C:\Users\Erick\Documents\dev\tokenminer\client\src-tauri\src\miners\gpu\kernels
REM     profile_search.bat
REM
REM It writes ncu_report.txt and prints the sections that matter for this kernel: the stall reasons,
REM the instruction mix, and the achieved occupancy.

setlocal
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\ncu.bat"

rem Rebuild with line info so the report can attribute stalls to source lines.
%NVCC% -O3 -arch=sm_120a -diag-suppress 550 -diag-suppress 177 -Xcompiler /wd4477 ^
  -ccbin "%CCBIN%" -lineinfo -o bench_real.exe bench_real.cu
if errorlevel 1 (
  echo build failed
  exit /b 1
)

echo profiling...
%NCU% --target-processes all ^
  --kernel-name tokenminer_search_grid ^
  --launch-count 1 ^
  --set full ^
  --log-file ncu_report.txt ^
  bench_real.exe 8192

echo.
echo === key sections ===
%NCU% --import ncu_report.txt --page details 2>nul

echo done. ncu_report.txt is in this directory.
endlocal