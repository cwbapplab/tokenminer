# Initialises the miner git submodules that the desktop client links natively.
# Non-recursive on purpose: Pearl's CUTLASS submodule lives under its Python/GPU
# stack (`miner/pearl-gemm`), which we do not build, and llmjob's own vendored
# dependencies are not read by the CUDA core we compile.
$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')

Write-Host 'Initialising miner submodules...'
# vendor/llmjob supplies the Pearl CUDA core that build.rs compiles from source;
# without it the build fails with a message naming this command.
git -C $root submodule update --init client/miners/quantus-miner client/miners/pearl vendor/llmjob

Write-Host 'Miner submodules ready.'
