# Initialises the miner git submodules that the desktop client links natively.
# Non-recursive on purpose: Pearl's CUTLASS submodule lives under its Python/GPU
# stack (`miner/pearl-gemm`), which we do not build.
$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')

Write-Host 'Initialising miner submodules...'
git -C $root submodule update --init client/miners/quantus-miner client/miners/pearl

Write-Host 'Miner submodules ready.'
