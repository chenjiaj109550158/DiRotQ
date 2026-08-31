#!/usr/bin/env bash
# SDXL-base full run: chain-1 (SVDQuant side) then chain-2 (ours + official).
set -uo pipefail
R=/home/dev/DiRotQ/absorb_basis
if bash "$R/sdxl/run_sdxl30_chain1.sh"; then
  bash "$R/sdxl/run_sdxl30_chain2.sh"
  echo "SDXLB_ALL_DONE"
else
  echo "SDXLB_ALL_ABORTED chain1"
fi
