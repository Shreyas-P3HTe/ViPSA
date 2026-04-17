#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vipsa-mplconfig}"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"
exec python3 scripts/viewfinder_stress_sim.py "$@"
