#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# CMF caches are generated locally rather than shipped as release artifacts.
"$PYTHON" "$ROOT/legacy/cmf_comparison/run_sphere_cmf.py"
"$PYTHON" "$ROOT/legacy/cmf_comparison/run_hyperbolic_cmf.py"
"$PYTHON" "$ROOT/legacy/cmf_comparison/run_se3_cmf.py"

"$PYTHON" "$ROOT/sphere/build_sphere_unified_metrics.py"
"$PYTHON" "$ROOT/hyperbolic/build_hyperbolic_unified_metrics.py"
"$PYTHON" "$ROOT/se3/build_se3_unified_metrics.py"
