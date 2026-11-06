#!/usr/bin/env bash
# core/ (namespace opdsl) must stay independent of OPS so that it can move to its own repository and be
# shared with op2-mlir. Fails if a file under core/ includes an OPS header or names the OPS namespace.
#
#   scripts/check_core_independence.sh [core dir]
set -euo pipefail
core=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/core}

pattern='#[[:space:]]*include[[:space:]]*[<"](ops_[a-z_0-9]*\.h|ops-mlir/|Dialect/OPS|OPS[A-Za-z]*\.h)|ops_mlir::|namespace[[:space:]]+ops_mlir'
if hits=$(grep -rnE --include='*.h' --include='*.cpp' --include='*.td' --include='*.cmake' --include='CMakeLists.txt' \
          "$pattern" "$core" 2>/dev/null); then
  echo "core/ must not depend on OPS:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "core/ is independent of OPS"
