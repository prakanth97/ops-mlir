#!/usr/bin/env bash
# Runs the ctest suite of an OPS build tree (BUILD_OPS_APPS=ON) and records the result.
#
#   scripts/run_ops_ctest.sh <build-tree> <cpu|gpu|all> <tag> [extra ctest args]
#
# cpu: every test whose name does not contain "cuda"; gpu: those that do (run it on a GPU node).
# The test list with pass/fail goes to $OPSJIT_WORK/results/<tag>.txt, the full log to <tag>.log.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

tree=$1
mode=${2:-cpu}
tag=${3:-ctest}
shift 3 || true

results=$OPSJIT_WORK/results
mkdir -p "$results"
case "$mode" in
  cpu) filter=( -E cuda ) ;;
  gpu) filter=( -R cuda ) ;;
  all) filter=() ;;
  *) echo "mode must be cpu, gpu or all" >&2; exit 2 ;;
esac

echo "== $(date -Is) ctest in $tree ($mode) on $(hostname)"
ctest --test-dir "$tree" "${filter[@]}" -j "${CTEST_JOBS:-4}" --timeout "${CTEST_TIMEOUT:-600}" \
      --output-on-failure "$@" 2>&1 | tee "$results/$tag.log" | tail -40
status=${PIPESTATUS[0]}
# The summary: one line per test (name, Passed/Failed, seconds).
grep -E "^ *[0-9]+/[0-9]+ Test +#[0-9]+:" "$results/$tag.log" | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: //' >"$results/$tag.txt"
echo "== $(date -Is) done, status $status; $(grep -c Passed "$results/$tag.txt") passed, $(grep -c -E 'Failed|Timeout|Exception' "$results/$tag.txt") failed; $results/$tag.txt"
exit "$status"
