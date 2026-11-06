#!/usr/bin/env bash
# Create the OPS translator's Python venv on the LOCAL machine, where the system Python has no
# ensurepip/pip (python3-venv is not installed and there is no sudo).
#
# OPS's CMake normally runs ops_translator/setup_venv_cmake.sh at configure time, which needs
# `python3 -m venv` with pip and network access. Here we instead:
#   1. create a venv without pip, and
#   2. copy the site-packages of the venv that the cluster build created from the same
#      requirements.txt (same Python 3.12 minor version; the packages are pure Python plus the
#      libclang 16.0.6 wheel).
# OPS's CMake then sees $OPS_INSTALL/translator/ops_translator/ops_venv and skips its own setup.
#
# Usage: scripts/setup_translator_local.sh        (needs `ssh cl`, and the cluster OPS install)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

VENV=$OPS_INSTALL/translator/ops_translator/ops_venv
REMOTE_VENV=ops-jit-cl/install/ops/translator/ops_translator/ops_venv
pyver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')

if [ -f "$VENV/.opsjit-populated" ]; then echo "translator venv already set up: $VENV"; exit 0; fi

remote_site=$(ssh cl "ls -d $REMOTE_VENV/lib/python*/site-packages" | head -1)
case "$remote_site" in
  *python"$pyver"/*) ;;
  *) echo "cluster venv uses a different Python than local $pyver ($remote_site)" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$VENV")"
python3 -m venv --without-pip "$VENV"
rsync -a --exclude '__pycache__' --exclude 'pip*' --exclude 'setuptools*' --exclude '_distutils_hack*' \
  "cl:$remote_site/" "$VENV/lib/python$pyver/site-packages/"
touch "$VENV/.opsjit-populated"
"$VENV/bin/python3" -c "import clang.cindex, jinja2; print('translator venv ok: libclang', clang.cindex.__file__)"
