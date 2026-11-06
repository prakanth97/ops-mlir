#!/usr/bin/env python3
"""ClangIR spike, P5.1: emit CIR for the translation units of the OPS applications and measure what a CIR frontend would have to raise.

For every application tree built by apps/ (<build>/<app>_tmp, with jit_kernels.json) the script

  1. copies the sources to a scratch directory, replacing the legacy `ops_seq.h` by `ops_seq_v2.h` (the legacy header, with
     its per-loop global arrays, is what clean-TU frontends choke on),
  2. runs `clang -fclangir -emit-cir` on every source of the kernels (the translation units that define them),
  3. counts, in the CIR of the kernel functions, the operations by name, the unsupported ("NYI") messages, and the calls
     (accessors, libm, helpers).

Output: JSON lines on stdout (one per TU) and a summary table at the end. Run on the cluster with the CIR build:

    scripts/cir/corpus.py --clang ~/ops-jit-cl/build/llvm-cir/bin/clang --ops-include <install>/include <build>/OPS-jit-apps ...
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def kernel_function_text(cir, kernel):
    """The text of the cir.func of `kernel` (a C++ function: the mangled name contains the name with its length).
    The function ends at the first line that closes it with the indentation of its first line."""
    lines = cir.splitlines()
    pattern = re.compile(r"cir\.func\s+[^@\n]*@(_Z[A-Za-z0-9_]*?" + str(len(kernel)) + re.escape(kernel) + r"[A-Za-z0-9_]*)")
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if not m or not line.rstrip().endswith("{"):
            continue
        indent = len(line) - len(line.lstrip())
        for j in range(i + 1, len(lines)):
            if lines[j].startswith(" " * indent + "}") and len(lines[j]) - len(lines[j].lstrip()) == indent:
                return m.group(1), "\n".join(lines[i + 1:j])
        return m.group(1), "\n".join(lines[i + 1:])
    return None, None


OP = re.compile(r"^\s*(?:%[\w#:,\s]+=\s*)?(cir\.[\w.]+)", re.M)
CALL = re.compile(r"cir\.call\s+@([\w$.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clang", required=True)
    ap.add_argument("--ops-include", required=True)
    ap.add_argument("--extra", action="append", default=[], help="extra clang arguments")
    ap.add_argument("trees", nargs="+", help="application build trees (directories holding <app>_tmp)")
    args = ap.parse_args()

    totals = collections.Counter()
    ops_all = collections.Counter()
    calls_all = collections.Counter()
    failures = collections.Counter()
    per_kernel = {}
    for tree in args.trees:
        for entry in sorted(os.listdir(tree)):
            tmp = os.path.join(tree, entry)
            manifest = os.path.join(tmp, "jit_kernels.json")
            if not entry.endswith("_tmp") or not os.path.exists(manifest):
                continue
            with open(manifest) as fh:
                kernels = json.load(fh)["kernels"]
            by_source = collections.defaultdict(list)
            for k in kernels:
                by_source[k["source"]].append(k["kernel"])
            work = tempfile.mkdtemp(prefix="cir_")
            try:
                for f in os.listdir(tmp):
                    p = os.path.join(tmp, f)
                    if os.path.isfile(p) and f.endswith((".cpp", ".h", ".in", ".inc")):
                        text = open(p, errors="replace").read()
                        text = re.sub(r'#\s*include\s*[<"]ops_seq\.h[>"]', '#include "ops_seq_v2.h"', text)
                        open(os.path.join(work, f), "w").write(text)
                for source, names in sorted(by_source.items()):
                    src = os.path.join(work, os.path.basename(source))
                    out = os.path.join(work, "out.cir")
                    cmd = [args.clang, "-fclangir", "-emit-cir", "-std=c++17", "-DOPS_ACC_IGNORE", "-I" + args.ops_include, "-I" + work] + args.extra + [src, "-o", out]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    rec = {"app": entry, "source": os.path.basename(source), "kernels": len(names), "ok": r.returncode == 0}
                    totals["tus"] += 1
                    totals["kernels"] += len(names)
                    if r.returncode != 0 or not os.path.exists(out):
                        rec["ok"] = False
                        lines = [l for l in (r.stderr or "").splitlines() if "error" in l or "NYI" in l or "not implemented" in l]
                        rec["reason"] = (lines[0] if lines else ((r.stderr or "no output").strip().splitlines() or [""])[-1])[:200]
                        failures[re.sub(r"^.*?(error|NYI)", r"\1", rec["reason"])[:100]] += 1
                        print(json.dumps(rec))
                        continue
                    cir = open(out).read()
                    totals["tus_ok"] += 1
                    found = 0
                    for name in names:
                        sym, body = kernel_function_text(cir, name)
                        if body is None:
                            continue
                        found += 1
                        ops = collections.Counter(OP.findall(body))
                        calls = collections.Counter(CALL.findall(body))
                        per_kernel[(entry, name)] = (sum(ops.values()), dict(ops))
                        ops_all.update(ops)
                        calls_all.update(calls)
                    rec["kernels_found"] = found
                    totals["kernels_found"] += found
                    totals["nyi_messages"] += len(re.findall(r"NYI|not yet implemented|unsupported", r.stderr or ""))
                    print(json.dumps(rec))
            finally:
                shutil.rmtree(work, ignore_errors=True)

    print("== summary")
    print(json.dumps(dict(totals)))
    print("== failures")
    for reason, n in failures.most_common():
        print(n, reason)
    print("== operations in the kernel functions")
    for op, n in ops_all.most_common():
        print(n, op)
    print("== calls in the kernel functions (top 40)")
    for name, n in calls_all.most_common(40):
        print(n, name)
    if per_kernel:
        sizes = sorted(v[0] for v in per_kernel.values())
        print("== operations per kernel: min %d median %d max %d over %d kernels" % (sizes[0], sizes[len(sizes) // 2], sizes[-1], len(sizes)))


if __name__ == "__main__":
    sys.exit(main())
