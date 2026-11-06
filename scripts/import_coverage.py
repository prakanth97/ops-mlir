#!/usr/bin/env python3
"""Import coverage of frontend A over the OPS applications.

For every application directory under OPS/apps/c that has loops (a .cpp file calling ops_par_loop) the script

  1. copies the sources to a scratch directory and runs `ops-translator --jit` on them (this gives jit_kernels.json:
     the kernels with their parameter kinds, as the build sees them),
  2. runs ops-kernel-import on every kernel in the translation unit of the application, with the flags the apps build uses,
  3. collects status and reason from the manifests,

and writes a report (default docs/coverage.md, not kept in the repository): totals, the reasons ranked, and the failures per
application.

    scripts/import_coverage.py [--jobs 4] [--out docs/coverage.md] [--apps name,name]
"""
import argparse
import collections
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HOME = os.path.expanduser("~")
WORK = os.environ.get("OPSJIT_WORK", os.path.join(HOME, "ops-mlir-work"))


def app_dirs(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("handcoded", "original", "legacy", ".git")]
        for f in filenames:
            if f.endswith(".cpp") and not f.endswith("_ops.cpp"):
                try:
                    with open(os.path.join(dirpath, f), errors="replace") as fh:
                        if "ops_par_loop" in fh.read():
                            found.append(dirpath)
                            break
                except OSError:
                    pass
    return sorted(found)


def run_app(args, app):
    name = os.path.relpath(app, args.ops_apps)
    result = {"app": name, "kernels": [], "error": None}
    tmp = tempfile.mkdtemp(prefix="cov_", dir=args.scratch)
    try:
        sources = []
        for f in sorted(os.listdir(app)):
            p = os.path.join(app, f)
            if f.endswith((".cpp", ".h", ".in", ".inc")) and os.path.isfile(p):
                shutil.copy(p, tmp)
            if f.endswith(".cpp") and not f.endswith("_ops.cpp"):
                with open(p, errors="replace") as fh:
                    if "ops_par_loop" in fh.read():
                        sources.append(f)
        cmd = [args.python, args.translator, "-DOPS_ACC_IGNORE", "-I", args.ops_include, "--jit", "--file_paths"] + sources
        try:
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result["error"] = "translator timed out"
            return result
        manifest = os.path.join(tmp, "jit_kernels.json")
        if r.returncode != 0 or not os.path.exists(manifest):
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            result["error"] = "translator failed: " + tail[0][:200]
            return result
        with open(manifest) as fh:
            kernels = json.load(fh)["kernels"]
        for k in kernels:
            out = os.path.join(tmp, "k_" + k["kernel"])
            cmd = [args.importer, "--kernel", k["kernel"], "--source", k["source"], "--ndim", str(k["ndim"]),
                   "--manifest", out + ".json", "-o", out + ".mlirbc"]
            if k.get("kinds"):
                cmd += ["--kinds", ",".join(k["kinds"])]
            cmd += ["--", "-DOPS_ACC_IGNORE", "-I" + args.ops_include, "-I" + tmp]
            if args.extra:
                cmd += args.extra.split()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                result["kernels"].append({"kernel": k["kernel"], "status": "failed", "reason": "importer timed out"})
                continue
            status, reason = "failed", "no manifest"
            if os.path.exists(out + ".json"):
                with open(out + ".json") as fh:
                    m = json.load(fh)
                status = m.get("status", "failed")
                reason = m.get("reason", "") or ""
                if status == "ok":
                    dropped = m.get("traits", {}).get("dropped_printf", 0)
                    if dropped:
                        reason = "printf dropped"
            result["kernels"].append({"kernel": k["kernel"], "status": status, "reason": reason})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def normalise(reason):
    reason = re.sub(r"'[^']*'", "'..'", reason)
    reason = re.sub(r"\d+", "N", reason)
    return reason.strip()[:140] or "(none)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops-src", default=os.path.join(WORK, "OPS"))
    ap.add_argument("--ops-install", default=os.path.join(WORK, "install", "ops"))
    ap.add_argument("--importer", default=os.path.join(WORK, "build", "ops-mlir", "bin", "ops-kernel-import"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "coverage.md"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--apps", default="", help="comma-separated application directory names (relative to apps/c)")
    ap.add_argument("--extra", default="", help="more clang flags for the importer")
    args = ap.parse_args()

    args.ops_apps = os.path.join(args.ops_src, "apps", "c")
    args.ops_include = os.path.join(args.ops_install, "include")
    tr = os.path.join(args.ops_install, "translator", "ops_translator")
    args.translator = os.path.join(tr, "ops-translator")
    args.python = os.path.join(tr, "ops_venv", "bin", "python3")
    args.scratch = tempfile.mkdtemp(prefix="opsjit_cov_")

    apps = app_dirs(args.ops_apps)
    if args.apps:
        want = set(args.apps.split(","))
        apps = [a for a in apps if os.path.relpath(a, args.ops_apps) in want]
    print(f"{len(apps)} applications", file=sys.stderr)

    results = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_app, args, a): a for a in apps}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            ok = sum(1 for k in r["kernels"] if k["status"] == "ok")
            print(f"  {r['app']}: {ok}/{len(r['kernels'])} imported" + (f"  [{r['error']}]" if r["error"] else ""),
                  file=sys.stderr)
            results.append(r)
    shutil.rmtree(args.scratch, ignore_errors=True)
    results.sort(key=lambda r: r["app"])

    total = sum(len(r["kernels"]) for r in results)
    ok = sum(1 for r in results for k in r["kernels"] if k["status"] == "ok")
    reasons = collections.Counter(normalise(k["reason"]) for r in results for k in r["kernels"] if k["status"] != "ok")
    lines = ["# Import coverage of frontend A", "",
             "Generated by `scripts/import_coverage.py` (OPS `%s`, importer `A/1.0`). Every kernel that the translator finds in an "
             "application is imported in the translation unit of the application." % subprocess.run(
                 ["git", "-C", args.ops_src, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip(),
             "", f"**{ok} of {total} kernels import** ({100.0 * ok / max(total, 1):.1f}%) in {len(results)} applications; "
             f"{sum(1 for r in results if r['error'])} applications could not be translated.", "",
             "## Reasons for failures", "", "| count | reason |", "|---:|---|"]
    for reason, n in reasons.most_common():
        lines.append(f"| {n} | {reason} |")
    lines += ["", "## Applications", "", "| application | imported | translator |", "|---|---:|---|"]
    for r in results:
        n_ok = sum(1 for k in r["kernels"] if k["status"] == "ok")
        lines.append(f"| {r['app']} | {n_ok}/{len(r['kernels'])} | {r['error'] or 'ok'} |")
    lines += ["", "## Failed kernels", ""]
    for r in results:
        failed = [k for k in r["kernels"] if k["status"] != "ok"]
        if failed:
            lines.append(f"### {r['app']}")
            lines.append("")
            for k in failed:
                lines.append(f"- `{k['kernel']}`: {k['reason'][:200]}")
            lines.append("")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{ok}/{total} kernels import; wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
