# lit configuration of the ops-mlir test suite (loaded through the generated lit.site.cfg.py).
import os

import lit.formats

config.name = "OPS-MLIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".test", ".cpp", ".h"]
config.excludes = ["Inputs"]

# Tools: FileCheck/not/count/mlir-opt/mlir-translate from the pinned LLVM install, and our own tools
# (ops-opt, ops-kernel-import, ...) from the build tree.
tool_dirs = [config.opsjit_tools_dir, config.llvm_tools_dir]
config.environment["PATH"] = os.pathsep.join(tool_dirs + [config.environment.get("PATH", "")])

# Do not leak the user's environment into the tests (OPS_JIT_* options in particular).
for key in list(config.environment):
    if key.startswith("OPS_"):
        del config.environment[key]

config.substitutions.append(("%opsjit_src", config.opsjit_source_dir))
config.substitutions.append(("%opsjit_bin", config.opsjit_binary_dir))

# Runtime libraries of the MLIR execution tests (mlir-runner): printing and C helpers.
config.substitutions.append(("%mlir_runner_utils", config.llvm_lib_dir + "/libmlir_runner_utils.so"))
config.substitutions.append(("%mlir_c_runner_utils", config.llvm_lib_dir + "/libmlir_c_runner_utils.so"))

import platform

if platform.machine() in ("x86_64", "AMD64"):
    config.available_features.add("x86")
def _find_libomp():
    for d in (config.llvm_lib_dir, os.path.join(config.llvm_lib_dir, "x86_64-unknown-linux-gnu")):
        path = os.path.join(d, "libomp.so")
        if os.path.exists(path):
            return path
    return "libomp.so"


config.substitutions.append(("%mlir_omp", _find_libomp()))

# Kernel importer tests need the headers of OPS (ops_lib_core.h defines ACC and the MIN/MAX/SIGN macros).
if os.path.exists(os.path.join(config.ops_include_dir, "ops_lib_core.h")):
    config.available_features.add("ops-headers")
config.substitutions.append(("%ops_include", config.ops_include_dir))
