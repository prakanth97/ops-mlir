# Command-line drivers: ops-opt, ops-kernel-import (P2.1), ops-jit-replay (P3.8).
# Register them with add_opsjit_tool() so that the lit tests depend on them.
add_subdirectory(ops-opt)
add_subdirectory(ops-kernel-import)
add_subdirectory(ops-jit-replay)
