set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OPSTransforms)
add_public_tablegen_target(OPSMLIRTransformsIncGen)
