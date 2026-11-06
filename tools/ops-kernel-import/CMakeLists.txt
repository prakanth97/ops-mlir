add_opsjit_tool(ops-kernel-import ops-kernel-import.cpp
  LINK OPSMLIRFrontend OPSMLIRDialect MLIRBytecodeWriter MLIRArithDialect MLIRMathDialect MLIRSCFDialect LLVMSupport)
llvm_update_compile_flags(ops-kernel-import)
