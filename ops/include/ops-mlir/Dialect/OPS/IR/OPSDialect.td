//===- OPSDialect.td - the `ops` dialect --------------------*- tablegen -*-===//
//
// The `ops` dialect describes OPS structured-mesh loops as IR that can be analysed, fused and lowered,
// instead of a copy of the OPS data structures.
//
//   !ops.dat<T, rank, ncomp>     a dat (field), as an argument of a loop chain
//   !ops.acc<T, rank, ncomp>     accessor of a dat inside a loop body or kernel
//   !ops.gbl<T, n>              read-only global array (kernel parameter)
//   !ops.red<T>                 reduction target (kernel parameter)
//
//   ops.kernel   @k(...)         a user kernel (loop independent: accessors are only typed)
//   ops.chain    @c(...)         a chain of loops, the unit the executor lifts from OPS's queue
//   ops.par_loop                 one parallel loop over a range, with the kernel body as its region
//
// See docs/design.md for the full description.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_DIALECT_OPS_OPSDIALECT
#define OPS_MLIR_DIALECT_OPS_OPSDIALECT

include "mlir/IR/OpBase.td"

def OPS_Dialect : Dialect {
  let name = "ops";
  let summary = "OPS structured-mesh parallel loops";
  let description = [{
    Buffer-semantics representation of OPS `ops_par_loop`s: dats are mutable, loops have side effects on the
    dats they write, and the kernel body is a region of accessor operations.
  }];
  let cppNamespace = "::ops_mlir::ops";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  // Registered in OPSTypes.cpp and OPSAttrs.cpp, which define the storage classes.
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // OPS_MLIR_DIALECT_OPS_OPSDIALECT
