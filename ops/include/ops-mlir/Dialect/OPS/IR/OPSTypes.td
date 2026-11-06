//===- OPSTypes.td - types of the `ops` dialect --------------*- tablegen -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_OPSTYPES
#define OPS_MLIR_DIALECT_OPS_OPSTYPES

include "mlir/IR/AttrTypeBase.td"
include "ops-mlir/Dialect/OPS/IR/OPSDialect.td"

class OPS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<OPS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Element types of dats, accessors, globals and reductions: f16, f32, f64, i32, i64.

class OPS_FieldLikeType<string name, string typeMnemonic> : OPS_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::Type":$elementType, "int64_t":$rank, "int64_t":$ncomp);
  let assemblyFormat = "`<` $elementType `,` $rank `,` $ncomp `>`";
  let genVerifyDecl = 1;
}

def OPS_DatType : OPS_FieldLikeType<"Dat", "dat"> {
  let summary = "an OPS dat";
  let description = [{
    A dat on a block of rank `rank` (1 to 3), with `ncomp` components per point. Extents, halos and memory
    layout are run-time properties of the dat: they are not part of the type.
  }];
}

def OPS_AccType : OPS_FieldLikeType<"Acc", "acc"> {
  let summary = "accessor of a dat inside a loop body or kernel";
  let description = [{
    Read through `ops.read` and written through `ops.write` at constant offsets from the current point.
    The access mode (read, write, rw, inc, min, max) and the stencil belong to the loop, not to the type.
  }];
}

def OPS_GblType : OPS_Type<"Gbl", "gbl"> {
  let summary = "read-only global array (kernel parameter)";
  let parameters = (ins "::mlir::Type":$elementType, "int64_t":$size);
  let assemblyFormat = "`<` $elementType `,` $size `>`";
  let genVerifyDecl = 1;
}

def OPS_RedType : OPS_Type<"Red", "red"> {
  let summary = "reduction target (kernel parameter)";
  let description = [{
    A kernel contributes to a reduction with `ops.reduce`; the loop combines the contributions of all points
    with the reduction kind it declares.
  }];
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

#endif // OPS_MLIR_DIALECT_OPS_OPSTYPES
