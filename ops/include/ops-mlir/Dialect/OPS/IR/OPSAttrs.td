//===- OPSAttrs.td - attributes of the `ops` dialect ---------*- tablegen -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_OPSATTRS
#define OPS_MLIR_DIALECT_OPS_OPSATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "ops-mlir/Dialect/OPS/IR/OPSDialect.td"

// The numeric values of the access modes are the ones of OPS (ops_access in ops_lib_core.h).
def OPS_AccessMode : I32EnumAttr<"AccessMode", "OPS access mode", [
    I32EnumAttrCase<"Read", 0, "read">,
    I32EnumAttrCase<"Write", 1, "write">,
    I32EnumAttrCase<"RW", 2, "rw">,
    I32EnumAttrCase<"Inc", 3, "inc">,
    I32EnumAttrCase<"Min", 4, "min">,
    I32EnumAttrCase<"Max", 5, "max">]> {
  let cppNamespace = "::ops_mlir::ops";
  let genSpecializedAttr = 0;
}
def OPS_AccessModeAttr : EnumAttr<OPS_Dialect, OPS_AccessMode, "access"> {
  let assemblyFormat = "`<` $value `>`";
}

// How the contributions of all points of a loop are combined (OPS_INC, OPS_MIN, OPS_MAX on a global).
def OPS_ReductionKind : I32EnumAttr<"ReductionKind", "reduction kind", [
    I32EnumAttrCase<"Add", 0, "add">,
    I32EnumAttrCase<"Min", 1, "min">,
    I32EnumAttrCase<"Max", 2, "max">]> {
  let cppNamespace = "::ops_mlir::ops";
  let genSpecializedAttr = 0;
}
def OPS_ReductionKindAttr : EnumAttr<OPS_Dialect, OPS_ReductionKind, "red_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def OPS_StencilAttr : AttrDef<OPS_Dialect, "Stencil"> {
  let mnemonic = "stencil";
  let summary = "stencil of a dat argument";
  let description = [{
    The offsets a loop accesses a dat at, in OPS axis order (x first), and per dimension the stride with which
    the dat's index follows the loop's: 1 normally, 0 for a dimension the dat does not have (a low-dimensional
    dat on a higher-dimensional block). Textual form:

    ```mlir
    #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>
    #ops.stencil<[[0, 0]], strides = [1, 0]>
    ```
  }];
  // `points` holds numPoints * rank offsets, point after point. `strides` always has `rank` entries.
  let parameters = (ins "int64_t":$rank, ArrayRefParameter<"int64_t">:$points, ArrayRefParameter<"int64_t">:$strides);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Stencil with unit strides.
    static StencilAttr get(::mlir::MLIRContext *ctx, int64_t rank, ::llvm::ArrayRef<int64_t> points);
    int64_t getNumPoints() const { return getRank() ? getPoints().size() / getRank() : 0; }
    /// Offsets of point `i`.
    ::llvm::ArrayRef<int64_t> getPoint(int64_t i) const {
      return getPoints().slice(i * getRank(), getRank());
    }
    /// True if (`offsets`) is one of the stencil's points.
    bool contains(::llvm::ArrayRef<int64_t> offsets) const;
    /// True if the only point is the zero offset and all strides are 1.
    bool isPointLocal() const;
    /// Smallest and largest offset along dimension `d`.
    std::pair<int64_t, int64_t> getExtent(int64_t d) const;
  }];
}

def OPS_AccessModeArrayAttr : TypedArrayAttrBase<OPS_AccessModeAttr, "array of access modes">;
def OPS_StencilArrayAttr : TypedArrayAttrBase<OPS_StencilAttr, "array of stencils">;
def OPS_ReductionKindArrayAttr : TypedArrayAttrBase<OPS_ReductionKindAttr, "array of reduction kinds">;

#endif // OPS_MLIR_DIALECT_OPS_OPSATTRS
