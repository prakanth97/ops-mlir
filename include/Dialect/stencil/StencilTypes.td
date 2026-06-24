//===- StencilTypes.td - Stencil Dialect Type Definitions ------*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//

#ifndef STENCIL_TYPES
#define STENCIL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "StencilBase.td"
include "StencilAttr.td"

// Common parameters shared by field/temp: per-dimension bounds + element
// type, printed xDSL-style (see xdsl_impl/laplace_stencil.mlir), e.g.
// !stencil.field<[-1,4097]x[-1,4097]xf64> -- per-dim [lb,ub] pairs joined
// by `x`, followed directly by the element type. That repeated-group-by-
// rank shape isn't expressible as a plain declarative assemblyFormat, so
// each of Field/TempType gets a hand-written parse()/print() (see
// StencilDialect.cpp's parseBoundedType/printBoundedType helpers).
class Stencil_BoundedType<string name> : TypeDef<Stencil_Dialect, name> {
  let parameters = (ins Stencil_BoundsAttr:$bounds, "mlir::Type":$element_type);
  let hasCustomAssemblyFormat = 1;
}

def Stencil_FieldType : Stencil_BoundedType<"Field"> {
  let mnemonic = "field";
  let summary = "A stencil field: storage for a structured mesh dataset";
}

def Stencil_TempType : Stencil_BoundedType<"Temp"> {
  let mnemonic = "temp";
  let summary = "A stencil temp: an intermediate, register-like view of a field";
}

#endif // STENCIL_TYPES
