//===- StencilAttr.td - Stencil Dialect Attribute Definitions -*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//

#ifndef STENCIL_ATTRS
#define STENCIL_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "StencilBase.td"

// stencil.index -- a rank-1..3 list of per-dimension offsets, e.g.
// #stencil.index<[1, 0]>. Backed by a plain DenseI64ArrayAttr rather than a
// hand-rolled bracket-list parser, to keep this dialect's first cut fully
// ODS-generated (no custom C++ parse/print).
def Stencil_IndexAttr : AttrDef<Stencil_Dialect, "Index"> {
  let mnemonic = "index";
  let parameters = (ins "mlir::DenseI64ArrayAttr":$array);
  let assemblyFormat = "`<` $array `>`";
  let genVerifyDecl = 1;
}

// stencil.bounds -- per-dimension [lb, ub) bounds of a stencil.apply's
// iteration domain or a stencil.store's target sub-range, e.g.
// #stencil.bounds<[-1, -1], [4097, 4097]>.
def Stencil_BoundsAttr : AttrDef<Stencil_Dialect, "StencilBounds"> {
  let mnemonic = "bounds";
  let summary = "Stencil bounds attribute, lower and upper bounds for each dimension";
  let parameters = (ins "mlir::DenseI64ArrayAttr":$lb, "mlir::DenseI64ArrayAttr":$ub);
  let assemblyFormat = "`<` $lb `,` $ub `>`";
  let genVerifyDecl = 1;
}

#endif // STENCIL_ATTRS
