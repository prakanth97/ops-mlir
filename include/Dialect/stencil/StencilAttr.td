//===- StencilAttr.td - Stencil Dialect Attribute Definitions ----*- tablegen -*-===//  
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

// stencil.index
def Stencil_IndexAttr : AttrDef<Stencil_Dialect, "Index"> {
  let mnemonic = "index";
  let parameters = (ins ArrayAttr:$array);  // ArrayAttr<IntAttr>, rank 1–3
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  // TODO: Do we need to add extra declarations to match xdsl impl
}

def Stencil_BoundsAttr : AttrDef<Stencil_Dialect, "StencilBounds"> {
  let mnemonic = "bounds";
  let summary = "Stencil bounds attribute, lower and upper bounds for each dimension";
  let parameters = (ins IndexAttr:$lb, IndexAttr:$ub);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  // TODO: Do we need to add extra declarations to match xdsl impl
}

def Stencil_StencilType : TypeDef<Stencil_Dialect, "Type"> {
  let mnemonic = "type";
  let parameters = (ins
    "Stencil_BoundsAttr | IntAttr":$bounds,
    Type:$element_type
  )
  let hasCustomAssemblyFormat = 1;
}

def Stencil_FieldType : Stencil_StencilType<Stencil_Dialect, "Field"> {
  let mnemonic = "field";
}

def Stencil_TempType : Stencil_StencilType<Stencil_Dialect, "Field"> {
  let mnemonic = "temp";
}

def Stencil_ResultType : TypeDef<Stencil_Dialect, "Result"> {
  let mnemonic = "result";
  let parameters = (ins Type:$elem);
}
