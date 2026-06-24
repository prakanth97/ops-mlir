//===- StencilBase.td - Stencil Dialect & base op class --------*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//
//
// Split out from StencilOps.td so that StencilAttr.td/StencilTypes.td (which
// reference Stencil_Dialect) can be included before the op definitions that
// use those attrs/types -- avoids forward-reference ordering issues.
//===----------------------------------------------------------------------===//

#ifndef STENCIL_BASE
#define STENCIL_BASE

include "mlir/IR/OpBase.td"

def Stencil_Dialect : Dialect {
  let name = "stencil";
  let summary = "Stencil DSL dialect for structured mesh computations";
  let description = [{
    The Stencil dialect represents structured mesh computations as captured from
    stencil DSLs. It models the loop structure, data dependencies, and stencil
    operations. The dialect is designed to be lowered to control flow and
    then to concrete execution targets.
  }];
  let cppNamespace = "::mlir::stencil";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Stencil_Op<string mnemonic, list<Trait> traits = []>
  : Op<Stencil_Dialect, mnemonic, traits>;

#endif // STENCIL_BASE
