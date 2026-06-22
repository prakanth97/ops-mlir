// --- StencilOps.td - Stencil Dialect Operation Definitions ----*- tablegen -*----
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===// 

#ifndef STENCIL_OPS
#define STENCIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "StencilAttr.td"

//===----------------------------------------------------------------------===//
// Stencil Dialect Definition
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Base Op class
//===----------------------------------------------------------------------===//

class Stencil_Op<str mnemonic, list<Trait> traits = []> 
  : Op<Stencil_Dialect, mnemonic, traits>;

// stencil.access — relative-offset read inside stencil.apply (Pure, HasAncestor<ApplyOp>)
def Stencil_AccessOp : Stencil_Op<"access", [
    Pure, 
    HasAncestor<Stencil_ApplyOp>,
    DeclareOpInterfaceMethods<HasShapeInferencePatterns>
  ]> {
  let summary = "Access a stencil temp at a static relative offset";
  let arguments = (ins
    Stencil_TempType:$temp,
    Stencil_IndexAttr:$offset,
    OptionalAttr<Stencil_IndexAttr>:$offset_mapping
  );
  let results = (outs AnyType:$res);  // element type of $temp
  let assemblyFormat = "$temp `at` $offset (`mapping` $offset_mapping^)? attr-dict-with-keyword"
                       " `:` type($temp) `->` type($res)";
  let hasVerifier = 1;

}

// stencil.alloc
def Stencil_AllocOp : Stencil_Op<"alloc", 
    [DeclareOpInterfaceMethods<MemoryEffectOpInterface>]> {
  let summary = "Allocate a new stencil field";
  let results = (outs Stencil_FieldType:$field);
  let assemblyFormat = "attr-dict `:` type($field)";
}

// stencil.cast — casts dynamic field to statically shaped field (NoSideEffect)
def Stencil_CastOp : Stencil_Op<"cast", [
    Pure,
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>,
    DeclareOpInterfaceMethods<HasCanonicalizationPatterns>]> {
  let arguments = (ins Stencil_FieldType:$field);
  let results  = (outs Stencil_FieldType:$result);
    let assemblyFormat =
    "$field attr-dict-with-keyword `:` type($field) `->` type($result)";
  let hasVerifier = 1;
}

// stencil.apply — core compute op; IsolatedFromAbove
def Stencil_ApplyOp : Stencil_Op<"apply", [
    IsolatedFromAbove, 
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>,
    DeclareOpInterfaceMethods<HasCanonicalizationPatterns>,
    DeclareOpInterfaceMethods<HasShapeInferenceInterface>]> {
  let summary = "Apply a stencil computation over a domain";
  let arguments = (ins
    Variadic<AnyType>:$args,          // stencil.temp or stencil.field inputs
    Variadic<Stencil_FieldType>:$dest // buffer-semantic destinations (optional),
    OptionalAttr<StencilBoundsAttr>:$bounds
  );
  let results = (outs Variadic<Stencil_TempType>:$res);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$args (`to` $dest)? (`(` $bounds `)`)? $region attr-dict-with-keyword"
                       " `:` type($args) (`to` type($dest))? `->` type($res)";
}

// stencil.load — reads a field into a temp (ReadEffect on $field)
def Stencil_LoadOp : Stencil_Op<"load", [
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>,
    DeclareOpInterfaceMethods<HasCanonicalizationPatterns>
]> {
  let summary = "Load a stencil.field(with know bounds) into a stencil.temp";
  let arguments = (ins Stencil_FieldType:$field);
  let results  = (outs Stencil_TempType:$res);
  let assemblyFormat =
    "$field attr-dict-with-keyword `:` type($field) `->` type($res)";
  let hasVerifier = 1;
}

// stencil.index — loop induction index for a given dimension (Pure, HasAncestor<ApplyOp>)
def Stencil_IndexOp : Stencil_Op<"index", [
    Pure, 
    HasAncestor<Stencil_ApplyOp>
  ]> {
  let summary = "Return the current iteration index shifted by a static offset";
  let arguments = (ins
    IndexAttr:$dim,
    Stencil_IndexAttr:$offset
  );
  let results = (outs Index:$idx);
  let assemblyFormat = "$dim $offset attr-dict-with-keyword";
}

// stencil.return — terminator of stencil.apply; matches result types (IsTerminator, Pure)
def Stencil_ReturnOp : Stencil_Op<"return", [
    IsTerminator, 
    Pure, 
    HasParent<Stencil_ApplyOp>
  ]> {
  let summary = "Terminate stencil.apply and write computed values as results";
  let arguments = (ins 
    Variadic<AnyType>:$arg,
    OptionalAttr<IntAttr>:$unroll  
  );
  let assemblyFormat = "$arg (`unroll` $unroll^)? attr-dict-with-keyword `:` type($arg)";
}

// stencil.reduce — accumulates values across stencil iterations
def Stencil_ReduceOp : Stencil_Op<"reduce"> {
  let arguments = (ins AnyType:$acc, AnyType:$init);
  let regions  = (region SizedRegion<1>:$body);   // single block, 2 args of same type
  // must end with stencil.yield
  let assemblyFormat = "$acc `init` $init $body attr-dict `:` type($acc)";
  let hasVerifier = 1;
}

// stencil.store — writes a temp to a field (WriteEffect on $field)
def Stencil_StoreOp : Stencil_Op<"store", [
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>,
    DeclareOpInterfaceMethods<HasShapeInferencePatterns>
]> {
  let summary = "Write a stencil.temp to a stencil.field over a sub-range";
  let arguments = (ins
    Stencil_TempType:$temp,
    Stencil_FieldType:$field,         // must have explicit (non-dynamic) bounds
    StencilBoundsAttr:$bounds
  );
  let assemblyFormat = "$temp `to` $field `(` $bounds `)` attr-dict-with-keyword"
                       " `:` type($temp) `to` type($field)";
}

// stencil.yield — terminator for stencil.reduce (IsTerminator)
def Stencil_YieldOp : Stencil_Op<"yield", [IsTerminator]> {
  let summary = "Yield a value from a stencil.reduce body";
  let arguments = (ins AnyType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}
