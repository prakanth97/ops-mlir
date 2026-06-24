//===- OPSOps.td - OPS Dialect Operation Definitions ----*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//

#ifndef OPS_OPS
#define OPS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// OPS Dialect Definition
//===----------------------------------------------------------------------===//

def OPS_Dialect : Dialect {
  let name = "ops";
  let summary = "OPS DSL dialect for structured mesh computations";
  let description = [{
    The OPS dialect represents structured mesh parallel loops as captured from
    the OPS API. It models the loop structure, data dependencies, and stencil
    operations. The dialect is designed to be lowered to control flow and
    then to concrete execution targets.
  }];
  let cppNamespace = "::ops_mlir::ops";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class OPS_Op<string mnemonic, list<Trait> traits = []>
    : Op<OPS_Dialect, mnemonic, traits> {
  let cppNamespace = "::ops_mlir::ops";
}

//===----------------------------------------------------------------------===//
// OPS Attributes
//===----------------------------------------------------------------------===//

def OPS_DatAttr : AttrDef<OPS_Dialect, "Dat"> {
  let mnemonic = "dat";
  let summary = "Storage descriptor for an OPS dataset";
  let parameters = (ins
    "int32_t":$index,
    "int64_t":$block, // TODO: this is pointer need change
    "int32_t":$dim,
    "int32_t":$type_size,
    "int32_t":$elem_size,

    "mlir::DenseI64ArrayAttr":$size,
    "mlir::DenseI64ArrayAttr":$base,
    "mlir::DenseI64ArrayAttr":$d_m,
    "mlir::DenseI64ArrayAttr":$d_p,
    "mlir::DenseI64ArrayAttr":$stride,

    "mlir::StringAttr":$name,
    "mlir::StringAttr":$type,

    "int64_t":$data, // TODO: this is pointer need change
    "int64_t":$data_d // TODO: this is pointer need change
  );
  let assemblyFormat = "`<` $index `,` $block `,` $dim `,` $type_size `,`"
                       " $elem_size `,` $size `,` $base `,` $d_m `,` $d_p `,`"
                       " $stride `,` $name `,` $type `,` $data `,` $data_d `>`";
}

def OPS_StencilAttr : AttrDef<OPS_Dialect, "Stencil"> {
  let mnemonic = "stencil";
  let summary = "Stencil descriptor of kernel";
  let parameters = (ins
    "int32_t":$index,
    "int32_t":$dims,
    "int32_t":$points,
    "mlir::StringAttr":$name,
    "int64_t":$stencil,
    "int64_t":$stride,
    "int64_t":$mgrid_stride,
    "int32_t":$type
  );
  let assemblyFormat = "`<` $index `,` $dims `,` $points `,` $name `,`"
                       " $stencil `,` $stride `,` $mgrid_stride `,` $type `>`";
}


// Structured descriptor for a single ops.par_loop argument, mirroring
// ops_mlir::ArgDesc (runtime/Core.h). Replaces the previous positional
// ArrayAttr encoding produced by IRBuilder::buildArgAttr.
def OPS_ArgAttr : AttrDef<OPS_Dialect, "Arg"> {
  let mnemonic = "arg";
  let summary = "OPS dataset argument descriptor";
  let description = [{
    Represents an OPS dataset argument with its kind, access pattern,
    dataset shape, and stencil offsets.
  }];
  let parameters = (ins
    "ops_mlir::ops::DatAttr":$dat,
    "ops_mlir::ops::StencilAttr":$stencil,
    "int32_t":$dim,
    "int32_t":$elem_size,
    "int64_t":$data,
    "int64_t":$data_d,
    "int32_t":$acc,
    "int32_t":$argtype,
    "int32_t":$opt
  );
  let assemblyFormat = "`<` $dat `,` $stencil `,` $dim `,` $elem_size `,`"
                       " $data `,` $data_d `,` $acc `,` $argtype `,` $opt `>`";
}


//===----------------------------------------------------------------------===//
// OPS Operations
//===----------------------------------------------------------------------===//

def OPS_ParLoopOp : OPS_Op<"par_loop", []> {
  let summary = "OPS parallel loop over a structured mesh block";
  let description = [{
    Represents a single OPS parallel loop capture. This operation encodes:
    - The kernel function pointer and metadata
    - The iteration range and dimensions
    - The arguments (datasets, stencils, reductions, globals, indices)
    - Access patterns for each argument

    The operation carries all loop metadata as attributes and is designed
    for lowering to control flow and memory operations during compilation.
  }];

  let arguments = (ins
    I64Attr:$kernel_ptr,
    StrAttr:$kernel_name,
    I32Attr:$dims,
    DenseI64ArrayAttr:$range,
    TypedArrayAttrBase<OPS_ArgAttr, "array of OPS dataset argument descriptors">:$args,
    OptionalAttr<I32Attr>:$block_dims
  );

  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

#endif // OPS_OPS
