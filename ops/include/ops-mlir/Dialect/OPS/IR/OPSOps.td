//===- OPSOps.td - operations of the `ops` dialect -----------*- tablegen -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_OPSOPS
#define OPS_MLIR_DIALECT_OPS_OPSOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ops-mlir/Dialect/OPS/IR/OPSDialect.td"
include "ops-mlir/Dialect/OPS/IR/OPSTypes.td"
include "ops-mlir/Dialect/OPS/IR/OPSAttrs.td"

class OPS_Op<string mnemonic, list<Trait> traits = []> : Op<OPS_Dialect, mnemonic, traits>;

// Element types of values read from and written to accessors.
def OPS_ScalarType : AnyTypeOf<[F16, F32, F64, I32, I64]>;
// Operand type of a data-dependent offset.
def OPS_OffsetType : AnyTypeOf<[Index, I32, I64]>;

//===----------------------------------------------------------------------===//
// Operations of kernel bodies (inside ops.kernel and ops.par_loop regions)
//===----------------------------------------------------------------------===//

def OPS_ReadOp : OPS_Op<"read", [
    MemoryEffects<[MemRead]>,
    TypesMatchWith<"result type is the element type of the accessor", "acc", "result",
                   "::llvm::cast<::ops_mlir::ops::AccType>($_self).getElementType()">]> {
  let summary = "read component `comp` of an accessor at a constant offset from the current point";
  let description = [{
    ```mlir
    %v = ops.read %a[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %w = ops.read %b[0, 0] {comp = 2 : i64} : !ops.acc<f64, 2, 3> -> f64
    ```
    Inside an `ops.par_loop` the offset must be one of the stencil points of the dat, and the access mode
    must allow reading (everything but `write`).
  }];
  let arguments = (ins OPS_AccType:$acc, DenseI64ArrayAttr:$offsets, DefaultValuedAttr<I64Attr, "0">:$comp);
  let results = (outs OPS_ScalarType:$result);
  let assemblyFormat = "$acc custom<Offsets>($offsets) attr-dict `:` qualified(type($acc)) `->` type($result)";
  let hasVerifier = 1;
}

def OPS_ReadDynOp : OPS_Op<"read_dyn", [
    MemoryEffects<[MemRead]>,
    TypesMatchWith<"result type is the element type of the accessor", "acc", "result",
                   "::llvm::cast<::ops_mlir::ops::AccType>($_self).getElementType()">]> {
  let summary = "read an accessor at a data-dependent offset";
  let description = [{
    The offsets are computed by the kernel. The kernel's contract is that they are points of the stencil;
    lowering selects among the declared points.

    ```mlir
    %v = ops.read_dyn %a[%dx, %dy : i32, i32] : !ops.acc<f64, 2, 1> -> f64
    ```
  }];
  let arguments = (ins OPS_AccType:$acc, Variadic<OPS_OffsetType>:$offsets, DefaultValuedAttr<I64Attr, "0">:$comp);
  let results = (outs OPS_ScalarType:$result);
  let assemblyFormat = "$acc `[` $offsets `:` type($offsets) `]` attr-dict `:` qualified(type($acc)) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def OPS_WriteOp : OPS_Op<"write", [
    MemoryEffects<[MemWrite]>,
    TypesMatchWith<"value type is the element type of the accessor", "acc", "value",
                   "::llvm::cast<::ops_mlir::ops::AccType>($_self).getElementType()">]> {
  let summary = "write component `comp` of an accessor at a constant offset from the current point";
  let description = [{
    Writes at non-zero offsets are legal (boundary kernels write halo points): inside an `ops.par_loop` the
    offset must be one of the stencil points of the dat and the access mode must allow writing (everything
    but `read`).

    ```mlir
    ops.write %a[-1, 0], %v : !ops.acc<f64, 2, 1>
    ```
  }];
  let arguments = (ins OPS_AccType:$acc, DenseI64ArrayAttr:$offsets, OPS_ScalarType:$value,
                       DefaultValuedAttr<I64Attr, "0">:$comp);
  let assemblyFormat = "$acc custom<Offsets>($offsets) `,` $value attr-dict `:` qualified(type($acc))";
  let hasVerifier = 1;
}

def OPS_GblReadOp : OPS_Op<"gbl_read", [
    Pure,
    TypesMatchWith<"result type is the element type of the global", "gbl", "result",
                   "::llvm::cast<::ops_mlir::ops::GblType>($_self).getElementType()">]> {
  let summary = "read element `index` of a read-only global";
  let arguments = (ins OPS_GblType:$gbl, I64Attr:$index);
  let results = (outs OPS_ScalarType:$result);
  let assemblyFormat = "$gbl `[` $index `]` attr-dict `:` qualified(type($gbl)) `->` type($result)";
  let hasVerifier = 1;
}

def OPS_ReduceOp : OPS_Op<"reduce", [
    MemoryEffects<[MemWrite]>,
    TypesMatchWith<"value type is the element type of the reduction", "red", "value",
                   "::llvm::cast<::ops_mlir::ops::RedType>($_self).getElementType()">]> {
  let summary = "contribute a value to a reduction";
  let description = [{
    The loop declares how contributions are combined (add, min, max). A point that does not call
    `ops.reduce` contributes the identity.
  }];
  let arguments = (ins OPS_RedType:$red, OPS_ScalarType:$value);
  let assemblyFormat = "$red `,` $value attr-dict `:` qualified(type($red))";
}

def OPS_PrintOp : OPS_Op<"print", [MemoryEffects<[MemWrite]>]> {
  let summary = "printf from a kernel";
  let description = [{
    The C format string and the (already promoted) arguments of a `printf` call in a kernel. The order of the lines of
    different points is unspecified, as with the OpenMP and CUDA code of OPS. The runtime option OPS_JIT_PRINTF decides
    whether the operation is kept (host: `printf`, GPU: `gpu.printf`), dropped, or makes the JIT decline the loop.
  }];
  let arguments = (ins StrAttr:$format, Variadic<AnyTypeOf<[F64, I32, I64]>>:$values);
  let assemblyFormat = "$format (`,` $values^ `:` type($values))? attr-dict";
}

def OPS_IndexOp : OPS_Op<"index", [Pure]> {
  let summary = "global index of the current point along dimension `dim` (OPS axis order, x first)";
  let description = [{
    The index in the global index space of the block: with MPI the loop adds the displacement of the process's
    part of the block.
  }];
  let arguments = (ins I64Attr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
  let hasVerifier = 1;
}

def OPS_ConstRefOp : OPS_Op<"const_ref", [Pure]> {
  let summary = "value of a constant declared with ops_decl_const (kernel only)";
  let description = [{
    `@name` is the constant's name, `offset` the byte offset of the value inside the constant (a member of a
    struct constant), and for arrays an optional dynamic `index` with the element stride `array_stride` in
    bytes. ops-materialize-kernels turns references into arguments of the chain.

    ```mlir
    %dt = ops.const_ref @dt : f64
    %xmax = ops.const_ref @field {offset = 8 : i64} : i32
    %e = ops.const_ref @states[%i : i32] {offset = 8 : i64, array_stride = 72 : i64} : f64
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$name, DefaultValuedAttr<I64Attr, "0">:$offset,
                       Optional<OPS_OffsetType>:$index, DefaultValuedAttr<I64Attr, "0">:$array_stride);
  let results = (outs OPS_ScalarType:$result);
  let assemblyFormat = "$name (`[` $index^ `:` type($index) `]`)? attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def OPS_ReturnOp : OPS_Op<"return", [Pure, ReturnLike, Terminator, ParentOneOf<["KernelOp", "ChainOp"]>]> {
  let summary = "terminator of ops.kernel (no operands) and ops.chain (the reduction values)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def OPS_YieldOp : OPS_Op<"yield", [Pure, ReturnLike, Terminator, HasParent<"ParLoopOp">]> {
  let summary = "terminator of the body of an ops.par_loop";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Functions: kernels and chains
//===----------------------------------------------------------------------===//

def OPS_KernelOp : OPS_Op<"kernel", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "a user kernel";
  let description = [{
    The body of an OPS kernel function over its parameters: `!ops.acc` for dat arguments, `!ops.gbl` for
    read-only globals and `!ops.red` for reductions. A kernel does not depend on a loop: which offsets it may
    use and how an accessor is accessed are checked when it is inlined into an `ops.par_loop`.

    ```mlir
    ops.kernel @ideal_gas(%density: !ops.acc<f64, 2, 1>, %pressure: !ops.acc<f64, 2, 1>) {
      %d = ops.read %density[0, 0] : !ops.acc<f64, 2, 1> -> f64
      ops.write %pressure[0, 0], %d : !ops.acc<f64, 2, 1>
      ops.return
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility, OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::FunctionType":$type), [{
    $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name));
    $_state.addAttribute(getFunctionTypeAttrName($_state.name), ::mlir::TypeAttr::get(type));
    (void)$_state.addRegion();
  }]>];
  let extraClassDeclaration = [{
    /// Creates the entry block with one argument per parameter. The body must be empty.
    ::mlir::Block *addEntryBlock() {
      assert(getBody().empty() && "the function already has a body");
      ::mlir::Block *entry = new ::mlir::Block();
      getBody().push_back(entry);
      ::llvm::SmallVector<::mlir::Location> locs(getArgumentTypes().size(), getLoc());
      entry->addArguments(getArgumentTypes(), locs);
      return entry;
    }
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    static ::llvm::StringRef getDefaultDialect() { return "ops"; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def OPS_ChainOp : OPS_Op<"chain", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "a chain of loops";
  let description = [{
    What a chain executor lifts from OPS's queue: dats, loop ranges (index) and scalars as arguments, a
    sequence of `ops.par_loop`s, and the values of the reductions as results.

    ```mlir
    ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index,
                     %lo1: index, %hi1: index) {
      ops.par_loop ...
      ops.return
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility, OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::FunctionType":$type), [{
    $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name));
    $_state.addAttribute(getFunctionTypeAttrName($_state.name), ::mlir::TypeAttr::get(type));
    (void)$_state.addRegion();
  }]>];
  let extraClassDeclaration = [{
    /// Creates the entry block with one argument per parameter. The body must be empty.
    ::mlir::Block *addEntryBlock() {
      assert(getBody().empty() && "the function already has a body");
      ::mlir::Block *entry = new ::mlir::Block();
      getBody().push_back(entry);
      ::llvm::SmallVector<::mlir::Location> locs(getArgumentTypes().size(), getLoc());
      entry->addArguments(getArgumentTypes(), locs);
      return entry;
    }
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    static ::llvm::StringRef getDefaultDialect() { return "ops"; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}


def OPS_CallOp : OPS_Op<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "run a group of loops: a call of an ops.chain";
  let description = [{
    What `ops-outline-groups` leaves in a chain: each group of fused loops is its own `ops.chain`, and the
    original chain calls them in order. The operands are values of the calling chain, in the callee's argument
    order; the results are the reductions of the callee.

    ```mlir
    %m = ops.call @group_3(%a, %b, %lo0, %hi0, %lo1, %hi1) : (!ops.dat<f64, 2, 1>, !ops.dat<f64, 2, 1>, index,
                                                               index, index, index) -> f64
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)";
}

//===----------------------------------------------------------------------===//
// The loop
//===----------------------------------------------------------------------===//

def OPS_ParLoopOp : OPS_Op<"par_loop", [
    AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "a parallel loop over a range of a block";
  let description = [{
    Executes the body for every point of the range `[lo0, hi0) x [lo1, hi1) x ...` (OPS axis order, x first).
    The order of points is unspecified: the body must not depend on it.

    Each dat operand comes with its access mode and its stencil. The body has one `!ops.acc` argument per dat
    operand, then one argument per scalar operand (read-only values of globals, with the type of the
    operand), then one `!ops.red` argument per result. A result holds the value of one reduction; its kind is
    in `reduction_kinds`.

    ```mlir
    %m = ops.par_loop @calc_dt_kernel_min range(%lo0, %hi0, %lo1, %hi1)
        dats(%wk : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>)
        scalars()
        attributes {reduction_kinds = [#ops.red_kind<min>]} -> f64 {
    ^bb0(%w: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>):
      %v = ops.read %w[0, 0] : !ops.acc<f64, 2, 1> -> f64
      ops.reduce %r, %v : !ops.red<f64>
    }
    ```

    The `kernel` symbol is informational once the body is present: an empty body (only `ops.yield`) with a
    `kernel` symbol is the unmaterialised form, whose body ops-materialize-kernels fills in.
  }];
  let arguments = (ins Variadic<Index>:$range,
                       Variadic<OPS_DatType>:$dats,
                       Variadic<AnyType>:$scalars,
                       OptionalAttr<FlatSymbolRefAttr>:$kernel,
                       DefaultValuedAttr<I64Attr, "-1">:$loop_id,
                       DefaultValuedAttr<I64Attr, "0">:$block,
                       OPS_AccessModeArrayAttr:$accesses,
                       OPS_StencilArrayAttr:$stencils,
                       OptionalAttr<OPS_ReductionKindArrayAttr>:$reduction_kinds);
  let results = (outs Variadic<OPS_ScalarType>:$reductions);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($kernel^)? `range` `(` $range `)`
    `dats` `(` custom<DatOperands>($dats, type($dats), $accesses, $stencils) `)`
    `scalars` `(` ($scalars^ `:` type($scalars))? `)`
    attr-dict-with-keyword (`->` type($reductions)^)? $body
  }];
  let extraClassDeclaration = [{
    /// Rank of the block (number of loop dimensions).
    unsigned getRank() { return getRange().size() / 2; }
    unsigned getNumDats() { return getDats().size(); }
    AccessMode getAccess(unsigned i);
    StencilAttr getStencil(unsigned i);
    ReductionKind getReductionKind(unsigned i);
    ::mlir::Block &getBodyBlock() { return getBody().front(); }
    /// The accessor argument of the body that stands for dat operand `i`.
    ::mlir::BlockArgument getAccessorArg(unsigned i) { return getBodyBlock().getArgument(i); }
    /// The body argument for scalar operand `i`.
    ::mlir::BlockArgument getScalarArg(unsigned i) { return getBodyBlock().getArgument(getNumDats() + i); }
    /// The `!ops.red` argument of the body for result `i`.
    ::mlir::BlockArgument getReductionArg(unsigned i) {
      return getBodyBlock().getArgument(getNumDats() + getScalars().size() + i);
    }
    /// True if the body is only the terminator.
    bool isBodyEmpty() { return getBodyBlock().without_terminator().empty(); }
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // OPS_MLIR_DIALECT_OPS_OPSOPS
