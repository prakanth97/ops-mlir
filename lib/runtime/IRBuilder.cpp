#include "runtime/IRBuilder.h"
#include "Dialect/OPS/OPSDialect.h"
#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"


// TODO: Casting pointers to int64_t is not portable. We should use a better way to represent pointers in MLIR attributes.

namespace ops_mlir {

IRBuilder::IRBuilder(mlir::MLIRContext *ctx) : ctx_(ctx) {
  // Register OPS dialect
  ctx_->getOrLoadDialect<ops_mlir::ops::OPSDialect>();
}

mlir::ModuleOp IRBuilder::buildModule(const std::vector<LoopDesc> &loops) {
  if (loops.empty()) {
    auto loc = mlir::UnknownLoc::get(ctx_);
    return mlir::ModuleOp::create(loc);
  }

  auto loc = mlir::UnknownLoc::get(ctx_);
  auto module = mlir::ModuleOp::create(loc);

  for (const auto &loop : loops) {
    auto *op = buildParLoopOp(loop);
    if (!op) {
      llvm::errs() << "Failed to build ops.par_loop for: " << loop.kernel_name
                   << "\n";
      return nullptr;
    }
    module.getBody()->push_back(op);
  }

  return module;
}

std::string IRBuilder::moduleToString(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  return result;
}

mlir::Operation *IRBuilder::buildParLoopOp(const LoopDesc &loop) {
  auto loc = mlir::UnknownLoc::get(ctx_);
  mlir::OpBuilder builder(ctx_);

  // Infer block dims from dat arguments
  int ndim = loop.dims;
  for (const auto &arg : loop.args) {
    if (arg.argtype == OPS_ARG_DAT && !arg.dat.size.empty()) {
      ndim = std::max(ndim, static_cast<int>(arg.dat.size.size()));
    }
  }

  // Build range attribute
  std::vector<int64_t> rangeVec(loop.range.begin(), loop.range.end());
  auto rangeAttr = mlir::DenseI64ArrayAttr::get(ctx_, rangeVec);

  // Build kernel ptr attribute
  auto kernelPtrAttr =
      builder.getI64IntegerAttr(static_cast<int64_t>(loop.kernel_token));

  // Build kernel name attribute
  auto kernelNameAttr = builder.getStringAttr(loop.kernel_name);

  // Build dims attribute
  auto dimsAttr = builder.getI32IntegerAttr(loop.dims);

  // Build argument descriptors
  mlir::SmallVector<mlir::Attribute> argAttrs;
  argAttrs.reserve(loop.args.size());
  for (const auto &arg : loop.args) {
    auto argAttr = buildArgAttr(arg);
    if (!argAttr) {
      llvm::errs() << "Failed to build arg attribute\n";
      return nullptr;
    }
    argAttrs.push_back(argAttr);
  }
  auto argsAttr = builder.getArrayAttr(argAttrs);

  // Build the ops.par_loop operation
  mlir::SmallVector<mlir::Value> operands; // No operands in Phase 1
  auto opState =
      mlir::OperationState(loc, ops_mlir::ops::ParLoopOp::getOperationName());
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "kernel_ptr"), kernelPtrAttr});
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "kernel_name"), kernelNameAttr});
  opState.attributes.push_back({mlir::StringAttr::get(ctx_, "dims"), dimsAttr});
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "range"), rangeAttr});
  opState.attributes.push_back({mlir::StringAttr::get(ctx_, "args"), argsAttr});

  if (ndim > 0) {
    opState.attributes.push_back({mlir::StringAttr::get(ctx_, "block_dims"),
                                  builder.getI32IntegerAttr(ndim)});
  }

  return builder.create(opState);
}

mlir::Attribute IRBuilder::buildArgAttr(const ArgDesc &arg) {
  auto datAttr = buildDatAttr(arg.dat);
  auto stencilAttr = buildStencilAttr(arg.stencil);

  return ops_mlir::ops::ArgAttr::get(
      ctx_,
      datAttr,
      stencilAttr,
      arg.dim,
      arg.elem_size,
      static_cast<int64_t>(arg.data),
      static_cast<int64_t>(arg.data_d),
      arg.acc,
      arg.argtype,
      arg.opt);
}

ops_mlir::ops::DatAttr IRBuilder::buildDatAttr(const DatDesc &dat) {
  mlir::OpBuilder builder(ctx_);

  auto toI64Array = [&](const std::vector<int64_t> &vals) {
    return mlir::DenseI64ArrayAttr::get(ctx_, vals);
  };

  return ops_mlir::ops::DatAttr::get(
      ctx_,
      dat.index,
      static_cast<int64_t>(dat.block),
      dat.dim,
      dat.type_size,
      dat.elem_size,
      toI64Array(dat.size),
      toI64Array(dat.base),
      toI64Array(dat.d_m),
      toI64Array(dat.d_p),
      toI64Array(dat.stride),
      builder.getStringAttr(dat.name),
      builder.getStringAttr(dat.type),
      static_cast<int64_t>(dat.data),
      static_cast<int64_t>(dat.data_d)
    );
}

ops_mlir::ops::StencilAttr IRBuilder::buildStencilAttr(const StencilDesc &stencil) {
  mlir::OpBuilder builder(ctx_);

  return ops_mlir::ops::StencilAttr::get(
      ctx_,
      stencil.index,
      stencil.dims,
      stencil.points,
      builder.getStringAttr(stencil.name),
      static_cast<int64_t>(stencil.stencil),
      static_cast<int64_t>(stencil.stride),
      static_cast<int64_t>(stencil.mgrid_stride),
      stencil.type);
}

} // namespace ops_mlir
