// runtime/GpuReductionWriteback.h / .cpp
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace ops_mlir {

// Runs BEFORE gpu-map-parallel-loops/convert-parallel-loops-to-gpu.
// This is because when convert-parallel-loops-to-gpu does not allow scf.parallel to return SSA values.
// It will remove the returned value, but only if it is not used anywhere later in the IR.
struct StripGpuReductionWritebackPass
    : public mlir::PassWrapper<StripGpuReductionWritebackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final {
    return "ops-strip-gpu-reduction-writeback";
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::scf::ParallelOp parallelOp) {
      for (mlir::Value result : parallelOp.getResults()) {
        llvm::SmallVector<mlir::Operation *> toErase;
        for (mlir::Operation *user : result.getUsers())
          if (llvm::isa<mlir::memref::StoreOp>(user))
            toErase.push_back(user);
        for (mlir::Operation *op : toErase)
          op->erase();
      }
    });
  }
};

// Runs AFTER convert-parallel-loops-to-gpu. 
// Matches gpu.all_reduce ops (in order, walking each gpu.launch body) against the enclosing func.func's
// reduction-handle parameters (in order).
struct InsertGpuReductionWritebackPass
    : public mlir::PassWrapper<InsertGpuReductionWritebackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final {
    return "ops-insert-gpu-reduction-writeback";
  }

  void runOnOperation() override {
    mlir::func::FuncOp fn = getOperation();

    llvm::SmallVector<mlir::Value> reduceHandles;
    for (mlir::BlockArgument arg : fn.getArguments()) {
      auto memrefType = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
      if (memrefType && memrefType.getElementType().isF64() &&
          memrefType.getRank() == 0)
        reduceHandles.push_back(arg);
    }
    if (reduceHandles.empty())
      return;

    std::size_t handleIndex = 0;
    fn.walk([&](mlir::gpu::LaunchOp launchOp) {
      launchOp.getBody().walk([&](mlir::gpu::AllReduceOp allReduceOp) {
        if (handleIndex >= reduceHandles.size()) {
          allReduceOp.emitWarning(
              "more gpu.all_reduce ops than reduction-handle arguments");
          return;
        }
        mlir::Value handle = reduceHandles[handleIndex++];

        mlir::OpBuilder b(allReduceOp->getContext());
        b.setInsertionPointAfter(allReduceOp);
        mlir::Location loc = allReduceOp.getLoc();

        mlir::Value tidX = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
        mlir::Value tidY = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::y);
        mlir::Value tidZ = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::z);
        mlir::Value c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value isX0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tidX, c0);
        mlir::Value isY0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tidY, c0);
        mlir::Value isZ0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tidZ, c0);
        mlir::Value isXY0 = b.create<mlir::arith::AndIOp>(loc, isX0, isY0);
        mlir::Value isThread0 = b.create<mlir::arith::AndIOp>(loc, isXY0, isZ0);

        b.create<mlir::scf::IfOp>(loc, isThread0,
            [&](mlir::OpBuilder &ib, mlir::Location l) {
              // TODO: combiner kind (maxf/minf/addf) is hardcoded below
              // need to get access to this, potentially by adding it as an attribute to the func.func?
              ib.create<mlir::memref::AtomicRMWOp>(
                  l, ib.getF64Type(), mlir::arith::AtomicRMWKind::maximumf,
                  allReduceOp.getResult(), handle, mlir::ValueRange{});
              ib.create<mlir::scf::YieldOp>(l);
            });
      });
    });
  }
};

} // namespace ops_mlir
