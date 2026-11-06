#include "ops-mlir/Analysis/Passes.h"

#include "ops-mlir/Analysis/ChainDependence.h"
#include "ops-mlir/Analysis/FusionPlanner.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace ops_mlir::ops {
namespace {

struct PrintChainDependencePass
    : public PassWrapper<PrintChainDependencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintChainDependencePass)
  StringRef getArgument() const final { return "ops-print-chain-dependence"; }
  StringRef getDescription() const final { return "print the loops and dependence edges of every ops.chain"; }
  void runOnOperation() override {
    getOperation().walk([&](ChainOp chain) {
      auto dep = ChainDependence::build(chain);
      if (failed(dep)) return signalPassFailure();
      llvm::errs() << "chain @" << chain.getSymName() << '\n';
      dep->print(llvm::errs());
    });
  }
};

struct PrintFusionPlanPass : public PassWrapper<PrintFusionPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintFusionPlanPass)
  PrintFusionPlanPass() = default;
  PrintFusionPlanPass(const PrintFusionPlanPass &other) : PassWrapper(other) {}

  Option<int64_t> maxLoops{*this, "max-loops", llvm::cl::desc("upper bound on loops per kernel"),
                           llvm::cl::init(8)};
  Option<bool> reorder{*this, "reorder", llvm::cl::desc("let independent loops move past each other"),
                       llvm::cl::init(true)};
  Option<std::string> placement{*this, "placement", llvm::cl::desc("earliest or latest group a loop joins"),
                                llvm::cl::init("earliest")};
  Option<bool> guarded{*this, "guarded", llvm::cl::desc("fuse loops of different ranges over their bounding box"),
                       llvm::cl::init(false)};
  Option<double> boxRatio{*this, "box-ratio", llvm::cl::desc("bounding box volume / member volumes limit"),
                          llvm::cl::init(1.0)};

  StringRef getArgument() const final { return "ops-print-fusion-plan"; }
  StringRef getDescription() const final { return "print the fusion plan of every ops.chain"; }
  void runOnOperation() override {
    FusionOptions opts;
    opts.maxGroupSize = std::max<int64_t>(1, maxLoops);
    opts.reorder = reorder;
    opts.allowGuarded = guarded;
    opts.boxRatio = boxRatio;
    if (placement == "latest") opts.placement = FusionOptions::Placement::Latest;
    else if (placement != "earliest") {
      getOperation().emitError("unknown placement '") << placement << "' (earliest or latest)";
      return signalPassFailure();
    }
    getOperation().walk([&](ChainOp chain) {
      auto loops = extractLoops(chain);
      if (failed(loops)) return signalPassFailure();
      FusionPlan plan = planFusion(*loops, opts);
      TrafficEstimate traffic = estimateTraffic(*loops, plan);
      llvm::errs() << "chain @" << chain.getSymName() << '\n';
      llvm::errs() << "plan: " << plan.digest() << '\n';
      llvm::errs() << "kernels: " << plan.groups.size() << " of " << loops->size() << " loops\n";
      llvm::errs() << "reordered: " << plan.numReordered() << '\n';
      for (const auto &[reason, count] : plan.blockedBy) llvm::errs() << "blocked-by " << reason << ": " << count << '\n';
      llvm::errs() << "traffic: unfused " << static_cast<long long>(traffic.unfusedBytes) << " fused "
                   << static_cast<long long>(traffic.fusedBytes) << '\n';
    });
  }
};

} // namespace

void registerAnalysisPasses() {
  PassRegistration<PrintChainDependencePass>();
  PassRegistration<PrintFusionPlanPass>();
}

} // namespace ops_mlir::ops
