#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "IRBuilder.h"
#include "Core.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <functional>
#include <mutex>

namespace ops_mlir {

class JITEngine {
public:
  using FlushCallback = std::function<void(const std::vector<LoopDesc> &)>;

  static JITEngine &instance();

  void setFlushCallback(FlushCallback callback);

  void enqueueParLoop(std::uintptr_t kernelToken, const char *kernelName,
                      ops_block block, int dims, const int *range,
                      const ops_arg *args, std::size_t nargs);

  void flush();

  void compile();

  void execute();

  const std::vector<LoopDesc> &queue() const { return queue_; }

private:
  JITEngine();

  mlir::MLIRContext ctx;
  mlir::ModuleOp module;
  IRBuilder builder{&ctx};

  LoopDesc buildLoopDesc(std::uintptr_t kernelToken, const char *kernelName,
                         ops_block block, int dims, const int *range,
                         const ops_arg *args, std::size_t nargs);

  ArgDesc buildArgDesc(const ops_arg &arg);

  DatDesc describeDat(ops_dat dat);
  StencilDesc describeStencil(ops_stencil stencil);

  /// Run the ops.par_loop -> func.func+stencil.* pass (see
  /// lib/passes/OPSToStencil.cpp) on a clone of `module`, verify it, and
  /// return its textual form (empty on failure).
  std::string runStencilLowering();

private:
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
};

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
