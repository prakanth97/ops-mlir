//===- StoreForwarding.h - read back what the loop body stored --------*- C++ -*-===//

#ifndef OPS_MLIR_TRANSFORMS_STOREFORWARDING_H
#define OPS_MLIR_TRANSFORMS_STOREFORWARDING_H

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

namespace ops_mlir::ops {

/// Inside the body of `loop`: `ops.write %a[o] {comp = c}, %v` followed by `ops.read %a[o] {comp = c}` of the same
/// accessor, with no write to that accessor in between (a nested region that writes it forgets the value), uses `%v`.
/// Accessors of a dat that the loop passes through two operands are left alone.
void forwardStoresInLoop(ParLoopOp loop);

/// Narrows the access modes to what the body does: an operand that is no longer read becomes `write`, one that
/// is never written `read`.
void narrowAccessModes(ParLoopOp loop);

} // namespace ops_mlir::ops

#endif // OPS_MLIR_TRANSFORMS_STOREFORWARDING_H
