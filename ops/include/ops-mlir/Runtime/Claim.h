//===- Claim.h - may the JIT execute this loop? ----------------------------*- C++ -*-===//
//
// Plan section 4.5. Cheap, from the descriptor and the artifact only (no IR). Every reason for declining is a
// sentence for OPS_JIT_EXPLAIN. A declined loop runs with the stock OPS code.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_CLAIM_H
#define OPS_MLIR_RUNTIME_CLAIM_H

#include "ops-mlir/Runtime/Artifact.h"

#include "ops_lib_core.h"
#include "ops_internal2.h"

#include <string>

namespace ops_mlir::runtime {

/// Empty if the loop can be executed by the JIT, else why not. `artifact` is the decoded artifact of the loop.
/// `deviceBuild`: whether the loops of this build run on a device (CUDA/HIP).
std::string claimReason(ops_kernel_descriptor *desc, const KernelArtifact &artifact, bool deviceSupported);

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_CLAIM_H
