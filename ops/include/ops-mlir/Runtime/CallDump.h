//===- CallDump.h - record a call of a compiled group, replay it offline ------*- C++ -*-===//
//
// OPS_JIT_DUMP_DIR=<dir> OPS_JIT_DUMP_DATA=1: the first call of every group writes, next to the group's IR
// (<entry>.group.mlir),
//
//   <entry>.call.json     pipeline, ABI (memref shapes, indices, scalars), results, checksums of the dats after the call
//   <entry>.dat<N>.bin    the allocation of dat N before the call
//
// `ops-jit-replay <dir> <entry>` (tools/ops-jit-replay) loads that, compiles the group, runs it on the recorded
// inputs and compares the checksums. No OPS needed.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_CALLDUMP_H
#define OPS_MLIR_RUNTIME_CALLDUMP_H

#include "ops-mlir/Runtime/PackedArgs.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir::runtime {

/// FNV-1a 64 over `n` bytes, as 16 hex digits.
std::string checksumHex(const void *data, size_t n);

/// Writes what is known before the call: the inputs. Returns the JSON text prefix is kept in the object.
class CallRecorder {
public:
  /// `datElems`: element type name per memref of `args`. Writes <dir>/<entry>.dat<N>.bin.
  CallRecorder(std::string dir, std::string entry, std::string pipeline, std::string pipelineOptions, const PackedArgs &args,
               std::vector<std::string> datElems);

  /// After the call: checksums of the dats, the results; writes <dir>/<entry>.call.json.
  bool finish(std::string &error);

private:
  std::string dir, entry, pipeline, pipelineOptions;
  const PackedArgs &args;
  std::vector<std::string> datElems;
  bool inputsWritten = false;
};

/// A recorded call, loaded: owns the buffers of the dats.
struct LoadedCall {
  std::string entry, pipeline, pipelineOptions;
  PackedArgs args;
  std::vector<std::vector<char>> buffers;
  std::vector<std::string> datElems;
  std::vector<std::string> expectedChecksums; ///< per dat, after the call
  std::vector<double> expectedResults;
};

/// Reads <dir>/<entry>.call.json and the inputs. False with `error` on failure.
bool loadCall(const std::string &dir, const std::string &entry, LoadedCall &out, std::string &error);

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_CALLDUMP_H
