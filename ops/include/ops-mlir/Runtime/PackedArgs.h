//===- PackedArgs.h - the argument block of a call through the packed wrapper -*- C++ -*-===//
//
// Plan section 4.6. A compiled group is called through `void(void **args)`: args[i] points at the storage of
// argument i, and after the last argument at the storage of the result(s). The arguments are the memref
// descriptors of the dats, then 64-bit indices (shifts, range bounds, displacements), then scalars. This class owns
// that storage. It knows nothing about OPS: the executor fills it from descriptors, the replay tool from a dump.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_PACKEDARGS_H
#define OPS_MLIR_RUNTIME_PACKEDARGS_H

#include "opdsl/JIT/JITSession.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir::runtime {

/// Size in bytes of an element type name (f64, f32, i64, i32, f16, i16).
size_t sizeOfElem(const std::string &elem);

class PackedArgs {
public:
  /// A strided memref over `data`. `sizes` and `strides` in memory order (slowest first), in elements; at most 4.
  void addMemref(void *data, const std::vector<int64_t> &sizes, const std::vector<int64_t> &strides);
  void addIndex(int64_t value) {
    order.push_back({Slot::Index, indices.size()});
    indices.push_back(value);
  }
  /// A scalar argument: `bytes` of storage holding a value of type `elem`.
  void addScalar(const std::string &elem, const void *bytes);
  /// The element types of the results (several: returned as one struct, natural alignment).
  void setResults(const std::vector<std::string> &elems);

  /// Builds the argument vector, in the order the arguments were added; call after everything was added, before call().
  void finalize();
  void call(opdsl::JITSession::PackedFn fn) { fn(argv.data()); }

  double resultAsDouble(size_t i) const;
  int64_t resultAsInt(size_t i) const;

  // For dumps and tests.
  struct MemrefInfo {
    void *data;
    std::vector<int64_t> sizes, strides;
  };
  const std::vector<MemrefInfo> &memrefInfos() const { return memrefInfo; }
  const std::vector<int64_t> &indexArgs() const { return indices; }
  struct ScalarInfo {
    std::string elem;
    std::array<char, 8> bytes;
  };
  const std::vector<ScalarInfo> &scalarInfos() const { return scalarInfo; }
  const std::vector<std::string> &resultElems() const { return resultElem; }

private:
  struct Memref {
    std::array<int64_t, 3 + 2 * 4> words{}; // allocated, aligned, offset, sizes[r], strides[r]
  };
  struct Slot {
    enum Kind { MemrefArg, Index, Scalar } kind;
    size_t i;
  };
  std::vector<Slot> order;
  std::vector<Memref> memrefs;
  std::vector<MemrefInfo> memrefInfo;
  std::vector<Memref *> memrefPointers;
  std::vector<int64_t> indices;
  std::vector<ScalarInfo> scalarInfo;
  std::vector<void *> argv;
  std::vector<char> results;
  std::vector<size_t> resultOffset;
  std::vector<std::string> resultElem;
};

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_PACKEDARGS_H
