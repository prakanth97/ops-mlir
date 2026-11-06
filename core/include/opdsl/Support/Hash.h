//===- Hash.h - SHA-256 helpers for cache keys ------------------*- C++ -*-===//
//
// Cache keys (structure, constants, compiled groups) are SHA-256 digests. Hasher feeds typed, length-
// prefixed fields so that ("ab","c") and ("a","bc") hash differently.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_SUPPORT_HASH_H
#define OPDSL_SUPPORT_HASH_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace opdsl {

/// Lower-case hex SHA-256 of `data`.
std::string sha256Hex(llvm::StringRef data);

/// Incremental hash over typed fields.
class Hasher {
public:
  Hasher &add(llvm::StringRef s);
  Hasher &add(const char *s) { return add(llvm::StringRef(s)); }
  Hasher &add(const std::string &s) { return add(llvm::StringRef(s)); }
  Hasher &addInt(int64_t v);
  Hasher &addBool(bool v) { return addInt(v ? 1 : 0); }

  /// Lower-case hex digest of everything added so far. The hasher can keep being used afterwards.
  std::string hex() const;

private:
  std::string buffer_;
};

} // namespace opdsl

#endif // OPDSL_SUPPORT_HASH_H
