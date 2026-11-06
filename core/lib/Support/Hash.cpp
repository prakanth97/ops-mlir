#include "opdsl/Support/Hash.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

namespace opdsl {

std::string sha256Hex(llvm::StringRef data) {
  llvm::SHA256 h;
  h.update(data);
  std::array<uint8_t, 32> digest = h.final();
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), /*LowerCase=*/true);
}

Hasher &Hasher::add(llvm::StringRef s) {
  // 'S', length, ':', bytes: unambiguous when concatenated.
  buffer_ += 'S';
  buffer_ += std::to_string(s.size());
  buffer_ += ':';
  buffer_.append(s.data(), s.size());
  return *this;
}

Hasher &Hasher::addInt(int64_t v) {
  buffer_ += 'I';
  buffer_ += std::to_string(v);
  buffer_ += ';';
  return *this;
}

std::string Hasher::hex() const { return sha256Hex(buffer_); }

} // namespace opdsl
