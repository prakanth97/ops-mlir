#include "opdsl/Cache/PersistentCache.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace opdsl {

namespace {

constexpr char kMagic[8] = {'O', 'P', 'D', 'S', 'L', 'C', '0', '1'};
constexpr size_t kFixedHeader = 8 + 4 + 8 + 8 + 8; // magic, version, header size, payload size, checksum
constexpr llvm::StringLiteral kSuffix = ".opdslc";

llvm::Error fail(const llvm::Twine &message) { return llvm::createStringError(llvm::errc::io_error, message); }

void put32(std::string &out, uint32_t v) {
  char b[4];
  llvm::support::endian::write32le(b, v);
  out.append(b, 4);
}
void put64(std::string &out, uint64_t v) {
  char b[8];
  llvm::support::endian::write64le(b, v);
  out.append(b, 8);
}

uint64_t checksum(llvm::StringRef header, llvm::StringRef payload) {
  return llvm::xxh3_64bits(header) * 1099511628211ull ^ llvm::xxh3_64bits(payload);
}

struct DiskEntry {
  std::string path;
  uint64_t size;
  llvm::sys::TimePoint<> time;
};

std::vector<DiskEntry> scan(llvm::StringRef dir) {
  std::vector<DiskEntry> entries;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (!llvm::StringRef(it->path()).ends_with(kSuffix)) continue;
    llvm::sys::fs::file_status status;
    if (it->status().getError()) continue;
    auto st = it->status();
    entries.push_back({it->path(), st->getSize(), st->getLastModificationTime()});
  }
  return entries;
}

} // namespace

std::string PersistentCache::defaultDirectory() {
  if (const char *dir = std::getenv("OPS_JIT_CACHE_DIR"); dir && *dir) return dir;
  llvm::SmallString<256> path;
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    path = xdg;
  } else if (!llvm::sys::path::home_directory(path)) {
    path = "/tmp";
  } else {
    llvm::sys::path::append(path, ".cache");
  }
  llvm::sys::path::append(path, "opdsl-jit");
  return std::string(path);
}

PersistentCache::PersistentCache(CacheOptions opts) : options(std::move(opts)) {
  dir = options.directory.empty() ? defaultDirectory() : options.directory;
  llvm::sys::fs::create_directories(dir);
  approxBytes = sizeOnDisk();
}

std::string PersistentCache::pathOf(llvm::StringRef key) const {
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, key + kSuffix);
  return std::string(path);
}

uint64_t PersistentCache::sizeOnDisk() const {
  uint64_t total = 0;
  for (const DiskEntry &e : scan(dir)) total += e.size;
  return total;
}

std::optional<CacheEntry> PersistentCache::lookup(llvm::StringRef key) {
  std::string path = pathOf(key);
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!buffer) {
    ++counters.misses;
    return std::nullopt;
  }
  llvm::StringRef data = (*buffer)->getBuffer();
  auto corrupt = [&]() -> std::optional<CacheEntry> {
    ++counters.corrupt;
    ++counters.misses;
    llvm::sys::fs::remove(path);
    return std::nullopt;
  };
  if (data.size() < kFixedHeader || std::memcmp(data.data(), kMagic, 8) != 0) return corrupt();
  const char *p = data.data() + 8;
  uint32_t version = llvm::support::endian::read32le(p);
  uint64_t headerSize = llvm::support::endian::read64le(p + 4);
  uint64_t payloadSize = llvm::support::endian::read64le(p + 12);
  uint64_t sum = llvm::support::endian::read64le(p + 20);
  if (version != options.formatVersion) return corrupt();
  if (headerSize > data.size() || payloadSize > data.size() || kFixedHeader + headerSize + payloadSize != data.size())
    return corrupt();
  llvm::StringRef header = data.substr(kFixedHeader, headerSize);
  llvm::StringRef payload = data.substr(kFixedHeader + headerSize, payloadSize);
  if (checksum(header, payload) != sum) return corrupt();

  ++counters.hits;
  // Refresh the use time for LRU eviction.
  int fd;
  if (!llvm::sys::fs::openFileForRead(path, fd)) {
    llvm::sys::fs::setLastAccessAndModificationTime(fd, std::chrono::system_clock::now());
    llvm::sys::Process::SafelyCloseFileDescriptor(fd);
  }
  return CacheEntry{header.str(), payload.str()};
}

llvm::Error PersistentCache::store(llvm::StringRef key, llvm::StringRef header, llvm::StringRef payload) {
  std::string blob;
  blob.reserve(kFixedHeader + header.size() + payload.size());
  blob.append(kMagic, 8);
  put32(blob, options.formatVersion);
  put64(blob, header.size());
  put64(blob, payload.size());
  put64(blob, checksum(header, payload));
  blob.append(header.data(), header.size());
  blob.append(payload.data(), payload.size());

  if (std::error_code ec = llvm::sys::fs::create_directories(dir)) return fail("cannot create " + dir + ": " + ec.message());

  // Temporary file in the same directory (same file system, so the rename is atomic), unique per writer.
  llvm::SmallString<256> tmpPattern(dir);
  llvm::sys::path::append(tmpPattern, ".tmp-%%%%%%%%%%%%");
  int fd;
  llvm::SmallString<256> tmpPath;
  if (std::error_code ec = llvm::sys::fs::createUniqueFile(tmpPattern, fd, tmpPath)) return fail("cannot create a temporary file in " + dir + ": " + ec.message());
  {
    llvm::raw_fd_ostream out(fd, /*shouldClose=*/true);
    out.write(blob.data(), blob.size());
    out.flush();
    if (out.has_error()) {
      std::error_code ec = out.error();
      out.clear_error();
      llvm::sys::fs::remove(tmpPath);
      return fail("cannot write " + std::string(tmpPath) + ": " + ec.message());
    }
  }
  if (std::error_code ec = llvm::sys::fs::rename(tmpPath, pathOf(key))) {
    llvm::sys::fs::remove(tmpPath);
    return fail("cannot rename into " + pathOf(key) + ": " + ec.message());
  }
  ++counters.stores;
  approxBytes += blob.size();
  if (approxBytes > options.maxBytes) trim();
  return llvm::Error::success();
}

void PersistentCache::trim() {
  std::vector<DiskEntry> entries = scan(dir);
  uint64_t total = 0;
  for (const DiskEntry &e : entries) total += e.size;
  approxBytes = total;
  if (total <= options.maxBytes) return;
  std::sort(entries.begin(), entries.end(), [](const DiskEntry &a, const DiskEntry &b) { return a.time < b.time; });
  const uint64_t target = options.maxBytes / 10 * 9;
  for (const DiskEntry &e : entries) {
    if (total <= target) break;
    if (!llvm::sys::fs::remove(e.path)) {
      total -= e.size;
      ++counters.evicted;
    }
  }
  approxBytes = total;
}

} // namespace opdsl
