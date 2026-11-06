//===- PersistentCache.h - on-disk cache of compiled code ----------*- C++ -*-===//
//
// Maps a key (a hex digest the caller computes, see plan section 4.7) to a payload (an object file) and a small
// header (JSON text: what it was built for, for humans and for sanity checks).
//
//  * Entries are files `<dir>/<key>.opdslc`: magic, format version, sizes, a checksum, header, payload.
//  * Writes are atomic: a temporary file in the same directory, then rename. Concurrent writers of one key (two
//    MPI ranks, two processes) both succeed and one file wins; a reader never sees a partial file.
//  * A file that is truncated, corrupt or of another format version is a miss (and is removed).
//  * The directory is capped: past `maxBytes` the least recently used entries are deleted (a hit refreshes the
//    modification time).
//  * Location: $OPS_JIT_CACHE_DIR, else $XDG_CACHE_HOME/opdsl-jit, else ~/.cache/opdsl-jit.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_CACHE_PERSISTENTCACHE_H
#define OPDSL_CACHE_PERSISTENTCACHE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace opdsl {

struct CacheOptions {
  /// Empty: PersistentCache::defaultDirectory().
  std::string directory;
  uint64_t maxBytes = uint64_t(2) << 30;
  /// Part of the file format check: bump it when the payload convention changes.
  uint32_t formatVersion = 1;
};

struct CacheEntry {
  std::string header;
  std::string payload;
};

struct CacheStats {
  uint64_t hits = 0, misses = 0, stores = 0, corrupt = 0, evicted = 0;
};

class PersistentCache {
public:
  explicit PersistentCache(CacheOptions options = {});

  static std::string defaultDirectory();

  const std::string &directory() const { return dir; }

  /// A valid entry, or nullopt (never an error: a cache miss is not a failure).
  std::optional<CacheEntry> lookup(llvm::StringRef key);

  /// Stores atomically. A failure is returned for logging; compiling goes on without the cache.
  llvm::Error store(llvm::StringRef key, llvm::StringRef header, llvm::StringRef payload);

  /// Deletes least recently used entries until the directory is below 90% of maxBytes (also done by store() when the
  /// size crosses the cap).
  void trim();

  /// Total size of the entries on disk now.
  uint64_t sizeOnDisk() const;

  const CacheStats &stats() const { return counters; }

  std::string pathOf(llvm::StringRef key) const;

private:
  CacheOptions options;
  std::string dir;
  uint64_t approxBytes = 0;
  CacheStats counters;
};

} // namespace opdsl

#endif // OPDSL_CACHE_PERSISTENTCACHE_H
