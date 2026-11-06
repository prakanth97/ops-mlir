#include "opdsl/Cache/PersistentCache.h"
#include "opdsl/Testing/TestHarness.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <thread>

using opdsl::CacheOptions;
using opdsl::PersistentCache;

namespace {

/// A fresh empty directory under the temp dir, removed at the end of the scope.
struct TempDir {
  std::string path;
  TempDir() {
    llvm::SmallString<128> p;
    llvm::sys::fs::createUniqueDirectory("opdsl-cache-test", p);
    path = std::string(p);
  }
  ~TempDir() { llvm::sys::fs::remove_directories(path); }
};

CacheOptions in(const TempDir &d, uint64_t cap = uint64_t(1) << 20) {
  CacheOptions o;
  o.directory = d.path;
  o.maxBytes = cap;
  return o;
}

} // namespace

OPDSL_TEST(PersistentCache, MissThenHit) {
  TempDir dir;
  PersistentCache cache(in(dir));
  OPDSL_EXPECT(!cache.lookup("abc123"));
  OPDSL_EXPECT(!cache.store("abc123", "{\"target\": \"x\"}", std::string("object\0bytes", 12)));
  auto hit = cache.lookup("abc123");
  OPDSL_EXPECT(hit.has_value());
  if (hit) {
    OPDSL_EXPECT_EQ(hit->header, "{\"target\": \"x\"}");
    OPDSL_EXPECT_EQ(hit->payload, std::string("object\0bytes", 12));
  }
  OPDSL_EXPECT_EQ(cache.stats().hits, uint64_t(1));
  OPDSL_EXPECT_EQ(cache.stats().misses, uint64_t(1));
  OPDSL_EXPECT_EQ(cache.stats().stores, uint64_t(1));
  // Another cache object on the same directory (another process, another run) sees it.
  PersistentCache other(in(dir));
  OPDSL_EXPECT(other.lookup("abc123").has_value());
  // Empty payloads and headers are fine.
  OPDSL_EXPECT(!cache.store("empty", "", ""));
  auto empty = cache.lookup("empty");
  OPDSL_EXPECT(empty.has_value() && empty->header.empty() && empty->payload.empty());
}

OPDSL_TEST(PersistentCache, OverwriteReplaces) {
  TempDir dir;
  PersistentCache cache(in(dir));
  OPDSL_EXPECT(!cache.store("k", "h1", "payload one"));
  OPDSL_EXPECT(!cache.store("k", "h2", "payload two, longer"));
  auto hit = cache.lookup("k");
  OPDSL_EXPECT(hit.has_value() && hit->payload == "payload two, longer" && hit->header == "h2");
}

OPDSL_TEST(PersistentCache, CorruptFilesAreMisses) {
  TempDir dir;
  PersistentCache cache(in(dir));
  OPDSL_EXPECT(!cache.store("good", "h", "some payload bytes"));
  std::string path = cache.pathOf("good");

  // Flip a payload byte.
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(-3, std::ios::end);
    f.put('X');
  }
  OPDSL_EXPECT(!cache.lookup("good"));
  OPDSL_EXPECT_EQ(cache.stats().corrupt, uint64_t(1));
  OPDSL_EXPECT(!llvm::sys::fs::exists(path)); // removed, so it is rewritten next time
  OPDSL_EXPECT(!cache.store("good", "h", "some payload bytes"));
  OPDSL_EXPECT(cache.lookup("good").has_value());

  // Truncated.
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << "OPDSLC01";
  }
  OPDSL_EXPECT(!cache.lookup("good"));
  // Garbage.
  {
    std::ofstream f(cache.pathOf("garbage"), std::ios::binary);
    f << "this is not a cache file at all, not even close to the right length or header";
  }
  OPDSL_EXPECT(!cache.lookup("garbage"));
  OPDSL_EXPECT_EQ(cache.stats().corrupt, uint64_t(3));
}

OPDSL_TEST(PersistentCache, FormatVersionMismatchIsAMiss) {
  TempDir dir;
  CacheOptions v1 = in(dir), v2 = in(dir);
  v1.formatVersion = 1;
  v2.formatVersion = 2;
  PersistentCache a(v1), b(v2);
  OPDSL_EXPECT(!a.store("k", "h", "p"));
  OPDSL_EXPECT(!b.lookup("k"));
  OPDSL_EXPECT(!b.store("k", "h", "p"));
  OPDSL_EXPECT(b.lookup("k").has_value());
}

OPDSL_TEST(PersistentCache, TwoWritersBothSucceedOneFileWins) {
  TempDir dir;
  CacheOptions o = in(dir);
  // Two processes store different payloads under one key, many times, at once.
  pid_t pids[2];
  for (int w = 0; w < 2; ++w) {
    pids[w] = fork();
    OPDSL_EXPECT(pids[w] >= 0);
    if (pids[w] == 0) {
      PersistentCache cache(o);
      int failures = 0;
      std::string payload(100000, w == 0 ? 'a' : 'b');
      for (int i = 0; i < 50; ++i)
        if (cache.store("shared", w == 0 ? "writer a" : "writer b", payload)) ++failures;
      _exit(failures == 0 ? 0 : 1);
    }
  }
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);
    OPDSL_EXPECT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  PersistentCache cache(o);
  auto hit = cache.lookup("shared");
  OPDSL_EXPECT(hit.has_value());
  if (hit) {
    // One of the two, whole.
    bool a = hit->header == "writer a" && hit->payload == std::string(100000, 'a');
    bool b = hit->header == "writer b" && hit->payload == std::string(100000, 'b');
    OPDSL_EXPECT(a || b);
  }
  // No temporary files are left behind.
  int tmp = 0;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(dir.path, ec), end; !ec && it != end; it.increment(ec))
    if (llvm::sys::path::filename(it->path()).starts_with(".tmp-")) ++tmp;
  OPDSL_EXPECT_EQ(tmp, 0);
}

OPDSL_TEST(PersistentCache, LeastRecentlyUsedEntriesAreEvicted) {
  TempDir dir;
  // Each entry is about 1000 bytes + header; the cap holds roughly four.
  PersistentCache cache(in(dir, 4600));
  std::string payload(1000, 'x');
  auto pause = [] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); }; // mtime granularity
  for (const char *k : {"k1", "k2", "k3", "k4"}) {
    OPDSL_EXPECT(!cache.store(k, "h", payload));
    pause();
  }
  // k1 is the oldest, but a hit makes it the newest.
  OPDSL_EXPECT(cache.lookup("k1").has_value());
  pause();
  // Two more entries push the total over the cap: k2 and k3 go first.
  OPDSL_EXPECT(!cache.store("k5", "h", payload));
  pause();
  OPDSL_EXPECT(!cache.store("k6", "h", payload));
  OPDSL_EXPECT(cache.stats().evicted >= 2);
  OPDSL_EXPECT(!llvm::sys::fs::exists(cache.pathOf("k2")));
  OPDSL_EXPECT(cache.lookup("k1").has_value());
  OPDSL_EXPECT(cache.lookup("k6").has_value());
  OPDSL_EXPECT(cache.sizeOnDisk() <= 4600);
}

OPDSL_TEST(PersistentCache, DefaultDirectoryFollowsTheEnvironment) {
  setenv("OPS_JIT_CACHE_DIR", "/tmp/opdsl-explicit", 1);
  OPDSL_EXPECT_EQ(PersistentCache::defaultDirectory(), "/tmp/opdsl-explicit");
  unsetenv("OPS_JIT_CACHE_DIR");
  setenv("XDG_CACHE_HOME", "/tmp/xdg-cache", 1);
  OPDSL_EXPECT_EQ(PersistentCache::defaultDirectory(), "/tmp/xdg-cache/opdsl-jit");
  unsetenv("XDG_CACHE_HOME");
  std::string d = PersistentCache::defaultDirectory();
  OPDSL_EXPECT(d.size() > 9 && d.substr(d.size() - 9) == "opdsl-jit");
}
