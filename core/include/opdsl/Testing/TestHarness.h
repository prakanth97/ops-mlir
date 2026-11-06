//===- TestHarness.h - minimal unit-test harness -----------------*- C++ -*-===//
//
// The pinned LLVM install has no gtest, and the tests here need very little:
//
//   OPDSL_TEST(Suite, Name) { OPDSL_EXPECT(x == 1); OPDSL_EXPECT_EQ(a, b); }
//
// A test executable runs every registered test, or those whose "Suite.Name" contains the first
// command-line argument. A failed expectation is reported with file and line and fails the test run.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_TESTING_TESTHARNESS_H
#define OPDSL_TESTING_TESTHARNESS_H

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace opdsl::testing {

struct TestCase {
  const char *suite;
  const char *name;
  void (*fn)();
};

inline std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int &failureCount() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char *suite, const char *name, void (*fn)()) { registry().push_back({suite, name, fn}); }
};

inline void reportFailure(const char *file, int line, const std::string &what) {
  failureCount()++;
  std::fprintf(stderr, "%s:%d: expectation failed: %s\n", file, line, what.c_str());
}

template <typename A, typename B>
void expectEq(const A &a, const B &b, const char *as, const char *bs, const char *file, int line) {
  if (a == b) return;
  std::ostringstream os;
  os << as << " == " << bs << "  (" << a << " vs " << b << ")";
  reportFailure(file, line, os.str());
}

/// Runs the registered tests; returns the process exit code.
inline int runAll(int argc, char **argv) {
  const char *filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0, failedTests = 0;
  for (const TestCase &t : registry()) {
    std::string full = std::string(t.suite) + "." + t.name;
    if (filter && full.find(filter) == std::string::npos) continue;
    int before = failureCount();
    std::printf("[ RUN  ] %s\n", full.c_str());
    t.fn();
    ++ran;
    if (failureCount() != before) {
      ++failedTests;
      std::printf("[ FAIL ] %s\n", full.c_str());
    } else {
      std::printf("[  OK  ] %s\n", full.c_str());
    }
  }
  std::printf("%d tests run, %d failed\n", ran, failedTests);
  if (ran == 0) {
    std::fprintf(stderr, "no tests matched\n");
    return 1;
  }
  return failedTests == 0 ? 0 : 1;
}

} // namespace opdsl::testing

#define OPDSL_TEST(suite, name)                                                                    \
  static void opdsl_test_##suite##_##name();                                                       \
  static ::opdsl::testing::Registrar opdsl_registrar_##suite##_##name(#suite, #name,               \
                                                                      opdsl_test_##suite##_##name); \
  static void opdsl_test_##suite##_##name()

#define OPDSL_EXPECT(cond)                                                                         \
  do {                                                                                             \
    if (!(cond)) ::opdsl::testing::reportFailure(__FILE__, __LINE__, #cond);                       \
  } while (0)

#define OPDSL_EXPECT_EQ(a, b) ::opdsl::testing::expectEq((a), (b), #a, #b, __FILE__, __LINE__)

#endif // OPDSL_TESTING_TESTHARNESS_H
