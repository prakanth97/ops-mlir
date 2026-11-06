#include "opdsl/Testing/TestHarness.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"

int main(int argc, char **argv) {
  // A crash in a test prints a (symbolised, if llvm-symbolizer is on the PATH) stack trace.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  return opdsl::testing::runAll(argc, argv);
}
