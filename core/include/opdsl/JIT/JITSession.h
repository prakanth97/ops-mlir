//===- JITSession.h - compile MLIR (LLVM dialect) to objects and run them -*- C++ -*-===//
//
// The backend half of a DSL's JIT, independent of any DSL:
//
//   compile()  an MLIR module in the LLVM dialect  ->  an object file (bytes you can cache on disk)
//   load()     an object file                       ->  a packed function you can call
//
// The two are separate because compiled code is cached by a key computed *before* the MLIR lowering pipeline runs:
// on a hit, the pipeline and compile() are skipped and the cached object goes straight to load().
//
// Compiled functions are called through a packed wrapper `void(void **args)`: args[i] points at the storage of
// argument i, and args[nargs] at the storage of the result(s) if there are any (several results: a struct). The wrapper is generated for the
// entry function, calling its C interface `_mlir_ciface_<entry>` (the descriptor-pointer ABI that MLIR's
// `llvm.emit_c_interface` produces) when that exists and the function itself otherwise.
//
// Every loaded object gets a JITDylib of its own, so independently compiled objects can define the same helper
// symbols.
//
//===----------------------------------------------------------------------===//

#ifndef OPDSL_JIT_JITSESSION_H
#define OPDSL_JIT_JITSESSION_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>
#include <vector>

namespace opdsl {

struct JITOptions {
  /// LLVM optimisation level of the generated code (0-3).
  unsigned optLevel = 3;
  /// Target CPU and feature string; empty = the host's.
  std::string cpu;
  std::string features;
  /// Allow the code generator to contract a*b+c into fused multiply-adds (what clang does within an expression by default;
  /// the results then differ in the last bit from unfused code).
  bool fpContract = false;
  /// Shared libraries whose symbols the compiled code may use (libomp, runtime shims), loaded into the process.
  std::vector<std::string> sharedLibraries;
};

class JITSession {
public:
  using PackedFn = void (*)(void **);

  static llvm::Expected<std::unique_ptr<JITSession>> create(const JITOptions &options = {});
  ~JITSession();

  /// Registers what compile() needs to translate to LLVM IR on `ctx` (builtin, LLVM, OpenMP dialects).
  static void registerTranslations(mlir::MLIRContext &ctx);

  /// `module` must be in the LLVM dialect. `entry` names the function that gets the packed wrapper.
  /// Optimises and emits an object for this session's target.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> compile(mlir::ModuleOp module, llvm::StringRef entry);

  /// Adds `object` (from compile() or a cache) to the session and returns the packed wrapper of `entry`.
  llvm::Expected<PackedFn> load(std::unique_ptr<llvm::MemoryBuffer> object, llvm::StringRef entry);

  /// "triple cpu +features": what cached objects depend on, for cache keys.
  const std::string &targetDescription() const;

  /// Name of the packed wrapper of `entry` in the object.
  static std::string packedName(llvm::StringRef entry) { return ("_opdsl_" + entry).str(); }

private:
  JITSession();
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace opdsl

#endif // OPDSL_JIT_JITSESSION_H
