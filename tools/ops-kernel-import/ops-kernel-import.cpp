//===- ops-kernel-import.cpp - build-time kernel importer (provider A) -----------===//
//
//   ops-kernel-import --kernel K (--source app.cpp | --header k.h) --ndim N
//                     [--kinds acc,acc,gbl,red,idx] -o out.mlir|out.mlirbc --manifest out.json
//                     [-- <clang flags: -I..., -D...>]
//
// Imports the OPS kernel `K`. The output is an MLIR module with `ops.kernel @K` (text for *.mlir, bytecode for
// anything else) and the manifest. A kernel that cannot be imported is not an error of the build: the manifest then
// says status "failed" with the reason, no module is written, and the exit code is 0 (use --strict for 1).
//
//===----------------------------------------------------------------------===//

#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Frontend/KernelImport.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

static cl::opt<std::string> kernel("kernel", cl::desc("name of the kernel function"), cl::Required);
static cl::opt<std::string> source("source", cl::desc("application source file that defines or includes the kernel"));
static cl::opt<std::string> header("header", cl::desc("header that defines the kernel (parsed after ops_lib_core.h)"));
static cl::opt<int> ndim("ndim", cl::desc("dimension of the block (1-3)"), cl::Required);
static cl::opt<std::string> kinds("kinds", cl::desc("comma-separated parameter kinds: acc,gbl,red,idx"));
static cl::opt<std::string> output("o", cl::desc("output module (.mlir: text, otherwise bytecode)"));
static cl::opt<std::string> manifestPath("manifest", cl::desc("output manifest (JSON)"));
static cl::opt<std::string> provider("provider", cl::desc("importer provider"), cl::init("A"));
static cl::opt<std::string> resourceDir("resource-dir", cl::desc("clang resource directory"));
static cl::opt<std::string> argsFile("args-file", cl::desc("file with more clang arguments, separated by blanks and newlines"));
static cl::opt<std::string> depFile("depfile", cl::desc("write the files the translation unit read, as a Makefile dependency rule"));
static cl::opt<bool> strict("strict", cl::desc("exit with 1 when the kernel cannot be imported"));

int main(int argc, char **argv) {
  // Everything after a bare "--" is for clang.
  std::vector<std::string> clangArgs;
  int ownArgc = argc;
  for (int i = 1; i < argc; ++i)
    if (StringRef(argv[i]) == "--") {
      ownArgc = i;
      for (int k = i + 1; k < argc; ++k) clangArgs.push_back(argv[k]);
      break;
    }
  cl::ParseCommandLineOptions(ownArgc, argv, "OPS kernel importer\n");
  if (provider != "A") {
    errs() << "ops-kernel-import: only provider A exists so far\n";
    return 2;
  }

  mlir::DialectRegistry registry;
  registry.insert<ops_mlir::ops::OPSDialect, mlir::arith::ArithDialect, mlir::math::MathDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  ops_mlir::frontend::ImportOptions options;
  options.kernelName = kernel;
  options.sourceFile = source;
  options.headerFile = header;
  options.ndim = ndim;
  options.resourceDir = resourceDir;
  for (StringRef rest(kinds); !rest.empty();) {
    auto [first, tail] = rest.split(',');
    options.kinds.push_back(first.str());
    rest = tail;
  }
  if (!argsFile.empty()) {
    auto buffer = MemoryBuffer::getFile(argsFile);
    if (!buffer) {
      errs() << "ops-kernel-import: cannot read " << argsFile << "\n";
      return 2;
    }
    std::string word;
    for (char c : (*buffer)->getBuffer()) {
      if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
        if (!word.empty()) options.clangArgs.push_back(word);
        word.clear();
      } else {
        word += c;
      }
    }
    if (!word.empty()) options.clangArgs.push_back(word);
  }
  for (const std::string &a : clangArgs) options.clangArgs.push_back(a);

  ops_mlir::frontend::ImportResult result = ops_mlir::frontend::importKernel(ctx, options);
  for (const std::string &w : result.warnings) errs() << "ops-kernel-import: warning: " << kernel << ": " << w << "\n";
  if (!result.ok) errs() << "ops-kernel-import: " << kernel << ": not imported: " << result.reason << "\n";

  if (!depFile.empty()) {
    std::string target = (!output.empty() && output != "-") ? std::string(output) : std::string(manifestPath);
    std::error_code ec;
    raw_fd_ostream os(depFile, ec, sys::fs::OF_Text);
    if (ec) {
      errs() << "ops-kernel-import: cannot write " << depFile << ": " << ec.message() << "\n";
      return 2;
    }
    auto escaped = [](StringRef s) {
      std::string out;
      for (char c : s) out += (c == ' ' ? "\\ " : std::string(1, c));
      return out;
    };
    os << escaped(target) << ":";
    for (const std::string &d : result.dependencies) os << " " << escaped(d);
    os << "\n";
  }

  if (!manifestPath.empty()) {
    std::error_code ec;
    raw_fd_ostream os(manifestPath, ec, sys::fs::OF_Text);
    if (ec) {
      errs() << "ops-kernel-import: cannot write " << manifestPath << ": " << ec.message() << "\n";
      return 2;
    }
    os << result.manifest;
  }
  if (result.ok && !output.empty()) {
    std::error_code ec;
    bool text = output == "-" || StringRef(output).ends_with(".mlir");
    raw_fd_ostream os(output, ec, text ? sys::fs::OF_Text : sys::fs::OF_None);
    if (ec) {
      errs() << "ops-kernel-import: cannot write " << output << ": " << ec.message() << "\n";
      return 2;
    }
    if (text) result.module->print(os);
    else if (mlir::failed(mlir::writeBytecodeToFile(result.module.get(), os))) {
      errs() << "ops-kernel-import: cannot write the bytecode\n";
      return 2;
    }
  }
  return result.ok || !strict ? 0 : 1;
}
