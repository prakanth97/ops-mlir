//===- kernel-ir-dump.cpp - CLI for KernelIRBuilder --------------*- C++ -*-===//
//
// Standalone tool that runs a single OPS kernel function through
// KernelIRBuilder and prints the resulting MLIR -- the same translation the
// CUDA backend performs internally (JITEngine::materializeGpuKernel) to turn
// a kernel's C++ body into real device code, since a GPU kernel can't be
// resolved by linking a compiled host symbol the way the CPU/OpenMP
// backends do. Useful for checking whether a given kernel is expressible in
// KernelIRBuilder's supported subset, and for inspecting exactly what it
// would compile to, without running the full app.
//
//===----------------------------------------------------------------------===//

#include "runtime/KernelIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>

using namespace llvm;

static cl::opt<std::string>
    SourceFile(cl::Positional, cl::Required,
              cl::desc("<kernel source/header file>"));
static cl::opt<std::string>
    KernelName(cl::Positional, cl::Required,
              cl::desc("<name of the kernel function to translate>"));
static cl::opt<int>
    IndexRank(cl::Positional, cl::Required,
             cl::desc("<idx array rank, e.g. 3 for a 3D block>"));

static cl::list<std::string> DoubleConsts(
    "double",
    cl::desc("Register an extern double the kernel references, as "
            "name=value (repeatable)"),
    cl::value_desc("name=value"));
static cl::list<std::string> IntConsts(
    "int",
    cl::desc("Register an extern int the kernel references, as name=value "
            "(repeatable)"),
    cl::value_desc("name=value"));

static cl::list<std::string> ConstDims(
    "const-dim",
    cl::desc("Register the array dimension for a `const double*` "
            "parameter, as name=dim (repeatable, required for kernels "
            "with const-array parameters)"),
    cl::value_desc("name=dim"));

static bool splitNameValue(const std::string &arg, std::string &name,
                           std::string &value) {
  auto pos = arg.find('=');
  if (pos == std::string::npos)
    return false;
  name = arg.substr(0, pos);
  value = arg.substr(pos + 1);
  return true;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "Dump the MLIR translation of a single OPS kernel (KernelIRBuilder) --\n"
      "the same translation the CUDA backend performs on kernel bodies.\n\n"
      "If the kernel references extern globals (e.g. Delta0block0, gama),\n"
      "register a value for each with -double/-int, or translation will\n"
      "fail with \"reference to unsupported symbol\". Values only affect\n"
      "constant-folding in the emitted IR, not translation shape, so any\n"
      "placeholder value works.\n\n"
      "Example:\n"
      "  kernel-ir-dump apps/c/taylor_green_vortex/opensbliblock00_kernels.h \\\n"
      "      opensbliblock00Kernel039 3 \\\n"
      "      -double Delta0block0=0.1 -double Delta1block0=0.1 \\\n"
      "      -double Delta2block0=0.1 -double Minf=0.1 -double gama=1.4\n");

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // KernelIRBuilder only stores addresses (mirroring how JITEngine bakes in
  // live app-side constants), so these must outlive the generate() call.
  std::vector<std::unique_ptr<double>> doubleStorage;
  std::vector<std::unique_ptr<int32_t>> intStorage;
  std::map<std::string, const void *> constants;

  for (const std::string &arg : DoubleConsts) {
    std::string name, value;
    if (!splitNameValue(arg, name, value)) {
      errs() << "kernel-ir-dump: -double expects name=value, got '" << arg
            << "'\n";
      return 1;
    }
    doubleStorage.push_back(std::make_unique<double>(std::stod(value)));
    constants[name] = doubleStorage.back().get();
  }
  for (const std::string &arg : IntConsts) {
    std::string name, value;
    if (!splitNameValue(arg, name, value)) {
      errs() << "kernel-ir-dump: -int expects name=value, got '" << arg
            << "'\n";
      return 1;
    }
    intStorage.push_back(std::make_unique<int32_t>(std::stoi(value)));
    constants[name] = intStorage.back().get();
  }
  std::vector<int> constArgDims;
  for (const std::string &arg : ConstDims)
    constArgDims.push_back(std::stoi(arg));
  

  ops_mlir::KernelIRBuilder builder(ctx);
  mlir::func::FuncOp fn =
      builder.generate(SourceFile, KernelName, IndexRank, constants, constArgDims, errs());
  if (!fn) {
    errs() << "kernel-ir-dump: failed to translate '" << KernelName.getValue()
          << "' from '" << SourceFile.getValue() << "'\n";
    return 1;
  }

  fn.print(outs());
  outs() << "\n";
  return 0;
}
