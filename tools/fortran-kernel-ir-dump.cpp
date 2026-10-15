//===- fortran-kernel-ir-dump.cpp - CLI for FortranKernelIRBuilder -*- C++ -*-===//
//
// Standalone tool that translates a single OPS kernel out of a Fortran
// kernel module and prints the resulting MLIR -- the same translation
// JITEngine::materializeKernelBody performs for a Fortran application.
//
// The Fortran counterpart of kernel-ir-dump. Useful for checking whether a
// kernel is expressible in the supported subset, and for seeing exactly
// what it lowers to, without building or running the full app.
//
//===----------------------------------------------------------------------===//

#include "runtime/fortran/FortranKernelIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt<std::string>
    KernelSource(cl::Positional, cl::Required,
                 cl::desc("<Fortran kernel module source>"));
static cl::opt<std::string>
    KernelName(cl::Positional, cl::Required,
               cl::desc("<name of the kernel procedure to import>"));
static cl::opt<int>
    IndexRank(cl::Positional, cl::Required,
              cl::desc("<block dimensionality, e.g. 2 for a 2D block>"));

static cl::list<std::string> Dependencies(
    "dep",
    cl::desc("A module the kernel module uses, compiled first (repeatable, "
             "in dependency order)"),
    cl::value_desc("source"));

static cl::opt<int> ScalarArgs(
    "scalars", cl::init(0),
    cl::desc("Number of f64 stencil/global arguments the loop passes"));
static cl::opt<bool> HasIdx(
    "idx", cl::init(false),
    cl::desc("The loop passes an ops_arg_idx argument"));
static cl::opt<bool> NoResult(
    "no-result", cl::init(false),
    cl::desc("The kernel is a subroutine (writes through an out argument)"));

static cl::list<std::string> DoubleConsts(
    "double",
    cl::desc("Register a real(8) module constant the kernel reads, as "
             "name=value (repeatable)"),
    cl::value_desc("name=value"));
static cl::list<std::string> IntConsts(
    "int",
    cl::desc("Register an integer(4) module constant the kernel reads, as "
             "name=value (repeatable)"),
    cl::value_desc("name=value"));

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
      "Dump the MLIR translation of a single OPS Fortran kernel -- the same\n"
      "translation the JIT performs on kernel bodies.\n\n"
      "The kernel ABI is the one the stencil lowering emits: -scalars gives\n"
      "the number of f64 arguments and -idx adds the trailing index memref.\n"
      "Module constants the kernel reads must be registered with\n"
      "-double/-int unless they are Fortran `parameter`s, whose values flang\n"
      "folds itself. Values only affect constant folding in the emitted IR,\n"
      "so any placeholder works.\n\n"
      "Example:\n"
      "  fortran-kernel-ir-dump apps/fortran/laplace_2d/laplace_kernels.F90 \\\n"
      "      -dep apps/fortran/laplace_2d/laplace_constants.F90 \\\n"
      "      left_bndcon 2 -idx -int jmax=4094\n");

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // FortranKernelIRBuilder only stores addresses (mirroring how JITEngine
  // bakes in live app-side constants), so these must outlive generate().
  std::vector<std::unique_ptr<double>> doubleStorage;
  std::vector<std::unique_ptr<int32_t>> intStorage;
  std::map<std::string, const void *> constants;

  for (const std::string &arg : DoubleConsts) {
    std::string name, value;
    if (!splitNameValue(arg, name, value)) {
      errs() << "fortran-kernel-ir-dump: -double expects name=value, got '"
             << arg << "'\n";
      return 1;
    }
    doubleStorage.push_back(std::make_unique<double>(std::stod(value)));
    constants[name] = doubleStorage.back().get();
  }
  for (const std::string &arg : IntConsts) {
    std::string name, value;
    if (!splitNameValue(arg, name, value)) {
      errs() << "fortran-kernel-ir-dump: -int expects name=value, got '" << arg
             << "'\n";
      return 1;
    }
    intStorage.push_back(std::make_unique<int32_t>(std::stoi(value)));
    constants[name] = intStorage.back().get();
  }

  // Reconstruct the declaration the stencil lowering would have left behind:
  // the f64 arguments first, then the index memref, matching declare_kernel()
  // in xdsl_impl/ops_to_stencil.py.
  mlir::OpBuilder builder(&ctx);
  std::vector<mlir::Type> inputs(ScalarArgs, builder.getF64Type());
  if (HasIdx)
    inputs.push_back(mlir::MemRefType::get({IndexRank}, builder.getI32Type()));
  std::vector<mlir::Type> results;
  if (!NoResult)
    results.push_back(builder.getF64Type());
  mlir::FunctionType opsABI = builder.getFunctionType(inputs, results);

  ops_mlir::FortranKernelIRBuilder firBuilder(ctx);
  ops_mlir::FortranKernelSource source{KernelSource, {}};
  source.dependencies.assign(Dependencies.begin(), Dependencies.end());
  mlir::func::FuncOp fn = firBuilder.generate(source, KernelName, IndexRank,
                                              opsABI, constants, errs());
  if (!fn) {
    errs() << "fortran-kernel-ir-dump: failed to translate '"
           << KernelName.getValue() << "' from '" << KernelSource.getValue()
           << "'\n";
    return 1;
  }

  fn.print(outs());
  outs() << "\n";
  return 0;
}
