#include "opdsl/JIT/JITSession.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

namespace opdsl {

namespace {

llvm::Error fail(const llvm::Twine &message) { return llvm::createStringError(llvm::errc::invalid_argument, message); }

void initializeLLVMOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

/// Defines `_opdsl_<entry>(void **args)`, calling `_mlir_ciface_<entry>` if the module has it and `<entry>` otherwise.
///
/// A function with several results returns them as one LLVM struct; the C interface of such a function takes a pointer
/// to the struct as an extra first argument (and returns void). The packed wrapper hides that: like for a single
/// result, args[nargs] points at the storage for the results, laid out as the struct (natural alignment).
bool definePackedWrapper(llvm::Module &module, llvm::StringRef entry) {
  llvm::Function *plain = module.getFunction(entry);
  llvm::Function *callee = module.getFunction(("_mlir_ciface_" + entry).str());
  const bool ciface = callee && !callee->isDeclaration();
  if (!ciface) callee = plain;
  if (!callee || callee->isDeclaration()) return false;
  const bool resultThroughPointer = ciface && plain && !plain->isDeclaration() && plain->getReturnType()->isStructTy();

  llvm::LLVMContext &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);
  auto *type = llvm::FunctionType::get(builder.getVoidTy(), builder.getPtrTy(), /*isVarArg=*/false);
  auto *wrapper = llvm::cast<llvm::Function>(module.getOrInsertFunction(JITSession::packedName(entry), type).getCallee());
  if (!wrapper->empty()) return true;

  auto *block = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  builder.SetInsertPoint(block);
  llvm::Value *argList = wrapper->arg_begin();
  auto loadSlot = [&](uint64_t index) {
    llvm::Value *slot = builder.CreateGEP(builder.getPtrTy(), argList, builder.getInt64(index));
    return builder.CreateLoad(builder.getPtrTy(), slot);
  };
  llvm::SmallVector<llvm::Value *, 8> args;
  const uint64_t numArgs = callee->arg_size() - (resultThroughPointer ? 1 : 0);
  if (resultThroughPointer) args.push_back(loadSlot(numArgs));
  for (auto [index, arg] : llvm::enumerate(callee->args())) {
    if (resultThroughPointer && index == 0) continue;
    uint64_t position = resultThroughPointer ? index - 1 : index;
    args.push_back(builder.CreateLoad(arg.getType(), loadSlot(position)));
  }
  llvm::Value *result = builder.CreateCall(callee, args);
  if (!result->getType()->isVoidTy()) builder.CreateStore(result, loadSlot(numArgs));
  builder.CreateRetVoid();
  llvm::appendToCompilerUsed(module, {wrapper});
  return true;
}

/// ORC does not run the static constructors of object files without a platform (and there is none here), but compiled
/// code needs them: the GPU module load that gpu.binary becomes is a global constructor. Replace `llvm.global_ctors` /
/// `llvm.global_dtors` by exported functions `name` that call the entries in order (destructors: reverse order); the
/// session runs them after loading and before unloading.
void wrapGlobalFunctions(llvm::Module &module, const char *globalName, const std::string &name, bool reverse) {
  llvm::GlobalVariable *global = module.getGlobalVariable(globalName);
  if (!global || !global->hasInitializer()) return;
  auto *entries = llvm::dyn_cast<llvm::ConstantArray>(global->getInitializer());
  if (!entries) return;
  std::vector<std::pair<uint64_t, llvm::Function *>> list;
  for (llvm::Value *entry : entries->operands()) {
    auto *pair = llvm::cast<llvm::ConstantStruct>(entry);
    auto *priority = llvm::cast<llvm::ConstantInt>(pair->getOperand(0));
    if (auto *fn = llvm::dyn_cast<llvm::Function>(pair->getOperand(1)->stripPointerCasts())) list.push_back({priority->getZExtValue(), fn});
  }
  std::stable_sort(list.begin(), list.end(), [&](auto &a, auto &b) { return reverse ? a.first > b.first : a.first < b.first; });
  llvm::LLVMContext &ctx = module.getContext();
  auto *type = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
  auto *fn = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, name, module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(ctx, "entry", fn));
  for (auto &[priority, callee] : list) builder.CreateCall(callee);
  builder.CreateRetVoid();
  llvm::appendToCompilerUsed(module, {fn});
  global->eraseFromParent();
}

} // namespace

struct JITSession::Impl {
  JITOptions options;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::string target;
  std::atomic<unsigned> nextDylib{0};
  std::vector<llvm::orc::JITDylib *> dylibs;
  std::vector<void (*)()> finalisers; ///< the destructors of the loaded objects, in order of loading
};

JITSession::JITSession() : impl(std::make_unique<Impl>()) {}
JITSession::~JITSession() {
  if (impl && impl->jit) {
    // Run the destructors of the objects (GPU module unloading...), newest first.
    for (auto it = impl->finalisers.rbegin(); it != impl->finalisers.rend(); ++it) (*it)();
    for (auto it = impl->dylibs.rbegin(); it != impl->dylibs.rend(); ++it) llvm::consumeError(impl->jit->deinitialize(**it));
  }
}

void JITSession::registerTranslations(mlir::MLIRContext &ctx) {
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerOpenMPDialectTranslation(ctx);
  // gpu.launch_func and gpu.binary become calls of the mgpu* runtime functions at this step.
  mlir::DialectRegistry gpu;
  mlir::registerGPUDialectTranslation(gpu);
  mlir::registerNVVMDialectTranslation(gpu);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(gpu);
  ctx.appendDialectRegistry(gpu);
}

llvm::Expected<std::unique_ptr<JITSession>> JITSession::create(const JITOptions &options) {
  initializeLLVMOnce();
  std::unique_ptr<JITSession> session(new JITSession());
  Impl &impl = *session->impl;
  impl.options = options;

  for (const std::string &lib : options.sharedLibraries) {
    std::string error;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(lib.c_str(), &error)) return fail("cannot load '" + lib + "': " + error);
  }

  auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!jtmb) return jtmb.takeError();
  if (!options.cpu.empty()) jtmb->setCPU(options.cpu);
  if (!options.features.empty()) jtmb->addFeatures(std::vector<std::string>{options.features});
  jtmb->setCodeGenOptLevel(static_cast<llvm::CodeGenOptLevel>(options.optLevel));
  jtmb->setRelocationModel(llvm::Reloc::PIC_);
  if (options.fpContract) jtmb->getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;

  auto tm = jtmb->createTargetMachine();
  if (!tm) return tm.takeError();
  impl.targetMachine = std::move(*tm);
  impl.target = impl.targetMachine->getTargetTriple().str() + " " + impl.targetMachine->getTargetCPU().str() + " " +
                impl.targetMachine->getTargetFeatureString().str() + (options.fpContract ? " fma=fast" : "");

  auto jit = llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(*jtmb)).create();
  if (!jit) return jit.takeError();
  impl.jit = std::move(*jit);
  return session;
}

const std::string &JITSession::targetDescription() const { return impl->target; }

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> JITSession::compile(mlir::ModuleOp module, llvm::StringRef entry) {
  registerTranslations(*module.getContext());
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) return fail("cannot translate the module to LLVM IR (is it fully lowered to the LLVM dialect?)");
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), impl->targetMachine.get());

  if (!definePackedWrapper(*llvmModule, entry)) return fail("the module does not define a function named '" + entry.str() + "'");
  wrapGlobalFunctions(*llvmModule, "llvm.global_ctors", "_opdsl_init_" + entry.str(), /*reverse=*/false);
  wrapGlobalFunctions(*llvmModule, "llvm.global_dtors", "_opdsl_fini_" + entry.str(), /*reverse=*/true);

  auto optimize = mlir::makeOptimizingTransformer(impl->options.optLevel, /*sizeLevel=*/0, impl->targetMachine.get());
  if (llvm::Error e = optimize(llvmModule.get())) return std::move(e);

  llvm::SmallVector<char, 0> buffer;
  {
    llvm::raw_svector_ostream stream(buffer);
    llvm::legacy::PassManager codegen;
    if (impl->targetMachine->addPassesToEmitFile(codegen, stream, nullptr, llvm::CodeGenFileType::ObjectFile))
      return fail("the target cannot emit object files");
    codegen.run(*llvmModule);
  }
  return llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(buffer.data(), buffer.size()), "opdsl-object");
}

llvm::Expected<JITSession::PackedFn> JITSession::load(std::unique_ptr<llvm::MemoryBuffer> object, llvm::StringRef entry) {
  llvm::orc::LLJIT &jit = *impl->jit;
  llvm::orc::JITDylib &jd = jit.getExecutionSession().createBareJITDylib("opdsl-object-" + std::to_string(impl->nextDylib++));
  // The compiled code finds its symbols (libm, the OpenMP runtime, shims) in the process and in the main dylib.
  auto generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit.getDataLayout().getGlobalPrefix());
  if (!generator) return generator.takeError();
  jd.addGenerator(std::move(*generator));
  if (llvm::Error e = jit.addObjectFile(jd, std::move(object))) return std::move(e);
  if (llvm::Error e = jit.initialize(jd)) return std::move(e);
  impl->dylibs.push_back(&jd);
  // Static constructors (see wrapGlobalFunctions), then the packed wrapper.
  if (auto init = jit.lookup(jd, ("_opdsl_init_" + entry).str())) init->toPtr<void (*)()>()();
  else llvm::consumeError(init.takeError());
  if (auto fini = jit.lookup(jd, ("_opdsl_fini_" + entry).str())) impl->finalisers.push_back(fini->toPtr<void (*)()>());
  else llvm::consumeError(fini.takeError());
  auto symbol = jit.lookup(jd, packedName(entry));
  if (!symbol) return symbol.takeError();
  return symbol->toPtr<PackedFn>();
}

} // namespace opdsl
