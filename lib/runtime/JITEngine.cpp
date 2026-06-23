#include "runtime/JITEngine.h"

namespace ops_mlir {

JITEngine &JITEngine::instance() {
  static JITEngine rt;
  return rt;
}

void JITEngine::setFlushCallback(FlushCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  flushCallback_ = std::move(callback);
}

void JITEngine::enqueueParLoop(std::uintptr_t kernelToken,
                                    const char *kernelName, ops_block block,
                                    int dims, const int *range,
                                    const ops_arg *args, std::size_t nargs) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(
      buildLoopDesc(kernelToken, kernelName, block, dims, range, args, nargs));
}

void JITEngine::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (flushCallback_ && !queue_.empty()) {
    flushCallback_(queue_);
  }
  queue_.clear();
}

LoopDesc JITEngine::buildLoopDesc(std::uintptr_t kernelToken,
                                       const char *kernelName, ops_block block,
                                       int dims, const int *range,
                                       const ops_arg *args, std::size_t nargs) {
  LoopDesc loop;
  loop.kernel_name = kernelName;
  loop.kernel_token = kernelToken;
  loop.block = reinterpret_cast<std::uintptr_t>(block);
  loop.dims = dims;

  loop.range.assign(range, range + 2 * dims);

  loop.args.reserve(nargs);
  for (std::size_t i = 0; i < nargs; ++i) {
    loop.args.push_back(buildArgDesc(args[i]));
  }

  return loop;
}

ArgDesc JITEngine::buildArgDesc(const ops_arg &arg) {
  ArgDesc desc;
  desc.dim = arg.dim;
  desc.elem_size = arg.elem_size;
  desc.data = reinterpret_cast<std::uintptr_t>(arg.data);
  desc.data_d = reinterpret_cast<std::uintptr_t>(arg.data_d);
  desc.acc = arg.acc;
  desc.argtype = arg.argtype;
  desc.opt = arg.opt;

  if (arg.argtype == OPS_ARG_DAT) {
    if (arg.dat)
      desc.dat = describeDat(arg.dat);
    if (arg.stencil)
      desc.stencil = describeStencil(arg.stencil);
  }

  return desc;
}

DatDesc JITEngine::describeDat(ops_dat dat) {
  DatDesc d;
  d.handle = reinterpret_cast<std::uintptr_t>(dat); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  d.index = dat->index;
  d.block = reinterpret_cast<std::uintptr_t>(dat->block); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  
  d.dim = dat->dim;
  d.type_size = dat->type_size;
  d.elem_size = dat->elem_size;

  int ndim = dat->block ? dat->block->dims : 1;
  for (int i = 0; i < ndim; ++i) {
    d.size.push_back(dat->size[i]);
    d.base.push_back(dat->base[i]);
    d.d_m.push_back(dat->d_m[i]);
    d.d_p.push_back(dat->d_p[i]);
    d.stride.push_back(dat->stride[i]);
  }

  d.name = dat->name ? dat->name : "";
  d.type = dat->type ? dat->type : "";

  d.data = reinterpret_cast<std::uintptr_t>(dat->data);
  d.data_d = reinterpret_cast<std::uintptr_t>(dat->data_d);

  return d;
}

StencilDesc JITEngine::describeStencil(ops_stencil stencil) {
  StencilDesc s;
  s.index = stencil->index;
  s.dims = stencil->dims;
  s.points = stencil->points;
  s.name = stencil->name ? stencil->name : "";

  s.stencil = reinterpret_cast<std::uintptr_t>(stencil->stencil);
  s.stride = reinterpret_cast<std::uintptr_t>(stencil->stride);
  s.mgrid_stride = reinterpret_cast<std::uintptr_t>(stencil->mgrid_stride);
  s.type = stencil->type;

  return s;
}

void JITEngine::compile() {
  module = builder.buildModule(queue_);

  std::cout << "=== OPS.PAR_LOOP MLIR IR ===\n\n";
  std::string ir = builder.moduleToString(module);
  std::cout << ir << "\n";
}

void JITEngine::execute() {
  // Placeholder for future execution logic
  std::cout << "Executing compiled module...\n";
}

const char *accessToString(int access) {
  switch (access) {
  case OPS_READ:
    return "READ";
  case OPS_WRITE:
    return "WRITE";
  case OPS_RW:
    return "RW";
  case OPS_INC:
    return "INC";
  case OPS_MIN:
    return "MIN";
  case OPS_MAX:
    return "MAX";
  default:
    return "UNKNOWN";
  }
}

const char *argKindToString(ArgKind kind) {
  switch (kind) {
  case ArgKind::Dat:
    return "Dat";
  case ArgKind::Gbl:
    return "Gbl";
  case ArgKind::Idx:
    return "Idx";
  case ArgKind::Reduce:
    return "Reduce";
  default:
    return "Unknown";
  }
}

} // namespace ops_mlir
