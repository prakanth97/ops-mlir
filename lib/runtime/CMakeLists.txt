# Add Runtime library

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

add_library(
	OPSRuntime
	IRBuilder.cpp
	JITEngine.cpp
	KernelIRBuilder.cpp
	KernelProfiler.cpp
)

target_include_directories(
	OPSRuntime
	PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
	PRIVATE
	${CLANG_INCLUDE_DIRS}
)

# KernelIRBuilder.cpp needs Clang's resource-dir (its own bundled
# stddef.h/stdarg.h/etc, resolved relative to the clang binary) since it
# builds ASTs via the library API directly, bypassing the driver logic
# that normally infers this.
file(GLOB CLANG_RESOURCE_DIR_CANDIDATES "${LLVM_BUILD_DIR}/lib/clang/*")
list(GET CLANG_RESOURCE_DIR_CANDIDATES 0 CLANG_RESOURCE_DIR)

target_compile_definitions(
	OPSRuntime
	PRIVATE
	OPS_XDSL_DIR="${CMAKE_SOURCE_DIR}/xdsl_impl"
	OPS_CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}"
	$<$<BOOL:${OPS_ENABLE_CUDA}>:OPS_ENABLE_CUDA>
	$<$<BOOL:${OPS_ENABLE_ROCM}>:OPS_ENABLE_ROCM>
)

# BackendPipeline.h (via JITEngine.cpp) defines custom mlir::Pass subclasses
# -- this LLVM/MLIR build has LLVM_ENABLE_RTTI=OFF, so mlir::Pass has no
# typeinfo emitted anywhere. Compiling this TU with the default (RTTI-on)
# would try to reference that nonexistent typeinfo and fail to link (same
# issue tools/kernel-ir-dump.cpp hit with llvm::cl::Option).
target_compile_options(OPSRuntime PRIVATE -fno-rtti)

if(OPS_ENABLE_CUDA)
  target_link_libraries(OPSRuntime PRIVATE CUDA::cuda_driver)
endif()

if(OPS_ENABLE_ROCM)
  target_link_libraries(OPSRuntime PRIVATE hip::host)
endif()

target_link_libraries(
	OPSRuntime
	PUBLIC
	MLIRIR
	MLIRDialectUtils
	MLIRSupport
	MLIRPass
	MLIROPSDialect
	MLIRExecutionEngine
	MLIRTargetLLVMIRExport
	MLIRToLLVMIRTranslationRegistration
	MLIRVectorToLLVM
	MLIRRegisterAllExtensions
	OPSPasses
	Python3::Python
	clangTooling
)
