# Add Runtime library

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

# Exactly one FIR provider is compiled in: the real importer when this build
# has flang, otherwise a stand-in that reports the missing support. Keeping
# the choice here rather than behind #ifdefs in JITEngine means the C++
# frontend is identical either way.
if(OPS_ENABLE_FORTRAN)
	set(OPS_FORTRAN_PROVIDER fortran/FortranKernelIRBuilder.cpp)
else()
	set(OPS_FORTRAN_PROVIDER fortran/FortranKernelIRBuilderUnavailable.cpp)
endif()

# c/KernelIRBuilder.cpp needs Clang's resource-dir (its own bundled
# stddef.h/stdarg.h/etc, resolved relative to the clang binary) since it
# builds ASTs via the library API directly, bypassing the driver logic
# that normally infers this.
file(GLOB CLANG_RESOURCE_DIR_CANDIDATES "${LLVM_BUILD_DIR}/lib/clang/*")
list(GET CLANG_RESOURCE_DIR_CANDIDATES 0 CLANG_RESOURCE_DIR)

# fortran/FortranKernelIRBuilder.cpp needs the same for flang: the intrinsic
# modules (iso_c_binding.mod, ...) that a kernel module may `use`. The flang
# driver finds them relative to itself, but run in-process the "driver" is
# the application, so ask flang where they are. They only exist once
# flang-rt is built; kernels that use no intrinsic module work without them.
set(OPS_FLANG_INTRINSIC_MODULE_DIR "")
if(OPS_ENABLE_FORTRAN)
	execute_process(
		COMMAND "${OPS_FLANG_ROOT}/bin/flang" -print-file-name=iso_c_binding.mod
		OUTPUT_VARIABLE _iso_c_binding_mod
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(EXISTS "${_iso_c_binding_mod}")
		cmake_path(GET _iso_c_binding_mod PARENT_PATH OPS_FLANG_INTRINSIC_MODULE_DIR)
	else()
		message(WARNING "flang's intrinsic modules were not found; Fortran "
			"kernels that USE one (e.g. iso_c_binding) will not translate. Build "
			"flang-rt (-DLLVM_ENABLE_RUNTIMES=flang-rt) if you need them.")
	endif()
endif()

#===============================================================================
# ops_add_runtime_library(<name> [FTN])
#===============================================================================
# The runtime is built twice when the Fortran frontend is enabled. The two
# variants are identical except for the OPS_FTN macro, which decides whether
# the OPS C API is declared `extern "C"`: OPS builds its Fortran core that
# way so ISO_C_BINDING can reach it, and JITEngine calls ops_exit() and
# ops_halo_transfer() directly, so the runtime's references have to be
# mangled the same way as the OPS core an application links. OPS itself
# ships separate `ops_seq` and `ops_for_seq` libraries for this reason.
#===============================================================================
function(ops_add_runtime_library name)
	cmake_parse_arguments(ARG "FTN" "" "" ${ARGN})

	add_library(
		${name}
		common/IRBuilder.cpp
		common/JITEngine.cpp
		c/KernelIRBuilder.cpp
		common/KernelProfiler.cpp
		fortran/FortranCapture.cpp
		${OPS_FORTRAN_PROVIDER}
	)

	target_include_directories(
		${name}
		PUBLIC
		${CMAKE_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
		PRIVATE
		${CLANG_INCLUDE_DIRS}
	)

	target_compile_definitions(
		${name}
		PRIVATE
		OPS_XDSL_DIR="${CMAKE_SOURCE_DIR}/xdsl_impl"
		OPS_CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}"
		OPS_FLANG_INTRINSIC_MODULE_DIR="${OPS_FLANG_INTRINSIC_MODULE_DIR}"
		$<$<BOOL:${OPS_ENABLE_CUDA}>:OPS_ENABLE_CUDA>
	)
	if(ARG_FTN)
		# PUBLIC: an application linking this variant must see the same
		# linkage for the OPS API as the runtime was compiled against.
		target_compile_definitions(${name} PUBLIC OPS_FTN)
	endif()

	# BackendPipeline.h (via common/JITEngine.cpp) defines custom mlir::Pass subclasses
	# -- this LLVM/MLIR build has LLVM_ENABLE_RTTI=OFF, so mlir::Pass has no
	# typeinfo emitted anywhere. Compiling this TU with the default (RTTI-on)
	# would try to reference that nonexistent typeinfo and fail to link (same
	# issue tools/kernel-ir-dump.cpp hit with llvm::cl::Option).
	target_compile_options(${name} PRIVATE -fno-rtti)

	if(OPS_ENABLE_CUDA)
		target_link_libraries(${name} PRIVATE CUDA::cuda_driver)
	endif()

	# FIR headers live in the flang source tree; its TableGen'd definitions
	# land in the LLVM build tree under tools/flang.
	if(OPS_ENABLE_FORTRAN)
		cmake_path(GET LLVM_SOURCE_DIR PARENT_PATH OPS_LLVM_PROJECT_DIR)
		target_include_directories(${name} PRIVATE
			"${OPS_LLVM_PROJECT_DIR}/flang/include"
			"${OPS_FLANG_ROOT}/tools/flang/include")
		# flangFrontend is the in-process compiler, the counterpart of
		# clangTooling below; it pulls in parsing, semantics and lowering.
		# CUFAttrs is not optional despite ops-mlir never touching CUDA
		# Fortran: several core FIR ops (fir.global, fir.declare) carry a
		# cuf.data_attr attribute, so FIRDialect's object files reference it
		# unconditionally.
		target_link_libraries(${name} PUBLIC
			flangFrontend
			FIRDialect
			FIRDialectSupport
			FIRSupport
			CUFAttrs
			FortranSupport)
	endif()

	target_link_libraries(
		${name}
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
endfunction()

ops_add_runtime_library(OPSRuntime)

if(OPS_ENABLE_FORTRAN)
	ops_add_runtime_library(OPSRuntimeFtn FTN)
endif()
