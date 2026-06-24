# Add Runtime library

add_library(
	OPSRuntime
	IRBuilder.cpp
	JITEngine.cpp
)

target_include_directories(
	OPSRuntime
	PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
)

target_link_libraries(
	OPSRuntime
	PUBLIC
	MLIRIR
	MLIRDialectUtils
	MLIRSupport
	MLIRPass
	MLIROPSDialect
	MLIRStencilDialect
	MLIROPSPasses
	MLIRFuncDialect
)
