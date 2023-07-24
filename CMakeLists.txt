cmake_minimum_required(VERSION 3.12)

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(nntoolchain LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(CCACHE ccache)
if(CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif()

include(FetchContent)

find_package(PythonInterp REQUIRED)

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})
message(STATUS "Using openssl ${OPENSSL_VERSION}")

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIR})
message(STATUS "Using zlib ${ZLIB_VERSION}")

find_package(pybind11 CONFIG REQUIRED)

#-------------------------------------------------------------------------------
# oneDNN Configuration
#-------------------------------------------------------------------------------
find_package(DNNL REQUIRED)
include_directories(${DNNL_INCLUDE_DIRS})
link_libraries(${DNNL_LIBRARY_DIRS})
message(STATUS "Using oneDNN in: ${DNNL_DIR}")
#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
FetchContent_Declare(cnpy
	GIT_REPOSITORY https://github.com/rogersce/cnpy
	GIT_TAG 4e8810b1a8637695171ed346ce68f6984e585ef4
	GIT_SHALLOW true
	PATCH_COMMAND patch -f -Np1 -i ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy.patch || true
	UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(cnpy)
include_directories(${cnpy_SOURCE_DIR})
#-------------------------------------------------------------------------------
# PROGRESSBAR Configuration
#-------------------------------------------------------------------------------
FetchContent_Declare(progressbar
	GIT_REPOSITORY https://github.com/gipert/progressbar
	GIT_TAG a67ccda1712d572e24cdfd02938e6a782e779787
	GIT_SHALLOW true
	UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(progressbar)
include_directories(${progressbar_SOURCE_DIR}/include)
#-------------------------------------------------------------------------------
# flatbuffers Configuration
#-------------------------------------------------------------------------------
find_program(FLATBUFFERS_CMD flatc REQUIRED)
find_package(FLATBUFFERS REQUIRED)
include_directories(${FLATBUFFERS_INCLUDE_DIRS})
link_libraries(${FLATBUFFERS_LIBRARY_DIRS})
message(STATUS "Using flatbuffer in: ${FLATBUFFERS_DIR}")
#-------------------------------------------------------------------------------
# nntoolchain Configuration
#-------------------------------------------------------------------------------
set(NNTOOLCHAIN_PATH ${PROJECT_SOURCE_DIR}/third_party/nntoolchain)
include_directories(${NNTOOLCHAIN_PATH}/include)
link_directories(${NNTOOLCHAIN_PATH}/lib)
#-------------------------------------------------------------------------------
# CV18xx Configuration
#-------------------------------------------------------------------------------
set(CV18XX_PATH ${PROJECT_SOURCE_DIR}/third_party/CV18xx)
include_directories(${CV18XX_PATH}/include)
link_directories(${CV18XX_PATH}/lib)
#-------------------------------------------------------------------------------
# caffe Configuration
#-------------------------------------------------------------------------------
set(CPU_ONLY ON)
set(USE_HDF5 OFF)
set(OBUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(caffe
	GIT_REPOSITORY https://github.com/sophgo/caffe
	GIT_TAG e188c0ee5491fbd4c5dc9ea35248b8580a65f4f8
	GIT_SHALLOW true
	PATCH_COMMAND patch -f -Np1 -i ${CMAKE_CURRENT_SOURCE_DIR}/third_party/caffe.patch || true
	UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(caffe)
set(BUILD_SHARED_LIBS ${OBUILD_SHARED_LIBS})
#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
find_package(LLVM CONFIG REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR CONFIG)
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} "-c" "import mlir;"
  RESULT_VARIABLE __mlirpy_missing
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
 )
FetchContent_Declare(mlir
	URL https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/mlir-16.0.6.src.tar.xz
	URL_HASH MD5=9c9dac6bd7fdaf682f60fabca839c578
	DOWNLOAD_EXTRACT_TIMESTAMP true
)
if(NOT MLIR_FOUND)
	message(STATUS "Missing MLIR, build it")
	set(MLIR_ENABLE_BINDINGS_PYTHON ON)
	FetchContent_MakeAvailable(mlir)
	list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
elseif(MLIR_FOUND AND __mlirpy_missing)
	message(STATUS "Found MLIR, but no mlirpy, build it")
	set(MLIR_STANDALONE_BUILD ON)
	set(MLIR_ENABLE_EXECUTION_ENGINE ON)
	FetchContent_Populate(mlir)
	set(MLIR_DIR ${mlir_SOURCE_DIR})
	set(MLIR_CMAKE_DIR ${mlir_SOURCE_DIR}/cmake/modules)
	list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

	include(AddMLIR)
	include(AddMLIRPython)
	include(MLIRDetectPythonEnv)
	mlir_configure_python_dev_packages()
	set(MLIR_SOURCE_DIR ${mlir_SOURCE_DIR})
	set(MLIR_BINARY_DIR ${CMAKE_INSTALL_PREFIX})
	add_custom_target(LinalgOdsGen)
	add_subdirectory(${mlir_SOURCE_DIR}/python)
else()
	list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
endif()
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
include(AddMLIR)
include(AddMLIRPython)
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")

## custom functions for mlir
function(tpu_add_mlir_dialect dialect dialect_namespace)
	message(STATUS "generate TPU MLIR dialect TpuDialect${dialect}IncGen")
	cmake_parse_arguments(TPU_ADD_DIAL "ATTR;ENUM;DOC" "" "" ${ARGN})
	set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
	if(TPU_ADD_DIAL_DOC)
		add_mlir_doc(${dialect}Ops ${dialect}Ops ./ -gen-op-doc)
	endif()
	if(TPU_ADD_DIAL_ENUM)
		mlir_tablegen(${dialect}Enum.h.inc -gen-enum-decls)
		mlir_tablegen(${dialect}Enum.cpp.inc -gen-enum-defs)
	endif()
	if(TPU_ADD_DIAL_ATTR)
		mlir_tablegen(${dialect}Attr.h.inc -gen-attrdef-decls)
		mlir_tablegen(${dialect}Attr.cpp.inc -gen-attrdef-defs)
	endif()
	add_public_tablegen_target(TpuDialect${dialect}IncGen)
	add_dependencies(mlir-headers TpuDialect${dialect}IncGen)
endfunction()

function(tpu_add_mlir_enum dialect)
	message(STATUS "generate TPU MLIR enum TpuEnum${dialect}IncGen")
	set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
	mlir_tablegen(${dialect}Enum.h.inc -gen-enum-decls)
	mlir_tablegen(${dialect}Enum.cpp.inc -gen-enum-defs)
	add_public_tablegen_target(TpuEnum${dialect}IncGen)
	add_dependencies(mlir-headers TpuEnum${dialect}IncGen)
endfunction()

function(tpu_add_mlir_interface interface)
	message(STATUS "generate TPU MLIR interface TpuInterface${interface}IncGen")
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
	add_public_tablegen_target(TpuInterface${interface}IncGen)
	add_dependencies(mlir-generic-headers TpuInterface${interface}IncGen)
endfunction()

function(tpu_add_mlir_pass name file)
	message(STATUS "generate TPU MLIR pass TpuPass${file}IncGen")
	set(LLVM_TARGET_DEFINITIONS ${file}.td)
	mlir_tablegen(${file}.h.inc -gen-pass-decls -name ${name})
	add_public_tablegen_target(TpuPass${name}IncGen)
endfunction()

function(tpu_add_library name)
	message(STATUS "generate TPU MLIR library TpuLib${name}")
	cmake_parse_arguments(TPU_ADD "EXCLUDE_FROM_ALL" "" "DEPENDS;LIBS;INCLUDES" ${ARGN})
	add_library(TpuLib${name} ${TPU_ADD_UNPARSED_ARGUMENTS})
	if(TPU_ADD_DEPENDS)
		add_dependencies(TpuLib${name} ${TPU_ADD_DEPENDS})
	endif()
	if(TPU_ADD_INCLUDES)
		target_include_directories(TpuLib${name} ${TPU_ADD_INCLUDES})
	endif()
	if(TPU_ADD_LIBS)
		target_link_libraries(TpuLib${name} ${TPU_ADD_LIBS})
	endif()
endfunction()

function(tpu_add_executable name)
	message(STATUS "generate TPU MLIR executable ${name}")
	cmake_parse_arguments(TPU_ADD "EXCLUDE_FROM_ALL" "" "DEPENDS;LIBS;INCLUDES" ${ARGN})
	add_executable(${name} ${TPU_ADD_UNPARSED_ARGUMENTS})
	if(TPU_ADD_DEPENDS)
		add_dependencies(${name} ${TPU_ADD_DEPENDS})
	endif()
	if(TPU_ADD_INCLUDES)
		target_include_directories(${name} ${TPU_ADD_INCLUDES})
	endif()
	if(TPU_ADD_LIBS)
		target_link_libraries(${name} ${TPU_ADD_LIBS})
	endif()
endfunction()

function(tpu_add_all_subdir)
	file(GLOB sources_list LIST_DIRECTORIES true */CMakeLists.txt)
	foreach(dir ${sources_list})
			IF(IS_DIRECTORY ${dir})
					add_subdirectory(${dir})
			ELSE()
					CONTINUE()
			ENDIF()
	endforeach()
endfunction()
################################

add_subdirectory(third_party)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

#-------------------------------------------------------------------------------
# generate version
execute_process(
  COMMAND git describe --tags --always
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_EXEC_RESULT
  OUTPUT_VARIABLE GIT_SHORT_HASH)

string(STRIP ${GIT_SHORT_HASH} GIT_SHORT_HASH)
string(FIND ${GIT_SHORT_HASH} "-" iLength)
string(SUBSTRING ${GIT_SHORT_HASH} 0 ${iLength} MAIN_VERSION)
math(EXPR iLength "${iLength} + 1")
string(SUBSTRING ${GIT_SHORT_HASH} ${iLength} -1 PATCH_VERSION)
set(GIT_SHORT_HASH "${MAIN_VERSION}.${PATCH_VERSION}")

string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(TPU_MLIR_VERSION "${GIT_SHORT_HASH}-${BUILD_TIME}" CACHE STRING "tpu-mlir version" FORCE)
message(STATUS "tpu-mlir version: ${TPU_MLIR_VERSION}")
add_definitions(-DTPU_MLIR_VERSION="${TPU_MLIR_VERSION}")
#-------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-unused-result -Wreturn-type -Wno-unused-variable -Wno-deprecated-declarations -Wno-non-c-typedef-for-linkage -Wno-free-nonheap-object -Wno-switch -DFLATBUFFERS_LOCALE_INDEPENDENT=0")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(bindings)
add_subdirectory(python)
