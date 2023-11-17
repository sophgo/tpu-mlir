set(LLVM_OPTIONAL_SOURCES
  null.cpp
)
include_directories(${PROJECT_SOURCE_DIR}/tools/tpuc-opt-experiment/)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_INSTALL_PREFIX}/include)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

add_subdirectory(InputConversion)
add_subdirectory(GlobalOptimization)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRInputConversion
  MLIRGlobalOptimization
  )

add_llvm_executable(tpuc-opt-exp
  tpuc-opt-exp.cpp

  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
  )
target_link_libraries(tpuc-opt-exp PRIVATE ${LIBS})
llvm_update_compile_flags(tpuc-opt-exp)

mlir_check_all_link_libraries(tpuc-opt-exp)
