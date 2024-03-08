add_subdirectory(TilingInterface)
add_subdirectory(Dialect)
include_directories(${CMAKE_BINARY_DIR}/experimental/include)
include_directories(${PROJECT_SOURCE_DIR}/experimental/include)

set(LLVM_LINK_COMPONENTS Core Support)

set(LIBS

  MLIROptLib
  MLIRPass
  MLIRTransforms
  TPUMLIRTilingInterfaceTestPasses
  TPUMLIRCodegenPass
  TPUMLIRBM1690Dialect
  MLIRMLProgramDialect
)

add_llvm_executable(tpuc-test
 tpuc-test.cpp

 DEPENDS
 ${LIBS}
)

target_link_libraries(tpuc-test PRIVATE ${LIBS})


llvm_update_compile_flags(tpuc-test)

mlir_check_all_link_libraries(tpuc-test)
