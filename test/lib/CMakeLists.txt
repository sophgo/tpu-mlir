add_subdirectory(TilingInterface)

set(LLVM_LINK_COMPONENTS Core Support)

set(LIBS

  MLIROptLib
  MLIRPass
  MLIRTransforms
  TPUMLIRTilingInterfaceTestPasses
  )

add_llvm_executable(tpuc-test
 tpuc-test.cpp

 DEPENDS
 ${LIBS}
)

target_link_libraries(tpuc-test PRIVATE ${LIBS})
llvm_update_compile_flags(tpuc-test)

mlir_check_all_link_libraries(tpuc-test)
