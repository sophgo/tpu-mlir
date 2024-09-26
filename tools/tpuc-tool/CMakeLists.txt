
set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRTosaDialect
  MLIRFuncDialect
  MLIROptLib
  TPUMLIRInitAll
  )

add_llvm_executable(tpuc-tool
  tpuc-tool.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(tpuc-tool PRIVATE ${LIBS})
llvm_update_compile_flags(tpuc-tool)

mlir_check_all_link_libraries(tpuc-tool)

install(TARGETS tpuc-tool DESTINATION bin)
