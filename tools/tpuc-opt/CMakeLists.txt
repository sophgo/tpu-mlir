
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

add_llvm_executable(tpuc-opt
  tpuc-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(tpuc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(tpuc-opt)

mlir_check_all_link_libraries(tpuc-opt)

install(TARGETS tpuc-opt DESTINATION bin)
