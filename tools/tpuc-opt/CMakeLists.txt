
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${dialect_libs}
  MLIROptLib
  SophgoInitAll
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
