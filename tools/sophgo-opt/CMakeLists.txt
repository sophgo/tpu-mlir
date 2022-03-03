
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

add_llvm_executable(sophgo-opt
  sophgo-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(sophgo-opt PRIVATE ${LIBS})
llvm_update_compile_flags(sophgo-opt)

mlir_check_all_link_libraries(sophgo-opt)

install(TARGETS sophgo-opt DESTINATION bin)
