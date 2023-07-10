set(LIBS
TPUMLIRBuilder
TPUMLIRInitAll
TPUMLIRSupport
cviruntime
)
add_llvm_executable(cvimodel_debug
  cvimodel_debug.cpp
  cpu_func.cpp
  utils.cpp
  )
target_link_libraries(cvimodel_debug PRIVATE ${LIBS})
llvm_update_compile_flags(cvimodel_debug)

mlir_check_all_link_libraries(cvimodel_debug)

install(TARGETS cvimodel_debug DESTINATION bin)
