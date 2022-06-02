set(LIBS
  TPUMLIRBuilder
  flatbuffers
  )

add_llvm_executable(model_tool
  model_tool.cpp
  )
target_link_libraries(model_tool PRIVATE ${LIBS})
llvm_update_compile_flags(model_tool)

mlir_check_all_link_libraries(model_tool)

install(TARGETS model_tool DESTINATION bin)
