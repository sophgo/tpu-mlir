set(LIBS
  SophgoBuilder
  flatbuffers
  )

add_llvm_executable(bm_model.bin
  bm_model.cpp
  )
target_link_libraries(bm_model.bin PRIVATE ${LIBS})
llvm_update_compile_flags(bm_model.bin)

mlir_check_all_link_libraries(bm_model.bin)

install(TARGETS bm_model.bin DESTINATION bin)
