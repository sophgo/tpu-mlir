add_custom_target(TPUMLIRUnitTests)
set_target_properties(TPUMLIRUnitTests PROPERTIES FOLDER "TPU-MLIR Tests")

function(add_tpumlir_unittest test_dirname)
  add_unittest(TPUMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Backend)
add_subdirectory(Linalg)
add_subdirectory(Target)
add_subdirectory(Support)
