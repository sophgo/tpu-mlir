add_tpumlir_unittest(
 TensorManipulationTest
 TensorManipulationTest.cpp
 PARTIAL_SOURCES_INTENDED
)

target_link_libraries(
  TensorManipulationTest
  PRIVATE
  MLIRSupport
)
