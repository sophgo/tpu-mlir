pybind11_add_module(pymlir pymlir.cpp)

target_link_libraries(pymlir PRIVATE
	TpuLibInitAll
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)
install(TARGETS pymlir DESTINATION python)
