pybind11_add_module(pyruntime_bm pyruntime_bm.cpp)
target_link_libraries(pyruntime_bm PRIVATE
  bmrt
)
pybind11_add_module(pyruntime_cvi pyruntime_cvi.cpp)
target_link_libraries(pyruntime_cvi PRIVATE
  cviruntime
)
install(TARGETS pyruntime_bm pyruntime_cvi DESTINATION python)
