cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pyruntime)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(pyruntime_bm pyruntime_bm.cpp)
target_link_libraries(pyruntime_bm PRIVATE
  bmrt
  cpuop
)
nanobind_add_module(pyruntime_cvi pyruntime_cvi.cpp)
target_link_libraries(pyruntime_cvi PRIVATE
  cviruntime
)

nanobind_add_module(pyruntime_tpuv7 pyruntime_tpuv7.cpp)
target_link_libraries(pyruntime_tpuv7 PRIVATE
  tpuv7_rt
  tpuv7_modelrt
  cdm_daemon_emulator
)
install(TARGETS pyruntime_bm pyruntime_cvi pyruntime_tpuv7 DESTINATION python)
