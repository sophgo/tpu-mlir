if ("$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "backend")
  # clean up ppl cache
  set(PPL_CACHE_PATH $ENV{PPL_CACHE_PATH})
  if("${PPL_CACHE_PATH}" STREQUAL "")
    set(PPL_CACHE_PATH "$ENV{HOME}/.ppl/cache")
  endif()
  file(REMOVE_RECURSE ${PPL_CACHE_PATH})
  if(EXISTS ${PPL_CACHE_PATH})
    message(WARNING "Failed to remove cache directory: ${PPL_CACHE_PATH}")
  else()
    message(STATUS "Cache directory removed successfully: ${PPL_CACHE_PATH}")
  endif()
  # gen ppl head and src
  set(PPL_TOP "$ENV{PPL_PROJECT_ROOT}")
  file(GLOB PL_FILES "${SRC_DIR}/PplBackend/src/*.pl")

  foreach(FILE ${PL_FILES})
      set(COMMAND "ppl-compile ${FILE} --I ${PPL_TOP}/inc --desc --O2 --o .")
      message(STATUS "Compiling file: ${FILE}")
      execute_process(COMMAND bash -c "${COMMAND}")
  endforeach()
  # prepare build env
  set(CUS_TOP ${PPL_TOP}/runtime/customize)
  include_directories(${CUS_TOP}/include)
  include_directories(${SRC_DIR}/PplBackend/include)
  file(GLOB_RECURSE PL_SRC
    ${CMAKE_BINARY_DIR}/host/*.cpp
    ${SRC_DIR}/PplBackend/src/*.cpp
    ${SRC_DIR}/PplBackend/src/*.c
    ${CUS_TOP}/src/host_utils.cpp
    ${CUS_TOP}/src/ppl_jit.cpp
  )
endif()

