cmake_minimum_required(VERSION 3.5)
project(TPUKernelSamples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(parse_list INPUT OUTPUT CHAR)
  string(REGEX REPLACE ":" "${CHAR}" TMP_LIST "${INPUT}")
  set(${OUTPUT} ${TMP_LIST} PARENT_SCOPE)
endfunction()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--no-undefined -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined -Werror")

if(NOT DEFINED ENV{PPL_PROJECT_ROOT})
  message(FATAL_ERROR "Please set environ PPL_PROJECT_ROOT to ppl release path")
else()
  set(PPL_TOP $ENV{PPL_PROJECT_ROOT})
  message(NOTICE "PPL_PATH: ${PPL_TOP}")
endif()

if(NOT DEFINED CHIP)
  message(FATAL_ERROR "Please set -DCHIP to chip type")
else()
  message(NOTICE "CHIP: ${CHIP}")
endif()
# Add chip arch defination
add_definitions(-D__${CHIP}__)
add_definitions(-DUSING_CMODEL)


if(DEFINED DEV_MODE)
  message(NOTICE "DEV_MODE: ${DEV_MODE}")
else()
  message(FATAL_ERROR "Please set -DDEV_MODE to cmodel/pcie/soc")
endif()

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  #set(CMAKE_BUILD_TYPE "Release")
  add_definitions(-O2 -g)
endif()

include($ENV{PPL_RUNTIME_PATH}/scripts/GenChipDef.cmake)
include($ENV{PPL_RUNTIME_PATH}/chip/${CHIP}/config_common.cmake)
# deal extra flags
parse_list("${EXTRA_IDIRS}" EXTRA_IDIRS ";")
parse_list("${EXTRA_LDIRS}" EXTRA_LDIRS ";")
parse_list("${EXTRA_LDFLAGS}" EXTRA_LDFLAGS ";")
parse_list("${EXTRA_CFLAGS}" EXTRA_CFLAGS " ")

option(USE_MPI "Build with MPI support" OFF)

include_directories(
  include
  ${CMAKE_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${KERNEL_TOP}
  ${TPUKERNEL_TOP}/kernel/include
  ${TPUKERNEL_TOP}/tpuDNN/include
  ${RUNTIME_TOP}/include
  ${CUS_TOP}/host/include
  ${CUS_TOP}/dev/utils/include
  ${CHECKER})


if(USE_MPI)
  add_definitions(-DMAX_TPU_CORE_NUM=${MAX_TPU_CORE_NUM})
  include_directories(
    ${PPL_TOP}/third_party/TPU1686/common/include
    ${PPL_TOP}/third_party/TPU1686/common/include/api
    ${PPL_TOP}/third_party/TPU1686/tpuDNN/src
    ${PPL_TOP}/third_party/TPU1686/tpuDNN/src/graph
  )
  set(CACHE_DIR "$ENV{HOME}/.ppl/cache")
  set(OMPI_VER "4.1.5")
  set(OMPI_DIR "${CACHE_DIR}/openmpi-${OMPI_VER}")
  set(OMPI_INSTALL_DIR "${CACHE_DIR}/openmpi-install")
  set(OMPI_TARBALL "${CACHE_DIR}/openmpi-${OMPI_VER}.tar.bz2")
  set(OMPI_URL "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VER}.tar.bz2")

  find_package(MPI QUIET)

  if(NOT MPI_FOUND)
    message(WARNING "System MPI not found. Will use cached/built OpenMPI ${OMPI_VER} under ${OMPI_INSTALL_DIR}")

    execute_process(
      COMMAND bash -lc "
        set -euo pipefail
        if [ -x \"${OMPI_INSTALL_DIR}/bin/mpicc\" ] && [ -e \"${OMPI_INSTALL_DIR}/lib/libmpi.so\" ]; then
          exit 0
        fi
        mkdir -p \"${CACHE_DIR}\"
        redownload() {
          wget -O \"${OMPI_TARBALL}.part\" \"${OMPI_URL}\" && mv \"${OMPI_TARBALL}.part\" \"${OMPI_TARBALL}\"
        }
        [ -f \"${OMPI_TARBALL}\" ] || redownload
        tar -tjf \"${OMPI_TARBALL}\" >/dev/null 2>&1 || { echo 'Corrupted tarball. Re-downloading...'; rm -f \"${OMPI_TARBALL}\"; redownload; tar -tjf \"${OMPI_TARBALL}\" >/dev/null; }
        [ -d \"${OMPI_DIR}\" ] || tar xjf \"${OMPI_TARBALL}\" -C \"${CACHE_DIR}\"
        mkdir -p \"${OMPI_DIR}/build\"
        cd \"${OMPI_DIR}/build\"
        ../configure --disable-mpi-fortran --enable-mpi --prefix=\"${OMPI_INSTALL_DIR}\"
        make -j\"$(nproc)\" install
      "
      RESULT_VARIABLE OMPI_RV
    )
    if(NOT OMPI_RV EQUAL 0)
      message(FATAL_ERROR "Failed to build/install OpenMPI to ${OMPI_INSTALL_DIR}")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH "${OMPI_INSTALL_DIR}")
    set(MPI_C_COMPILER "${OMPI_INSTALL_DIR}/bin/mpicc" CACHE FILEPATH "MPI C compiler" FORCE)
    set(MPI_CXX_COMPILER "${OMPI_INSTALL_DIR}/bin/mpicxx" CACHE FILEPATH "MPI CXX compiler" FORCE)

    set(ENV{CPATH} "${OMPI_INSTALL_DIR}/include:$ENV{CPATH}")
    set(ENV{LIBRARY_PATH} "${OMPI_INSTALL_DIR}/lib:$ENV{LIBRARY_PATH}")
    set(ENV{LD_LIBRARY_PATH} "${OMPI_INSTALL_DIR}/lib:$ENV{LD_LIBRARY_PATH}")

    foreach(v MPI_FOUND MPI_C_INCLUDE_PATH MPI_CXX_HEADER_DIR MPI_C_LIBRARIES MPI_C_LINK_FLAGS MPI_mpi_LIBRARY)
      unset(${v} CACHE)
    endforeach()
    find_package(MPI REQUIRED)
  endif()

  if (MPI_FOUND)
    add_definitions(-DUSING_MPI)
    if(MPI_CXX_HEADER_DIR)
      include_directories(${MPI_CXX_HEADER_DIR})
    elseif(MPI_C_INCLUDE_PATH)
      include_directories(${MPI_C_INCLUDE_PATH})
    endif()
  else()
    message(FATAL_ERROR "MPI is required but not found")
  endif()
endif()

link_directories(${BACKEND_LIB_PATH} ${RUNTIME_TOP}/lib)

set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/include/kernel_module_data.h")

add_custom_command(
    OUTPUT ${KERNEL_HEADER}
    COMMAND echo "const unsigned int kernel_module_data[] = {0}\;" > ${KERNEL_HEADER}
)

# Add a custom target that depends on the custom command
add_custom_target(gen_kernel_module_data_target DEPENDS ${KERNEL_HEADER})

aux_source_directory(host HOST_SRC_FILES)
add_library(host STATIC ${HOST_SRC_FILES})
target_include_directories(host PRIVATE ${EXTRA_IDIRS})
install(TARGETS host DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_dependencies(host gen_kernel_module_data_target)

message(NOTICE "EXTRA_LDFLAGS: ${EXTRA_LDFLAGS}")
message(NOTICE "EXTRA_IDIRS: ${EXTRA_IDIRS}")
message(NOTICE "EXTRA_CFLAGS: ${EXTRA_CFLAGS}")
message(NOTICE "EXTRA_LDIRS: ${EXTRA_LDIRS}")
find_package(ZLIB REQUIRED)
aux_source_directory(src APP_SRC_FILES)

add_executable(main ${APP_SRC_FILES}
                    ${CUS_TOP}/host/src/cnpy.cpp
                    ${CUS_TOP}/host/src/host_utils.cpp)
target_include_directories(main PRIVATE ${EXTRA_IDIRS})
target_link_directories(main PRIVATE ${EXTRA_LDIRS})
target_link_libraries(main PRIVATE ${EXTRA_LDFLAGS})
target_compile_options(main PRIVATE ${EXTRA_CFLAGS})
target_link_libraries(main PRIVATE host ${RUNTIME_LIBS} tpudnn pthread ${ZLIB_LIBRARIES})
if (USE_MPI)
  target_link_libraries(main PRIVATE dl ${MPI_C_LINK_FLAGS} ${MPI_mpi_LIBRARY} ${MPI_C_LIBRARIES})
endif()
add_dependencies(main gen_kernel_module_data_target)
set_target_properties(main PROPERTIES OUTPUT_NAME test_case)
install(TARGETS main DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

aux_source_directory(device KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES} ${CUS_TOP}/dev/utils/src/ppl_helper.c ${KERNEL_CHECKER})
target_compile_options(firmware PRIVATE ${EXTRA_PLFLAGS})
target_link_libraries(firmware PRIVATE ${FIRMWARE_CMODEL} m)
set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)
install(TARGETS firmware DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
