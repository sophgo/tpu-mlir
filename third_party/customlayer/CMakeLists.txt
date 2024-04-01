cmake_minimum_required(VERSION 3.5)
project(custom_layer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAFETY_FLAGS}")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if ("$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "unittest")
    include(utils/unittest.cmake)
elseif ("$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "plugin")
    include_directories(${SRC_DIR}/include)
    include_directories(${SRC_DIR}/include/kernel)
    include_directories(${SRC_DIR}/include/custom_ap)

    file(GLOB_RECURSE SRC_FILES
        ${SRC_DIR}/plugin/*.c
        ${SRC_DIR}/plugin/*.cpp
        ${SRC_DIR}/ap_src/*.cpp
    )

    include(utils/plugin.cmake)
elseif("$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "customcpuop")
    include(utils/customcpuop.cmake)
else()
    set(REGSTERED_OP_NAMES)
    macro(register_custom_op op_name)
        list(APPEND REGSTERED_OP_NAMES ${op_name})
    endmacro()
    set(REGSTERED_LOCAL_OP_NAMES)
    macro(register_custom_local_op op_name)
        list(APPEND REGSTERED_LOCAL_OP_NAMES ${op_name})
    endmacro()
    set(REGSTERED_GLOBAL_BFSZ_NAMES)
    macro(register_custom_global_bfsz op_name)
        list(APPEND REGSTERED_GLOBAL_BFSZ_NAMES ${op_name})
    endmacro()
    set(REGSTERED_LOCAL_BFSZ_NAMES)
    macro(register_custom_local_bfsz op_name)
        list(APPEND REGSTERED_LOCAL_BFSZ_NAMES ${op_name})
    endmacro()
    include(${SRC_DIR}/register_ops.cmake)
    list(REMOVE_DUPLICATES REGSTERED_OP_NAMES)
    list(REMOVE_DUPLICATES REGSTERED_LOCAL_OP_NAMES)
    list(REMOVE_DUPLICATES REGSTERED_GLOBAL_BFSZ_NAMES)
    list(REMOVE_DUPLICATES REGSTERED_LOCAL_BFSZ_NAMES)

    include(utils/gen_interface_include_file.cmake)
    if ("$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "backend")
        include(utils/gen_backend_api_file.cmake)
    else()
        include(utils/gen_entry_file.cmake)
    endif()

    include_directories(${CMAKE_BINARY_DIR}/include)
    include_directories(${SRC_DIR}/include)
    include_directories(${SRC_DIR}/include/kernel)

    file(GLOB_RECURSE SRC_FILES
        ${SRC_DIR}/src/*.cpp
        ${SRC_DIR}/src/*.c
        ${CMAKE_BINARY_DIR}/src/*.c
    )

    include(utils/$ENV{CUSTOM_LAYER_DEV_MODE}.cmake)
endif()
