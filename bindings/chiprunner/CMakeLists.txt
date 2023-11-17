cmake_minimum_required(VERSION 3.5)
project(AtomicExec)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(atomic_exec SHARED ./atomic_exec.cpp)

target_link_libraries(atomic_exec PRIVATE bmlib)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_BUILD_TYPE Debug)
install(TARGETS atomic_exec DESTINATION lib)
