set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_BUILD_TYPE Debug)

add_library(atomic_exec SHARED ./atomic_exec.cpp)
target_link_libraries(atomic_exec PRIVATE bmlib)
install(TARGETS atomic_exec DESTINATION lib)

# for debug usage
add_library(atomic_exec_bm1688 SHARED ./atomic_exec_bm1688.cpp)
target_link_libraries(atomic_exec_bm1688 PRIVATE bmlib)
install(TARGETS atomic_exec_bm1688 DESTINATION lib)


# add_executable(
#   atomic_exec_test
#   atomic_exec_bm1688.cpp
# )
# target_link_libraries(atomic_exec_test bmlib pthread dl)
