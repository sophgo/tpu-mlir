set(TPUV7_INCLUDE_DIRS "/opt/tpuv7/tpuv7-current/include")
set(TPUV7_LIB_DIRS "/opt/tpuv7/tpuv7-current/lib")

file(GLOB lib_names RELATIVE ${TPUV7_LIB_DIRS} ${TPUV7_LIB_DIRS}/*)

foreach(name ${lib_names})
    find_library(the_${name} "${name}" PATHS ${TPUV7_LIB_DIRS})
endforeach()

