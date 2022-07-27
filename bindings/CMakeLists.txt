set(Python_ADDITIONAL_VERSIONS 3.7)
find_package(PythonLibs 3 REQUIRED)

add_subdirectory(pymlir)
add_subdirectory(kld)
add_subdirectory(pyruntime)
