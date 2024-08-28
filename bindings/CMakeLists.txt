set(Python_ADDITIONAL_VERSIONS 3.7)
find_package(PythonLibs 3 REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.7)

add_subdirectory(pyfinalmlir)
add_subdirectory(pymlir)
add_subdirectory(kld)
add_subdirectory(pyruntime)
