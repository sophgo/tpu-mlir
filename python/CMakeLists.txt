include(AddMLIRPython)

set(TOP_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir")
set(TOP_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")

################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(TopMLIRPythonSources)
declare_mlir_python_sources(TopMLIRPythonSources.Dialects
  ROOT_DIR "${TOP_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TopMLIRPythonSources
)

################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TopMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir"
  TD_FILE dialects/TopOps.td
  SOURCES dialects/top.py
  DIALECT_NAME top)

################################################################################
# The fully assembled package of modules.
################################################################################
set(_source_components
  TopMLIRPythonSources
)
add_mlir_python_modules(TopMLIRPythonModules
  ROOT_PREFIX "${TOP_MLIR_PYTHON_PACKAGES_DIR}/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES ${_source_components}
  )


add_subdirectory(transform)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "test_onnx.py"
)
