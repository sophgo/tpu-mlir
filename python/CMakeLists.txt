include(AddMLIRPython)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(TOP_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir")
set(TOP_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")

################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(TPUMLIRPythonSources)
declare_mlir_python_sources(TPUMLIRPythonSources.Dialects
  ROOT_DIR "${TOP_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TPUMLIRPythonSources
)
################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TPUMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir"
  TD_FILE dialects/TopOps.td
  SOURCES dialects/top.py
  DIALECT_NAME top)


declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TPUMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir"
  TD_FILE dialects/TpuOps.td
  SOURCES dialects/tpu.py
  DIALECT_NAME tpu)

declare_mlir_python_extension(TPUMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpu_mlir"
  SOURCES
    RegisterEverything.cpp
  EMBED_CAPI_LINK_LIBS
    TPUMLIRCAPI
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
################################################################################

add_mlir_python_common_capi_library(TPUMLIRPythonCAPI
  INSTALL_COMPONENT TPUMLIRPythonModules
  INSTALL_DESTINATION python/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    TPUMLIRPythonSources
    TPUMLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
  EMBED_LIBS
    MLIRCAPIQuant
)

################################################################################
# The fully assembled package of modules.
################################################################################

add_mlir_python_modules(TPUMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES
    TPUMLIRPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.quant
    MLIRPythonSources.Dialects.tosa
    MLIRPythonSources.Dialects.func
    TPUMLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    TPUMLIRPythonCAPI
)


add_subdirectory(transform)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "*.js"
  PATTERN "*.html"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
  PATTERN "tpu_mlir" EXCLUDE
  PATTERN "tools/soc_infer" EXCLUDE
)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "test_onnx.py"
  PATTERN "test_tpulang.py"
)

install(CODE "
  execute_process(COMMAND find \"\${CMAKE_INSTALL_PREFIX}/python\" -type d -empty -delete)
")
