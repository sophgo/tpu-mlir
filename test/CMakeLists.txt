find_program(
  FileCheck
  NAMES FileCheck
  PATHS
  ${CMAKE_SOURCE_DIR}/third_party/llvm/bin
)

find_package(
  Python3 3.6 REQUIRED
  COMPONENTS Interpreter
)


set(MLIR_RUNNER_UTILS_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(
  MLIR_TEST_DEPENDS
  sophgo-opt
)

# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   list(
#     APPEND MLIR_TEST_DEPENDS
#     MLIRPythonModules
#   )
# endif()

# This target an be used to just build the dependencies
# for the check-mlir target without executing the tests.
# This is useful for bots when splitting the build step
# from the testing step.
add_custom_target(
  check-sophgo-mlir-build-only
  DEPENDS ${MLIR_TEST_DEPENDS}
)

include(llvm-lit.cmake)

add_lit_testsuite(
  check-sophgo-mlir "Running the SOPHGO-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
)
set_target_properties(check-sophgo-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(
  Sophgo-MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
)
