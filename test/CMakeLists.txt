configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py)

add_subdirectory(llvm-lit)

set(TPUMLIR_TEST_DEPENDS #
    FileCheck count not split-file tpuc-opt)

list(APPEND TPUMLIR_TEST_DEPENDS TPUMLIRUnitTests)
set(LLVM_DEFAULT_EXTERNAL_LIT ${PROJECT_BINARY_DIR}/bin/llvm-lit)

add_lit_testsuite(check-tpumlir "Running the TPU-MLIR regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TPUMLIR_TEST_DEPENDS})

set_target_properties(check-tpumlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(TPUMLIR ${CMAKE_CURRENT_BINARY_DIR} DEPENDS
                   ${TPUMLIR_TEST_DEPENDS})


add_subdirectory(lib)
