add_mlir_dialect(TPU tpu)

set(LLVM_TARGET_DEFINITIONS TPU.td)
mlir_tablegen(TPUAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(TPUAttr.cpp.inc -gen-attrdef-defs)
mlir_tablegen(TPUOps.h.inc -gen-op-decls)
mlir_tablegen(TPUOps.cpp.inc -gen-op-defs)
mlir_tablegen(TPUOpsDialect.h.inc -gen-dialect-decls -dialect=tpu)
mlir_tablegen(TPUOpsDialect.cpp.inc -gen-dialect-defs -dialect=tpu)
add_public_tablegen_target(MLIRTPUGen)

# Exclude tests from .so
add_mlir_library(TPUDialect
  TPUDialect.cpp
  PARTIAL_SOURCES_INTENDED
  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTPUGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)

target_include_directories(TPUDialect
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
