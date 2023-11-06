add_mlir_dialect_library(TPUMLIRSG2260Dialect
  SG2260Dialect.cpp
  SG2260Ops.cpp
  SG2260Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/SG2260

  DEPENDS
  MLIRSG2260Gen
  LINK_LIBS
  PUBLIC
  MLIRIR
)
