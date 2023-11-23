add_mlir_dialect_library(TPUMLIRSG2260Dialect
  SG2260Dialect.cpp
  SG2260Ops.cpp
  SG2260Types.cpp
  SG2260StructuredOpsInterfaces.cpp

  DEPENDS
  TPUMLIRSG2260Gen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRDialectUtils
)
