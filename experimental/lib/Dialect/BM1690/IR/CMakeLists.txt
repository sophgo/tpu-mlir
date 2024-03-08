add_mlir_dialect_library(TPUMLIRBM1690Dialect
  BM1690Dialect.cpp
  BM1690Ops.cpp
  BM1690Types.cpp
  BM1690StructuredOpsInterfaces.cpp

  DEPENDS
  TPUMLIRBM1690Gen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRDialectUtils
)
