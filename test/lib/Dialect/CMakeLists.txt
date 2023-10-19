add_mlir_dialect(SG2260 sg2260)

set(LLVM_TARGET_DEFINITIONS SG2260.td)
mlir_tablegen(SG2260Attr.h.inc -gen-attrdef-decls)
mlir_tablegen(SG2260Attr.cpp.inc -gen-attrdef-defs)
mlir_tablegen(SG2260Ops.h.inc -gen-op-decls)
mlir_tablegen(SG2260Ops.cpp.inc -gen-op-defs)
mlir_tablegen(SG2260OpsDialect.h.inc -gen-dialect-decls -dialect=sg2260)
mlir_tablegen(SG2260OpsDialect.cpp.inc -gen-dialect-defs -dialect=sg2260)
add_public_tablegen_target(MLIRSG2260Gen)

# Exclude tests from .so
add_mlir_library(SG2260Dialect
  SG2260Dialect.cpp
  PARTIAL_SOURCES_INTENDED
  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRSG2260Gen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDerivedAttributeOpInterface
  MLIRDestinationStyleOpInterface
  MLIRDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRReduce
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
)

target_include_directories(SG2260Dialect
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)

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
  MLIRDialectUtils
  MLIRIR
)

target_include_directories(TPUDialect
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
