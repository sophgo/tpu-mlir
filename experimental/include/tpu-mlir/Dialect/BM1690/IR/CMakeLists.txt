set(LLVM_TARGET_DEFINITIONS BM1690Enums.td)
mlir_tablegen(BM1690Enum.h.inc -gen-enum-decls -typedefs-dialect=bm1690)
mlir_tablegen(BM1690Enum.cpp.inc -gen-enum-defs -typedefs-dialect=bm1690)

set(LLVM_TARGET_DEFINITIONS BM1690Ops.td)
mlir_tablegen(BM1690Ops.h.inc -gen-op-decls)
mlir_tablegen(BM1690Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS BM1690Types.td)
mlir_tablegen(BM1690Types.h.inc -gen-typedef-decls -typedefs-dialect=bm1690)
mlir_tablegen(BM1690Types.cpp.inc -gen-typedef-defs -typedefs-dialect=bm1690)
mlir_tablegen(BM1690AttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(BM1690AttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS BM1690.td)
mlir_tablegen(BM1690Dialect.h.inc -gen-dialect-decls -dialect=bm1690)
mlir_tablegen(BM1690Dialect.cpp.inc -gen-dialect-defs -dialect=bm1690)

set(LLVM_TARGET_DEFINITIONS StructuredOpsInterfaces.td)
mlir_tablegen(StructuredOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(StructuredOpsInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TPUMLIRBM1690Gen)
