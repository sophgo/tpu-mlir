set(LLVM_TARGET_DEFINITIONS SG2260Enums.td)
mlir_tablegen(SG2260Enum.h.inc -gen-enum-decls -typedefs-dialect=sg2260)
mlir_tablegen(SG2260Enum.cpp.inc -gen-enum-defs -typedefs-dialect=sg2260)

set(LLVM_TARGET_DEFINITIONS SG2260Ops.td)
mlir_tablegen(SG2260Ops.h.inc -gen-op-decls)
mlir_tablegen(SG2260Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS SG2260Types.td)
mlir_tablegen(SG2260Types.h.inc -gen-typedef-decls -typedefs-dialect=sg2260)
mlir_tablegen(SG2260Types.cpp.inc -gen-typedef-defs -typedefs-dialect=sg2260)
mlir_tablegen(SG2260AttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(SG2260AttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS SG2260.td)
mlir_tablegen(SG2260Dialect.h.inc -gen-dialect-decls -dialect=sg2260)
mlir_tablegen(SG2260Dialect.cpp.inc -gen-dialect-defs -dialect=sg2260)

set(LLVM_TARGET_DEFINITIONS StructuredOpsInterfaces.td)
mlir_tablegen(StructuredOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(StructuredOpsInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TPUMLIRSG2260Gen)
