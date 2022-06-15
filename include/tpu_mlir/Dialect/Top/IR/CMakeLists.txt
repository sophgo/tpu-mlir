add_mlir_dialect(TopOps top)
add_mlir_doc(TopOps TopOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TopAttr.td)
mlir_tablegen(TopAttr.h.inc -gen-struct-attr-decls)
mlir_tablegen(TopAttr.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRTopAttrGen)
