add_mlir_dialect(TopOps top)
#add_mlir_doc(TopOps TopOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TopOps.td)
mlir_tablegen(TopAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(TopAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTopAttrGen)
