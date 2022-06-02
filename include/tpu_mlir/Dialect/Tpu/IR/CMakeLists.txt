add_mlir_dialect(TpuOps tpu)
add_mlir_doc(TpuOps TpuOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TpuAttr.td)
mlir_tablegen(TpuAttr.h.inc -gen-struct-attr-decls)
mlir_tablegen(TpuAttr.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRTpuAttrGen)
