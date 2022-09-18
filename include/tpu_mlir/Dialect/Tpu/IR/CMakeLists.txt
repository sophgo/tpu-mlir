add_mlir_dialect(TpuOps tpu)
#add_mlir_doc(TpuOps TpuOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TpuOps.td)
mlir_tablegen(TpuEnum.h.inc -gen-enum-decls)
mlir_tablegen(TpuEnum.cpp.inc -gen-enum-defs)
mlir_tablegen(TpuAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(TpuAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTpuAttrGen)
