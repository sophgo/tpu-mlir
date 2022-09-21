set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tpu)
add_public_tablegen_target(MLIRTpuPassIncGen)

#add_mlir_doc(Passes TpuPasses ./ -gen-pass-doc)
