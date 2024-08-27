set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tpu)
add_public_tablegen_target(MLIRTpuPassIncGen)

# set(LLVM_TARGET_DEFINITIONS RewritePattern.td)
# mlir_tablegen(RewritePattern.inc -gen-rewriters)
# add_public_tablegen_target(MLIRTpuRewriteIncGen)
#add_mlir_doc(Passes TpuPasses ./ -gen-pass-doc)
