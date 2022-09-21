set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Top)
add_public_tablegen_target(MLIRTopPassIncGen)

#add_mlir_doc(Passes TopPasses ./ -gen-pass-doc)
