set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tops)
add_public_tablegen_target(MLIRTopsPassIncGen)

add_mlir_doc(Passes TopsPasses ./ -gen-pass-doc)
