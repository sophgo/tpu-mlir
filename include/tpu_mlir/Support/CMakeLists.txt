
set(LLVM_TARGET_DEFINITIONS Module.td)
mlir_tablegen(ModuleEnum.h.inc -gen-enum-decls)
mlir_tablegen(ModuleEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(ModuleAttrGen)
