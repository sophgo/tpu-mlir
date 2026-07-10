# Generate Types and Dialect files (normally part of add_mlir_dialect, but
# we skip op .h.inc/.cpp.inc because we generate them separately below).
set(LLVM_TARGET_DEFINITIONS TpuOps.td)
mlir_tablegen(TpuOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=tpu)
mlir_tablegen(TpuOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tpu)
mlir_tablegen(TpuOpsDialect.h.inc -gen-dialect-decls -dialect=tpu)
mlir_tablegen(TpuOpsDialect.cpp.inc -gen-dialect-defs -dialect=tpu)
add_public_tablegen_target(MLIRTpuOpsIncGen)
add_dependencies(mlir-headers MLIRTpuOpsIncGen)

# Op declarations: generated WITH --op-shard-count. The patched mlir-tblgen
# unconditionally emits GET_OP_CLASSES (op class declarations) via
# emitOpClassDecls, and when sharding also emits registration hook declarations.
set(LLVM_TARGET_DEFINITIONS TpuOps.td)
mlir_tablegen(TpuOps.h.inc -gen-op-decls -op-shard-count=${TPUMLIR_OP_SHARD_COUNT})

# Op definitions: sharded — TpuOps.cpp.inc has GET_OP_DEFS_N guards.
mlir_tablegen(TpuOps.cpp.inc -gen-op-defs -op-shard-count=${TPUMLIR_OP_SHARD_COUNT})

# Generate per-shard .cpp files from the template (TpuOps.cpp) using
# mlir-src-sharder. Each shard defines GET_OP_DEFS_<index> to select its
# slice of op definitions from TpuOps.cpp.inc.
# mlir-tblgen --op-shard-count=N produces shards GET_OP_DEFS_0..N-1 (N shards),
# so we generate N .cpp files with indices 0..N-1.
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/lib/Dialect/Tpu/IR/TpuOps.cpp)
math(EXPR _last_shard "${TPUMLIR_OP_SHARD_COUNT} - 1")
foreach(index RANGE 0 ${_last_shard})
    tablegen(MLIR_SRC_SHARDER TpuOps.${index}.cpp -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRTpuOpsShardGen)
add_dependencies(mlir-headers MLIRTpuOpsShardGen)

#add_mlir_doc(TpuOps TpuOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TpuOps.td)
mlir_tablegen(TpuEnum.h.inc -gen-enum-decls)
mlir_tablegen(TpuEnum.cpp.inc -gen-enum-defs)
mlir_tablegen(TpuAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(TpuAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTpuAttrGen)
