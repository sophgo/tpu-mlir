# Generate Types and Dialect files (normally part of add_mlir_dialect, but
# we skip op .h.inc/.cpp.inc because we generate them separately below).
set(LLVM_TARGET_DEFINITIONS TopOps.td)
mlir_tablegen(TopOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=top)
mlir_tablegen(TopOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=top)
mlir_tablegen(TopOpsDialect.h.inc -gen-dialect-decls -dialect=top)
mlir_tablegen(TopOpsDialect.cpp.inc -gen-dialect-defs -dialect=top)
add_public_tablegen_target(MLIRTopOpsIncGen)
add_dependencies(mlir-headers MLIRTopOpsIncGen)

# Op declarations: generated WITH --op-shard-count. The patched mlir-tblgen
# unconditionally emits GET_OP_CLASSES (op class declarations) via
# emitOpClassDecls, and when sharding also emits registration hook declarations.
set(LLVM_TARGET_DEFINITIONS TopOps.td)
mlir_tablegen(TopOps.h.inc -gen-op-decls -op-shard-count=${TPUMLIR_OP_SHARD_COUNT})

# Op definitions: sharded — TopOps.cpp.inc has GET_OP_DEFS_N guards.
mlir_tablegen(TopOps.cpp.inc -gen-op-defs -op-shard-count=${TPUMLIR_OP_SHARD_COUNT})

# Generate per-shard .cpp files from the template (TopOps.cpp).
# mlir-tblgen --op-shard-count=N produces shards GET_OP_DEFS_0..N-1 (N shards),
# so we generate N .cpp files with indices 0..N-1.
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/lib/Dialect/Top/IR/TopOps.cpp)
math(EXPR _last_shard "${TPUMLIR_OP_SHARD_COUNT} - 1")
foreach(index RANGE 0 ${_last_shard})
    tablegen(MLIR_SRC_SHARDER TopOps.${index}.cpp -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRTopOpsShardGen)
add_dependencies(mlir-headers MLIRTopOpsShardGen)

#add_mlir_doc(TopOps TopOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TopOps.td)
mlir_tablegen(TopAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(TopAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTopAttrGen)
