# patched-tblgen

Pre-built mlir-tblgen and mlir-src-sharder with op-sharding support, backported
from LLVM 19.1.7 to LLVM 18.0.0git. See CMakeLists.txt (TPUMLIR_OP_SHARD_COUNT)
for usage and mlir-tblgen-shard.patch for implementation details.

## Files

| File | Description |
|---|---|
| `mlir-tblgen` | Patched mlir-tblgen with `--op-shard-count` support |
| `mlir-src-sharder` | Tool from LLVM 19, generates per-shard .cpp files |
| `mlir-tblgen-shard.patch` | Patch backporting shard logic to LLVM 18's mlir-tblgen |
| `build_patched_tblgen.sh` | Reproducible build script (fetches LLVM source, applies patch, compiles) |

## Rebuilding

```bash
bash build_patched_tblgen.sh
```

Outputs `mlir-tblgen` and `mlir-src-sharder` into this directory.
