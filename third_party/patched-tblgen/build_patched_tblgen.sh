#!/bin/bash
# build_patched_tblgen.sh - Build mlir-tblgen + mlir-src-sharder with op-sharding backport
#
# This script backports the --op-shard-count / mlir-src-sharder feature from LLVM 19
# to the LLVM 18.0.0git installed in the Docker image. It produces two binaries:
#   - mlir-tblgen     (patched, supports --op-shard-count=N)
#   - mlir-src-sharder (new tool from LLVM 19)
#
# The build is deterministic and reproducible. No network access is needed beyond
# the initial git fetch of the LLVM source commits.
#
# Usage: bash build_patched_tblgen.sh [output_dir]
#   output_dir defaults to the directory containing this script

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Exact LLVM commit from tpu-mlir Dockerfile (ARG LLVM_VERSION)
LLVM_COMMIT="c67e443895d5b922d1ffc282d23ca31f7161d4fb"
# LLVM 19 tag for mlir-src-sharder (new tool, doesn't exist in LLVM 18)
LLVM19_TAG="llvmorg-19.1.7"
# Script directory (for locating the patch file)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/mlir-tblgen-shard.patch"
# Output directory (defaults to the directory containing this script)
OUTPUT_DIR="${1:-${SCRIPT_DIR}}"
# Working directory
WORK_DIR="/tmp/mlir-tblgen-patched"
SRC_DIR="${WORK_DIR}/src"
# LLVM git repo cache (reused across runs)
LLVM_REPO="/tmp/llvm-project"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[build-tblgen] $*"; }
fail() { echo "[build-tblgen] ERROR: $*" >&2; exit 1; }

# ── Step 1: Fetch LLVM source ────────────────────────────────────────────────
log "Fetching LLVM source commits..."
mkdir -p "$(dirname "$LLVM_REPO")"
if [ -d "$LLVM_REPO" ] && [ ! -d "$LLVM_REPO/.git" ]; then
  fail "$LLVM_REPO exists but is not a git repo. Remove it or set LLVM_REPO."
fi
if [ ! -d "$LLVM_REPO/.git" ]; then
  git clone --depth=1 https://github.com/llvm/llvm-project.git "$LLVM_REPO"
fi
cd "$LLVM_REPO"
git fetch --depth=1 origin "$LLVM_COMMIT" 2>/dev/null || \
  fail "git fetch $LLVM_COMMIT failed. Check network or LLVM_REPO=$LLVM_REPO."
git fetch --depth=1 origin tag "$LLVM19_TAG" 2>/dev/null || \
  fail "git fetch tag $LLVM19_TAG failed. Check network or LLVM_REPO=$LLVM_REPO."
log "LLVM source ready."

# ── Step 2: Extract source files ─────────────────────────────────────────────
log "Extracting source from LLVM commit ${LLVM_COMMIT:0:8}..."
rm -rf "$WORK_DIR"
mkdir -p "$SRC_DIR"/{mlir-tblgen,mlir-src-sharder,lib/TableGen,lib/Support,lib/Tools/mlir-tblgen}
mkdir -p "$SRC_DIR"/include/mlir/{TableGen,Support,Tools/mlir-tblgen}

# mlir-tblgen sources (from mlir/tools/mlir-tblgen/CMakeLists.txt at LLVM_COMMIT)
TBLGEN_FILES=(
  AttrOrTypeDefGen.cpp AttrOrTypeFormatGen.cpp BytecodeDialectGen.cpp
  DialectGen.cpp DirectiveCommonGen.cpp EnumsGen.cpp FormatGen.cpp
  LLVMIRConversionGen.cpp LLVMIRIntrinsicGen.cpp mlir-tblgen.cpp
  OpClass.cpp OpDefinitionsGen.cpp OpDocGen.cpp OpFormatGen.cpp
  OpGenHelpers.cpp OpInterfacesGen.cpp OpPythonBindingGen.cpp
  PassCAPIGen.cpp PassDocGen.cpp PassGen.cpp RewriterGen.cpp SPIRVUtilsGen.cpp
)
# Also need .h files from mlir-tblgen directory
for f in "${TBLGEN_FILES[@]}"; do
  git show "${LLVM_COMMIT}:mlir/tools/mlir-tblgen/$f" > "$SRC_DIR/mlir-tblgen/$f"
done
for f in $(git ls-tree --name-only "$LLVM_COMMIT" mlir/tools/mlir-tblgen/ | grep '\.h$'); do
  git show "${LLVM_COMMIT}:$f" > "$SRC_DIR/mlir-tblgen/$(basename $f)"
done

# lib/TableGen sources (from mlir/lib/TableGen/CMakeLists.txt at LLVM_COMMIT)
TABLEGEN_LIB_FILES=(
  Argument.cpp Attribute.cpp AttrOrTypeDef.cpp Builder.cpp Class.cpp
  CodeGenHelpers.cpp Constraint.cpp Dialect.cpp Format.cpp GenInfo.cpp
  Interfaces.cpp Operator.cpp Pass.cpp Pattern.cpp Predicate.cpp Property.cpp
  Region.cpp SideEffects.cpp Successor.cpp Trait.cpp Type.cpp
)
for f in "${TABLEGEN_LIB_FILES[@]}"; do
  git show "${LLVM_COMMIT}:mlir/lib/TableGen/$f" > "$SRC_DIR/lib/TableGen/$f"
done

# MlirTblgenMain.cpp: mlir-tblgen entry point (main function). Part of the
# MLIRTblgenLib static library in mlir/lib/Tools/mlir-tblgen/CMakeLists.txt.
git show "${LLVM_COMMIT}:mlir/lib/Tools/mlir-tblgen/MlirTblgenMain.cpp" \
  > "$SRC_DIR/lib/Tools/mlir-tblgen/MlirTblgenMain.cpp"
git show "${LLVM_COMMIT}:mlir/include/mlir/Tools/mlir-tblgen/MlirTblgenMain.h" \
  > "$SRC_DIR/include/mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

# FileUtilities.cpp: openInputFile/openOutputFile helpers used by mlir-src-sharder.
# Part of the MLIRSupport library in mlir/lib/Support/CMakeLists.txt; extracted
# singly instead of pulling the whole library.
git show "${LLVM_COMMIT}:mlir/lib/Support/FileUtilities.cpp" \
  > "$SRC_DIR/lib/Support/FileUtilities.cpp"

# Headers required by the TableGen sources (mlir/TableGen/) and mlir-src-sharder
# (mlir/Support/ — FileUtilities.h, LLVM.h, LogicalResult.h, etc.).
for f in $(git ls-tree --name-only "$LLVM_COMMIT" mlir/include/mlir/TableGen/); do
  git show "${LLVM_COMMIT}:$f" > "$SRC_DIR/include/mlir/TableGen/$(basename $f)"
done
for f in $(git ls-tree --name-only "$LLVM_COMMIT" mlir/include/mlir/Support/); do
  git show "${LLVM_COMMIT}:$f" > "$SRC_DIR/include/mlir/Support/$(basename $f)"
done

# mlir-src-sharder from LLVM 19 (new tool)
git show "${LLVM19_TAG}:mlir/tools/mlir-src-sharder/mlir-src-sharder.cpp" \
  > "$SRC_DIR/mlir-src-sharder/mlir-src-sharder.cpp"

log "Source extracted."

# ── Step 3: Apply patches ────────────────────────────────────────────────────
log "Applying shard backport patch..."
cd "$SRC_DIR"
patch -p1 < "$PATCH_FILE" || fail "Patch failed to apply"
log "Patch applied."

# ── Step 4: Write CMakeLists.txt ─────────────────────────────────────────────
log "Writing CMakeLists.txt..."
cat > "$WORK_DIR/CMakeLists.txt" << 'CMAKEEOF'
cmake_minimum_required(VERSION 3.20)
project(mlir-tblgen-patched CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Optimize for size: -Os + section-level GC + strip at link time.
# Reduces mlir-tblgen from 3.7M to 2.0M, mlir-src-sharder from 1.5M to 0.5M.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions -Os -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -s")

set(LLVM_INCLUDE_DIR /usr/local/include)
set(LLVM_LIB_DIR /usr/local/lib)

include_directories(${CMAKE_SOURCE_DIR}/src/include ${LLVM_INCLUDE_DIR})
link_directories(${LLVM_LIB_DIR})

find_library(TINFO_LIB NAMES tinfo libtinfo.so.6 PATHS /usr/lib/x86_64-linux-gnu)

set(TBLGEN_SRCS
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/AttrOrTypeDefGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/AttrOrTypeFormatGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/BytecodeDialectGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/DialectGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/DirectiveCommonGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/EnumsGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/FormatGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/LLVMIRConversionGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/LLVMIRIntrinsicGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/mlir-tblgen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpClass.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpDefinitionsGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpDocGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpFormatGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpGenHelpers.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpInterfacesGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/OpPythonBindingGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/PassCAPIGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/PassDocGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/PassGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/RewriterGen.cpp
  ${CMAKE_SOURCE_DIR}/src/mlir-tblgen/SPIRVUtilsGen.cpp
)

set(TABLEGEN_LIB_SRCS
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Argument.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Attribute.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/AttrOrTypeDef.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Builder.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Class.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/CodeGenHelpers.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Constraint.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Dialect.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Format.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/GenInfo.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Interfaces.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Operator.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Pass.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Pattern.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Predicate.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Property.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Region.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/SideEffects.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Successor.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Trait.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/TableGen/Type.cpp
)

add_executable(mlir-tblgen ${TBLGEN_SRCS} ${TABLEGEN_LIB_SRCS}
  ${CMAKE_SOURCE_DIR}/src/lib/Tools/mlir-tblgen/MlirTblgenMain.cpp
)
target_link_libraries(mlir-tblgen PRIVATE
  LLVMTableGen LLVMSupport LLVMDemangle ${TINFO_LIB}
)

add_executable(mlir-src-sharder
  ${CMAKE_SOURCE_DIR}/src/mlir-src-sharder/mlir-src-sharder.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/Support/FileUtilities.cpp
)
target_link_libraries(mlir-src-sharder PRIVATE
  LLVMSupport LLVMDemangle ${TINFO_LIB}
)

install(TARGETS mlir-tblgen mlir-src-sharder DESTINATION bin)
CMAKEEOF

# ── Step 5: Build ────────────────────────────────────────────────────────────
log "Building..."
BUILD_DIR="${WORK_DIR}/build"
rm -rf "$BUILD_DIR"
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S "$WORK_DIR" -B "$BUILD_DIR"
ninja -C "$BUILD_DIR"

# ── Step 6: Verify ───────────────────────────────────────────────────────────
log "Verifying..."
"$BUILD_DIR/mlir-tblgen" --help 2>&1 | grep -q "op-shard-count" \
  || fail "mlir-tblgen missing --op-shard-count option"
test -x "$BUILD_DIR/mlir-src-sharder" \
  || fail "mlir-src-sharder not built"

# ── Step 7: Copy to output ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/mlir-tblgen" "$OUTPUT_DIR/"
cp "$BUILD_DIR/mlir-src-sharder" "$OUTPUT_DIR/"
log "Done. Binaries at: $OUTPUT_DIR"
log "  mlir-tblgen:     $OUTPUT_DIR/mlir-tblgen"
log "  mlir-src-sharder: $OUTPUT_DIR/mlir-src-sharder"
