#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CPU_NUM=$(cat /proc/stat | grep cpu[0-9] -c)
BUILD_MODE="${1:-RELEASE}"
usage() {
  echo "Usage: $0 [RELEASE|DEBUG]"
}

if [[ -z "$INSTALL_PATH" ]]; then
  echo "${RED}ERROR${NC}: Please source envsetup.sh firstly."
  exit 1
fi

DEBUG_FLAG=""
if [ "$BUILD_MODE" = "DEBUG" ]; then
    DEBUG_FLAG="-DDEBUG=ON"
elif [ "$BUILD_MODE" != "RELEASE" ]; then
    echo "Invalid build mode: $BUILD_MODE"
    usage
    exit 1
fi

# func
clean_cache() {
  # clear ppl compile cache (content-addressed; only need to wipe once per run)
  local CACHE_PATH=${PPL_CACHE_PATH}
  if [ x${CACHE_PATH} = x ]; then
    CACHE_PATH=${HOME}"/.ppl/cache"
  fi
  rm -rf "$CACHE_PATH"
}

clean_up() {
  local build_dir=${1:-$PPL_BUILD_PATH}
  rm -rf "$build_dir"
  mkdir -p "$build_dir"
}

# Ensure the target lib directory exists
mkdir -p "${INSTALL_PATH}/lib"

# Parallel ppl-compile over all *.pl files in PplBackend/src.
# Uses JOB_CORES (not CPU_NUM) so total parallelism stays bounded when
# multiple chip targets run concurrently.
# Args: chip [extra ppl-compile flags...]
ppl_compile_all() {
  local chip=$1
  shift
  ls "$DIR"/src/*.pl | xargs -n1 -P"${JOB_CORES}" -I{} \
    ppl-compile {} --chip "$chip" --mode 5 --O2 --o . "$@"
}

# Build one dynamic chip target.
# Each target uses its own build_dir and produces uniquely-named .so
# files, so multiple targets can build and install in parallel.
# Args: compile_chip cmake_chip [extra ppl-compile flags...]
build_dyn_chip() {
  local compile_chip=$1
  local cmake_chip=$2
  shift 2
  local build_dir="${PPL_BUILD_PATH}/build_${cmake_chip}_dyn"
  clean_up "$build_dir"
  (
    cd "$build_dir"
    ppl_compile_all "$compile_chip" "$@"
    cmake "$DIR" ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=OFF -DCHIP=${cmake_chip} -DCMODEL=ON -DBUILD_DIR=${build_dir}
    make -j${JOB_CORES} install
    cmake "$DIR" ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=OFF -DCHIP=${cmake_chip} -DCMODEL=OFF -DBUILD_DIR=${build_dir} -DBUILD_DYN_HOST=ON
    make -j${JOB_CORES} install
  )
}

# Build the static target (libppl_host.so).
build_static() {
  local build_dir="${PPL_BUILD_PATH}/build"
  clean_up "$build_dir"
  (
    cd "$build_dir"
    ls "$DIR"/src/*.pl | xargs -n1 -P"${JOB_CORES}" -I{} \
      ppl-compile {} --I "$PPL_PROJECT_ROOT/inc" --desc --O2 --o .
    cmake "$DIR" ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=ON -DBUILD_DIR=${build_dir}
    make -j${JOB_CORES} install
  )
}

generate_md5_list() {
  # Emit "<relpath> <md5>" lines for every regular file under the given
  # paths (files or directories), with paths relative to $DIR and sorted
  # for deterministic output.
  ( cd "$DIR" && find "$@" -type f -print0 \
      | xargs -0 -n64 -P"${CPU_NUM}" md5sum \
      | awk '{printf "%s %s\n", $2, $1}' \
      | sort )
}

has_md5_changes() {
  local md5file=$1
  shift
  local entry
  [[ -f $md5file ]] || return 0
  for entry in "$@"; do
    if ! grep -Fqx "$entry" "$md5file"; then
      echo "grep -Fqx "$entry" "$md5file""
      return 0
    fi
  done
  return 1
}
# check if files under PplBackend changed
MD5FILE=$DIR/.md5
file_changed=false
# Track every input that can affect the build outputs: PPL sources, public
# headers, the CMake build description, and the build script itself.
mapfile -t md5_list < <(generate_md5_list \
  src src_dyn include CMakeLists.txt Dynkernel.cmake)
if [ ! -f "$MD5FILE" ] || has_md5_changes "$MD5FILE" "${md5_list[@]}"; then
  file_changed=true
fi
# get latest nntoolchain version
lib_changed=false
NNTC_LIB_PATH=${PROJECT_ROOT}/third_party/nntoolchain/lib
PPL_VER_PATH=${PPL_PROJECT_ROOT}/version
LIBS=("libcmodel_bm1684x.a"  "libbm1684x_kernel_module.a"
      "libcmodel_bm1688.a"   "libbmtpulv60_kernel_module.a"
      "libcmodel_bm1690.a"   "libbm1690_kernel_module.a"
      "libcmodel_bm1690e.a"  "libbm1690e_kernel_module.a"
      "libcmodel_bm1684x2.a" "libbm1684x2_kernel_module.a")
mapfile -t libs_md5_list < <(
  for lib in "${LIBS[@]}"; do
    rel="$NNTC_LIB_PATH/$lib"
    [[ -f "$rel" ]] || continue
    md5=$(md5sum "$rel" | awk '{print $1}')
    printf '%s %s\n' "$lib" "$md5"
  done
)
# check whether the third_party/nntoolchain/lib or ppl is updated
# use a version file under the install path for cache tracking
VER_DIR="${INSTALL_PATH}/share/ppl"
mkdir -p "$VER_DIR"
VER_FILE="${VER_DIR}/.version"
if [ ! -f "$VER_FILE" ] || [ "$(head -n1 "$VER_FILE")" != "$BUILD_MODE" ] || has_md5_changes "$VER_FILE" "${libs_md5_list[@]}" || ! grep -Fxq -f "$PPL_VER_PATH" "$VER_FILE"; then
  lib_changed=true
fi
if [ "$lib_changed" = false ] && [ "$file_changed" = false ]; then
  exit 0
fi
# build ppl libs and install directly to INSTALL_PATH
PPL_BUILD_PATH=${PPL_BUILD_PATH:-"$PROJECT_ROOT/build/ppl"}
echo "rebuilding ppl in $PPL_BUILD_PATH"
mkdir -p "$PPL_BUILD_PATH"
# wipe ppl compile cache ONCE for the whole rebuild (not per-chip)
clean_cache
# --- Parallel build ---
# 7 independent targets: 5 dyn chips + 1 rvti + 1 static.
# Divide CPU cores among concurrent jobs so total processes ≈ CPU_NUM.
NUM_JOBS=7
JOB_CORES=$(( CPU_NUM / NUM_JOBS ))
[ "$JOB_CORES" -lt 1 ] && JOB_CORES=1
MAX_PARALLEL=$(( CPU_NUM / JOB_CORES ))
[ "$MAX_PARALLEL" -gt "$NUM_JOBS" ] && MAX_PARALLEL=$NUM_JOBS
[ "$MAX_PARALLEL" -lt 1 ] && MAX_PARALLEL=1
echo "PPL parallel build: ${MAX_PARALLEL} jobs × ${JOB_CORES} cores/job (${CPU_NUM} cores total)"

# FIFO-based semaphore: read -u 3 blocks until a slot is free; echo >&3 releases.
_sem_fifo=$(mktemp -u)
mkfifo "$_sem_fifo"
exec 3<>"$_sem_fifo"
rm "$_sem_fifo"
_i=0
while [ $_i -lt $MAX_PARALLEL ]; do
  echo >&3
  _i=$((_i + 1))
done

pids=()
labels=()

# dyn pio
for chip in bm1684x bm1688 bm1690 sg2260e bm1684x2; do
  read -u 3
  ( trap 'echo >&3' EXIT; build_dyn_chip "$chip" "$chip" ) &
  pids+=($!); labels+=("dyn_$chip")
done

# dyn rvti (ppl-compile uses --chip sg2260e --rv; cmake uses -DCHIP sg2260erv)
read -u 3
( trap 'echo >&3' EXIT; build_dyn_chip "sg2260e" "sg2260erv" --rv ) &
pids+=($!); labels+=("rvti_sg2260erv")

# static
read -u 3
( trap 'echo >&3' EXIT; build_static ) &
pids+=($!); labels+=("static")

# Wait for all background jobs; fail if any failed
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}" 2>/dev/null; then
    echo -e "\033[1;31mERROR: PPL build failed for ${labels[$i]} (pid ${pids[$i]})\033[0m"
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  exit 1
fi

# Check if each PPL_FW_LAYER_TYPE_T enum already exists in FW_LAYER_TYPE_T
header_file="${PROJECT_ROOT}/include/tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
ppl_header_file="${DIR}/include/ppl_dyn_fw.h"

awk '/typedef enum fw_layer_type {/,/} FW_LAYER_TYPE_T;/' "$header_file" |
grep -Eo '^[[:space:]]*PPL_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+' |
sort -u > existing_enums.tmp
missing_count=0
while read -r new_enum; do
  if ! grep -q "^[[:space:]]*${new_enum}$" existing_enums.tmp; then
    echo "$new_enum,"
    (( missing_count++ ))
  fi
done < <(
  grep -Eo '^[[:space:]]*PPL_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+' "$ppl_header_file"
)
rm -f existing_enums.tmp
if [ "$missing_count" -gt 0 ]; then
    echo -e "\n\033[1;35mTotal missing definitions: $missing_count\033[0m"
    echo -e "\033[1;31mERROR: Add above definitions to $header_file and rebuild tpu-mlir\033[0m"
    exit 1
fi
# Persist the input hashes so subsequent runs can short-circuit. We do this
# for DEBUG too; otherwise a DEBUG -> DEBUG rerun would always rebuild.
if [ "$file_changed" = true ]; then
  printf '%s\n' "${md5_list[@]}" > "$MD5FILE"
fi
# VER_FILE records the nntoolchain lib versions that the installed artifacts
# were built against, so only update it on RELEASE builds.
if [ "$lib_changed" = true ]; then
  printf '%s\n' "$BUILD_MODE" > "$VER_FILE"
  cat "$PPL_VER_PATH" >> "$VER_FILE"
  printf '%s\n' "${libs_md5_list[@]}" >> "$VER_FILE"
fi
