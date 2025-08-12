#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CPU_NUM=$(cat /proc/stat | grep cpu[0-9] -c)
BUILD_MODE="${1:-RELEASE}"
usage() {
  echo "Usage: $0 [RELEASE|DEBUG] [force]"
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
FORCE_BUILD=false
if [ -n "$2" ]; then
    if [ "$2" = "force" ]; then
       FORCE_BUILD=true
    else
        echo "Invalid param: $2"
        usage
        exit 1
    fi
fi
# func
clean_up() {
  local build_dir=${1:-build}
  # clear cache
	CACHE_PATH=${PPL_CACHE_PATH}
  if [ x${CACHE_PATH} = x ];then
    CACHE_PATH=${HOME}"/.ppl/cache"
  fi
  rm -rf $CACHE_PATH
  rm -rf $build_dir
  mkdir -p $build_dir
}

generate_md5_list() {
  for dir in "$@"; do
    while IFS= read -r -d '' file; do
      rel=${file#"$dir"/}
      md5=$(md5sum "$file" | awk '{print $1}')
      printf '%s %s\n' "$rel" "$md5"
    done < <(find "$dir" -type f -print0)
  done
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
mapfile -t md5_list < <(generate_md5_list "$DIR/src" "$DIR/src_dyn")
# printf '>> %s\n' "${md5_list[@]}"
if [ ! -f "$MD5FILE" ] || has_md5_changes "$MD5FILE" "${md5_list[@]}"; then
  file_changed=true
fi
# get latest nntoolchain version
lib_changed=false
NNTC_LIB_PATH=${PROJECT_ROOT}/third_party/nntoolchain/lib
PPL_VER_PATH=${PROJECT_ROOT}/third_party/ppl/version
VER_FILE=${PROJECT_ROOT}/third_party/nntoolchain/ppl/version
LIBS=("libcmodel_1684x.a" "libbm1684x_kernel_module.a" "libcmodel_1688.a" "libbmtpulv60_kernel_module.a")
mapfile -t libs_md5_list < <(
  for lib in "${LIBS[@]}"; do
    rel="$NNTC_LIB_PATH/$lib"
    [[ -f "$rel" ]] || continue
    md5=$(md5sum "$rel" | awk '{print $1}')
    printf '%s %s\n' "$lib" "$md5"
  done
)
# check whether the third_party/nntoolchain/lib or ppl is updated
if [ ! -f "$VER_FILE" ] || has_md5_changes "$VER_FILE" "${libs_md5_list[@]}" || ! grep -Fxq -f "$PPL_VER_PATH" "$VER_FILE"; then
  lib_changed=true
fi
if [ "$FORCE_BUILD" = false ] &&
   [ "$lib_changed" = false ] &&
   [ "$file_changed" = false ]; then
  exit 0
fi
# build third_party/nntoolchain/ppl lib
echo "rebuilding ppl..."
pushd "$DIR"
# dyn
chips=("bm1684x" "bm1688")
for chip in "${chips[@]}"; do
  build_dir="build_${chip}_dyn"
  clean_up "$build_dir"
  pushd "$build_dir"
  for file in `ls ../src/*.pl`
  do
    ppl-compile $file --chip $chip --mode 5 --O2 --o .
  done
  cmake ../ ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=OFF -DCHIP=${chip} -DCMODEL=ON -DBUILD_DIR=${build_dir}
  make -j${CPU_NUM} install
  cmake ../ ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=OFF -DCHIP=${chip} -DCMODEL=OFF -DBUILD_DIR=${build_dir} -DBUILD_DYN_HOST=ON
  make -j${CPU_NUM} install
  popd
done

# static
clean_up
pushd build
for file in `ls ../src/*.pl`
do
  ppl-compile $file --I $PPL_PROJECT_ROOT/inc  --desc --O2 --o .
done
cmake ../ ${DEBUG_FLAG} -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" -DBUILD_STATIC=ON
make install
popd

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
# write nntc and ppl version to VER_FILE
if [ "$BUILD_MODE" = "DEBUG" ];then
  exit 0
fi
if [ "$file_changed" = true ]; then
  printf '%s\n' "${md5_list[@]}" > "$MD5FILE"
fi
if [ "$lib_changed" = true ]; then
  printf '%s\n' "$BUILD_MODE" > "$VER_FILE"
  cat "$PPL_VER_PATH" >> "$VER_FILE"
  printf '%s\n' "${libs_md5_list[@]}" >> "$VER_FILE"
fi
