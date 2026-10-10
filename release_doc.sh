#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python docs/generate_operation.py ${DIR}/build/supported_ops.rst

# Parallelize doc builds with a bounded job pool.
#
# 2 doc dirs (quick_start, developer_manual) × 2 langs (en, zh) = 4 jobs.
# Each job builds latex→xelatex→PDF then HTML sequentially, so latex/html
# never race on the same build_${lang}/ directory.
#
# Cap concurrent jobs (default 2) to avoid OOM: a full release build already
# consumes a lot of memory/cache, and 8 concurrent sphinx+xelatex jobs have
# been observed to get oom-killed, leaving no release tarball.
#
# Override with DOC_MAX_JOBS=N if needed.
CPU_NUM=$(cat /proc/stat | grep cpu[0-9] -c)
DOC_DIRS=("quick_start" "developer_manual")
LANGS=("en" "zh")
MAX_JOBS=${DOC_MAX_JOBS:-2}
SPHINX_J=$(( CPU_NUM / MAX_JOBS ))
[ "$SPHINX_J" -lt 1 ] && SPHINX_J=1

# sphinx-build calls locale.setlocale(LC_ALL, '') and crashes if LANG/LC_ALL
# is unset or unsupported (common in non-interactive release shells).
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

LOG_DIR="${DIR}/build/doc_logs"
mkdir -p "$LOG_DIR"
rm -rf "${LOG_DIR:?}"/*

# build_one: latex PDF then HTML for one (doc_dir, lang) pair.
build_one() {
  set -e
  local doc_dir=$1 lang=$2 srcdir builddir module_name pdf_name logfile
  srcdir="${DIR}/docs/${doc_dir}/source_${lang}"
  builddir="${DIR}/docs/${doc_dir}/build_${lang}"
  module_name="TPU-MLIR"
  pdf_name="tpu-mlir_$([ "$doc_dir" = "quick_start" ] && echo "quick_start" || echo "technical_manual")_${lang}"
  logfile="${LOG_DIR}/${doc_dir}_${lang}.log"

  {
    echo "=== sphinx latex ${doc_dir}/${lang} ==="
    sphinx-build -M latex "$srcdir" "$builddir" -j "$SPHINX_J" -d "$builddir/doctrees_latex"
    echo "=== xelatex ${doc_dir}/${lang} ==="
    cd "$builddir/latex"
    # nonstopmode: never prompt for missing input (would hang release forever)
    xelatex -interaction=nonstopmode ${module_name}.tex
    xelatex -interaction=nonstopmode ${module_name}.tex
    mv ${module_name}.pdf ../"${pdf_name}.pdf"
    rm -rf "$builddir/latex"
    cd "$DIR"
    echo "=== sphinx html ${doc_dir}/${lang} ==="
    sphinx-build -M html "$srcdir" "$builddir" -j "$SPHINX_J" -d "$builddir/doctrees_html"
  } >"$logfile" 2>&1
}

# Clean build dirs for both docs before launching any jobs.
for d in "${DOC_DIRS[@]}"; do
  make -C "${DIR}/docs/${d}" clean > /dev/null 2>&1
done

# Build task list, then run with xargs -P for reliable concurrency limiting.
task_file="${LOG_DIR}/tasks.txt"
: >"$task_file"
for d in "${DOC_DIRS[@]}"; do
  for l in "${LANGS[@]}"; do
    echo "${d} ${l}" >>"$task_file"
  done
done

# Export helpers for the xargs child bash.
export DIR SPHINX_J LOG_DIR LC_ALL LANG
export -f build_one

set +e
xargs -P "$MAX_JOBS" -n 2 -a "$task_file" bash -c 'build_one "$1" "$2"' _
xargs_status=$?
set -e

if [ "$xargs_status" -ne 0 ]; then
  echo -e "\033[1;31mERROR: doc build failed (xargs status ${xargs_status})\033[0m"
  for d in "${DOC_DIRS[@]}"; do
    for l in "${LANGS[@]}"; do
      pdf="tpu-mlir_$([ "$d" = "quick_start" ] && echo "quick_start" || echo "technical_manual")_${l}.pdf"
      if [ ! -f "${DIR}/docs/${d}/build_${l}/${pdf}" ]; then
        echo -e "\033[1;31mERROR: missing ${d}/${l} -> ${pdf}\033[0m"
        echo "---- tail ${LOG_DIR}/${d}_${l}.log ----"
        tail -n 40 "${LOG_DIR}/${d}_${l}.log" || true
      fi
    done
  done
  exit 1
fi

# All jobs succeeded — collect outputs into build/docs/.
mkdir -p ${DIR}/build/docs/quick_start_zh
cp -rf ${DIR}/docs/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${DIR}/build/docs/quick_start_zh/
cp -rf ${DIR}/docs/quick_start/build_zh/html ${DIR}/build/docs/quick_start_zh

mkdir -p ${DIR}/build/docs/quick_start_en
cp -rf ${DIR}/docs/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${DIR}/build/docs/quick_start_en/
cp -rf ${DIR}/docs/quick_start/build_en/html ${DIR}/build/docs/quick_start_en

mkdir -p ${DIR}/build/docs/developer_manual_zh
cp -f ${DIR}/docs/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${DIR}/build/docs/developer_manual_zh/
cp -rf ${DIR}/docs/developer_manual/build_zh/html ${DIR}/build/docs/developer_manual_zh/

mkdir -p ${DIR}/build/docs/developer_manual_en
cp -f ${DIR}/docs/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${DIR}/build/docs/developer_manual_en/
cp -rf ${DIR}/docs/developer_manual/build_en/html ${DIR}/build/docs/developer_manual_en/

if [[ ! -z "$INSTALL_PATH" ]]; then
# only install pdf
mkdir -p ${INSTALL_PATH}/docs
cp -f ${DIR}/docs/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR快速入门指南.pdf"
cp -f ${PROJECT_ROOT}/docs/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR_Quick_Start.pdf"
cp -f ${PROJECT_ROOT}/docs/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR开发参考手册.pdf"
cp -f ${PROJECT_ROOT}/docs/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR_Technical_Reference_Manual.pdf"
fi
