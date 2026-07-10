#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python docs/generate_operation.py ${DIR}/build/supported_ops.rst

# Parallelize doc builds.
#
# 2 doc dirs (quick_start, developer_manual) × 2 langs (en, zh) = 4 (doc, lang)
# pairs. For each pair we launch 2 jobs: one builds latex→xelatex→PDF, the
# other builds HTML. Total 8 concurrent jobs.
#
# Directory isolation: each (doc_dir, lang) pair writes to its own
# build_${lang}/ directory. The latex job writes build_${lang}/latex/ and
# doctrees_latex/; the html job writes build_${lang}/html/ and doctrees_html/.
# Separate -d paths prevent the two sphinx-build processes from racing on the
# shared doctrees pickle cache.
#
# SPHINX_J = CPU_NUM / 8 so total sphinx threads ≈ CPU_NUM (8 jobs × 1/8 cores).
# xelatex is single-threaded and runs inside the latex job after sphinx finishes.
CPU_NUM=$(cat /proc/stat | grep cpu[0-9] -c)
DOC_DIRS=("quick_start" "developer_manual")
LANGS=("en" "zh")
SPHINX_J=$(( CPU_NUM / 8 ))
[ "$SPHINX_J" -lt 1 ] && SPHINX_J=1

pids=()
labels=()

# build_latex_pdf: sphinx-build → xelatex (×2 for cross-refs) → mv PDF → cleanup.
# Runs in a subshell with set -e so a sphinx or xelatex failure exits immediately
# instead of continuing into cd/xelatex/mv on missing files.
build_latex_pdf() {
  local doc_dir=$1 lang=$2 srcdir builddir module_name pdf_name
  srcdir="${DIR}/docs/${doc_dir}/source_${lang}"
  builddir="${DIR}/docs/${doc_dir}/build_${lang}"
  module_name="TPU-MLIR"
  pdf_name="tpu-mlir_$([ "$doc_dir" = "quick_start" ] && echo "quick_start" || echo "technical_manual")_${lang}"
  sphinx-build -M latex "$srcdir" "$builddir" -j "$SPHINX_J" -d "$builddir/doctrees_latex" > /dev/null 2>&1
  cd "$builddir/latex"
  xelatex ${module_name}.tex > /dev/null 2>&1
  xelatex ${module_name}.tex > /dev/null 2>&1
  mv ${module_name}.pdf ../"${pdf_name}.pdf"
  rm -rf "$builddir/latex"
}

# build_html: sphinx-build HTML output. Separate doctrees dir from latex.
build_html() {
  local doc_dir=$1 lang=$2 srcdir builddir
  srcdir="${DIR}/docs/${doc_dir}/source_${lang}"
  builddir="${DIR}/docs/${doc_dir}/build_${lang}"
  sphinx-build -M html "$srcdir" "$builddir" -j "$SPHINX_J" -d "$builddir/doctrees_html" > /dev/null 2>&1
}

# Clean build dirs for both docs before launching any parallel jobs.
for d in "${DOC_DIRS[@]}"; do
  make -C "${DIR}/docs/${d}" clean > /dev/null 2>&1
done

# Launch 8 jobs: for each (doc_dir, lang), one latex+PDF job and one HTML job.
# set -e in each subshell ensures fail-fast (sphinx/xelatex failure stops the
# subshell immediately, detected by the wait loop below).
for d in "${DOC_DIRS[@]}"; do
  for l in "${LANGS[@]}"; do
    ( set -e; build_latex_pdf "$d" "$l" ) &
    pids+=($!); labels+=("latex_${d}_${l}")
    ( set -e; build_html "$d" "$l" ) &
    pids+=($!); labels+=("html_${d}_${l}")
  done
done

# Wait for all jobs. Report failures but reap all zombies first.
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}" 2>/dev/null; then
    echo -e "\033[1;31mERROR: doc build failed for ${labels[$i]} (pid ${pids[$i]})\033[0m"
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
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
