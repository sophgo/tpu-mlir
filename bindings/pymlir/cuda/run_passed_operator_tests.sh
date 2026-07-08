#!/usr/bin/env bash
set -uo pipefail

# Commands extracted from passed-case entries in OPERATOR_CHANGELOG.md.
# Override TPU_MLIR_ROOT if the project is not mounted at /workspace/tpu-mlir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPU_MLIR_ROOT="${TPU_MLIR_ROOT:-/workspace/tpu-mlir}"
TEST_DIR="${TPU_MLIR_ROOT}/python/test"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${SCRIPT_DIR}/acceptance_logs/${RUN_ID}"
SUMMARY_CSV="${LOG_ROOT}/summary.csv"
SUMMARY_LOG="${LOG_ROOT}/summary.log"
ENV_LOG="${LOG_ROOT}/environment.log"

if [[ ! -d "${TEST_DIR}" ]]; then
  TPU_MLIR_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  TEST_DIR="${TPU_MLIR_ROOT}/python/test"
fi

if [[ ! -d "${TEST_DIR}" ]]; then
  echo "Cannot find python/test. Set TPU_MLIR_ROOT to the tpu-mlir root." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

{
  echo "run_id: ${RUN_ID}"
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "script_dir: ${SCRIPT_DIR}"
  echo "tpu_mlir_root: ${TPU_MLIR_ROOT}"
  echo "test_dir: ${TEST_DIR}"
  echo "python: $(python --version 2>&1)"
  if command -v nvcc >/dev/null 2>&1; then
    echo
    echo "nvcc:"
    nvcc --version
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "nvidia-smi:"
    nvidia-smi
  fi
} > "${ENV_LOG}" 2>&1

cd "${TEST_DIR}" || exit 1

COMMANDS=(
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Abs"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "AdaptiveAvgPool1d"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "AddConst"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Acos"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Atanh"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Arg"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchBatchNorm"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "BatchNorm2D"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Clip"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Compare"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "CompareCst"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ConstantFillDyn"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Conv3d"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Correlation"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Cos"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Cosh"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "CumSum"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Div"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "DivConst"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Elu"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Erf"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Exp"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Expand"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Floor"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GatherElements"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GatherND"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchGroupNorm"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GRU"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchHardSigmoid"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchHardSwish"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchInstanceNorm"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchIndexPut"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchInterp"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LRN"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LSTM"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LeakyRelu"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Log"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchLayerNormTrain"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LogB"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LogicalAnd"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reciprocal"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchRMSNorm"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Relu"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reshape"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reverse"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchNonZero"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShapeSlice"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShapeCast"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Round"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Rsqrt"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchNms"'
  'python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchRoiAlign"'
  'python test_torch.py --chip bm1684x --mode f32 --cuda --case "Attention"'
  'python test_torch.py --chip bm1684x --mode f32 --cuda --case "FAttention"'
)

FAILED=()
TOTAL=${#COMMANDS[@]}

echo "index,case,status,duration_seconds,log_file,command" > "${SUMMARY_CSV}"

for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$i]}"
  case_name="$(sed -n 's/.*--case "\([^"]*\)".*/\1/p' <<< "${cmd}")"
  if [[ -z "${case_name}" ]]; then
    case_name="case_$((i + 1))"
  fi
  safe_case_name="${case_name//[^A-Za-z0-9_.-]/_}"
  log_file="${LOG_ROOT}/$(printf "%03d" "$((i + 1))")_${safe_case_name}.log"
  start_time="$(date +%s)"

  echo
  echo "[$((i + 1))/${TOTAL}] ${cmd}"
  echo "  log: ${log_file}"

  if {
    echo "index: $((i + 1))/${TOTAL}"
    echo "case: ${case_name}"
    echo "command: ${cmd}"
    echo "start: $(date)"
    echo
    bash -lc "${cmd}"
  } > "${log_file}" 2>&1; then
    end_time="$(date +%s)"
    duration="$((end_time - start_time))"
    {
      echo
      echo "end: $(date)"
      echo "status: PASSED"
      echo "duration_seconds: ${duration}"
    } >> "${log_file}"
    echo "[PASSED] ${cmd}"
    echo "$((i + 1)),${case_name},PASSED,${duration},${log_file},${cmd}" >> "${SUMMARY_CSV}"
  else
    end_time="$(date +%s)"
    duration="$((end_time - start_time))"
    {
      echo
      echo "end: $(date)"
      echo "status: FAILED"
      echo "duration_seconds: ${duration}"
    } >> "${log_file}"
    echo "[FAILED] ${cmd}" >&2
    FAILED+=("${cmd}")
    echo "$((i + 1)),${case_name},FAILED,${duration},${log_file},${cmd}" >> "${SUMMARY_CSV}"
  fi
done

PASSED_COUNT=$((TOTAL - ${#FAILED[@]}))

echo
{
  echo "Run ID: ${RUN_ID}"
  echo "Total: ${TOTAL}"
  echo "Passed: ${PASSED_COUNT}"
  echo "Failed: ${#FAILED[@]}"
  echo "Environment log: ${ENV_LOG}"
  echo "Summary CSV: ${SUMMARY_CSV}"
  echo "Case logs: ${LOG_ROOT}"
} | tee "${SUMMARY_LOG}"

if (( ${#FAILED[@]} > 0 )); then
  echo
  {
    echo
    echo "Failed commands:"
    printf '  %s\n' "${FAILED[@]}"
  } | tee -a "${SUMMARY_LOG}" >&2
  exit 1
fi
