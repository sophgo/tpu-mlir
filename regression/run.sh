#!/bin/bash
# Per-commit / nightly test launcher used by CI.
#
# Usage: regression/run.sh <op|script|model|check_utests|cuda>
#
# Each branch is a single CI job; "all" is handled by regression/run_all.sh.
set -euo pipefail

pip list

# Rebuild the customlayer plugin/backend/firmware once per CI job that needs it.
_build_customlayer() {
    source "$PROJECT_ROOT/third_party/customlayer/envsetup.sh"
    rebuild_custom_plugin
    rebuild_custom_backend
    rebuild_custom_firmware_cmodel bm1684x
    rebuild_custom_firmware_cmodel bm1688
}

case "${1:-}" in
    op)
        echo "::RUN operation set 0 test."
        _build_customlayer
        "$REGRESSION_PATH/main_entry.py" --test_type basic --test_set torch
        ;;
    script)
        echo "::RUN check tests and unit tests."
        cmake --build "${BUILD_PATH}" --target check-tpumlir
        echo "::RUN script and model test."
        pip show gguf >/dev/null 2>&1 || pip install gguf==0.19.0
        "$REGRESSION_PATH/main_entry.py" --test_type basic --test_set script model
        ;;
    model)
        echo "::RUN operation set 1 test."
        "$REGRESSION_PATH/main_entry.py" --test_type basic --test_set onnx
        ;;
    check_utests)
        echo "::RUN check tests and unit tests."
        cmake --build "${BUILD_PATH}" --target check-tpumlir
        ;;
    cuda)
        echo "::RUN CUDA test."
        # "$REGRESSION_PATH/main_entry.py" --test_type basic --test_set cuda
        ;;
    *)
        echo "::RUN Other test."
        ;;
esac

