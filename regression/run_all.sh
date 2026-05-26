#!/bin/bash
# Nightly "run everything" entry point. The per-commit CI uses regression/run.sh.
set -euo pipefail

pip list

# Rebuild custom layer plugin/backend/firmware before kicking off the full set.
source "$PROJECT_ROOT/third_party/customlayer/envsetup.sh"
rebuild_custom_plugin
rebuild_custom_backend
rebuild_custom_firmware_cmodel bm1684x
rebuild_custom_firmware_cmodel bm1688

"$REGRESSION_PATH/main_entry.py" --test_type all
