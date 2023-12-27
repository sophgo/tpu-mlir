#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
PARENT_DIR=$(dirname "$SCRIPT_PATH")

python3 $PARENT_DIR/PerfAI.doc/run_doc.py $1 $2

python3 $PARENT_DIR/PerfAI.web/perfAI_html.py $1