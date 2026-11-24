#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Compile the unmodified July 2024 SenseVoice export with TPU-MLIR."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output_dir", type=Path, help="new working directory")
    args = parser.parse_args()
    source = args.onnx.resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != "9aa707208dd3bf3b70ba1ea327c94546b3559893bd44c363a417a93eba885fc8":
        parser.error("this unpadded specialization was validated for the 2024-07-17 export only")
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    # Dynamic import needs concrete values for shape-influencing input tensors.
    # These describe the maximum shape; the compiled network has four runtime inputs.
    np.savez(work / "import_inputs.npz",
             x=np.zeros((1, 256, 560), dtype=np.float32),
             x_length=np.array([256], dtype=np.int32),
             language=np.array([3], dtype=np.int32),
             text_norm=np.array([14], dtype=np.int32))
    (work / "precision.txt").write_text(
        "/encoder/after_norm/Add_1_output_0_LayerNormalization F32\n")
    subprocess.run(["model_transform.py", "--model_name", "sensevoice", "--model_def", str(source),
                    "--input_shapes", "[[1,256,560],[1],[1],[1]]", "--dynamic",
                    "--test_input", "import_inputs.npz", "--not_inference", "--mlir", "model.mlir"],
                   cwd=work, check=True)
    subprocess.run(["model_deploy.py", "--mlir", "model.mlir", "--processor", "cv84x6",
                    "--quantize", "F16", "--quantize_table", "precision.txt", "--dynamic",
                    "--num_core", "1", "--sensevoice_unpadded", "--model", "sensevoice.bmodel"],
                   cwd=work, check=True)


if __name__ == "__main__":
    main()
