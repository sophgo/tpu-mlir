#!/usr/bin/env python3
import subprocess
import sys
import os
from datetime import datetime

cases = [
    "Einsum",
    "Max", "MeanRstd", "MeanStdScale", "Min", "Mish",
    "Pack", "Pow1", "ScatterElements", "ScatterND",
    "Shape", "ShuffleChannel",
    "Sign", "Sin", "Sinh", "SliceAxis",
    "Softplus", "Softsign", "Split", "Sqrt",
    "StridedSlice", "SwapChannel", "Swish",
    "Tan", "Tanh", "TopK", "Trilu",
    "Unpack", "Where",
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"batch_gpu_test_{timestamp}.log")

with open(log_file, "w") as f:
    f.write(f"Batch GPU Test started at {datetime.now()}\n")
    f.write(f"Total: {len(cases)} cases\n")
    f.write("=" * 60 + "\n\n")

    passed = []
    failed = []

    for i, case in enumerate(cases, 1):
        msg = f"[{i}/{len(cases)}] Testing: {case}"
        print(msg)
        f.write(msg + "\n")
        f.flush()

        result = subprocess.run(
            ["python", "test_onnx.py", "--chip", "bm1684x", "--mode", "f32", "--cuda", "--case", case],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=600,
        )

        f.write(result.stdout)
        f.write(result.stderr)
        f.write(f"\nReturn code: {result.returncode}\n")

        if result.returncode == 0 and "TEST " + case + " Success" in result.stdout:
            msg = f"  => PASS"
            passed.append(case)
        else:
            msg = f"  => FAIL (rc={result.returncode})"
            failed.append(case)

        print(msg)
        f.write(msg + "\n\n")
        f.write("=" * 60 + "\n")
        f.flush()

    f.write(f"\nSummary: {len(passed)} passed, {len(failed)} failed\n")
    f.write(f"Passed: {passed}\n")
    f.write(f"Failed: {failed}\n")
    f.write(f"Finished at {datetime.now()}\n")

print(f"\nLog saved to: {log_file}")
print(f"Passed ({len(passed)}): {passed}")
print(f"Failed ({len(failed)}): {failed}")
