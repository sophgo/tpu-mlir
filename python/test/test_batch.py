#!/usr/bin/env python3
"""批量测试所有验证通过的 CUDA 算子，日志保存到文件"""
import subprocess
import sys
import datetime

MODE = sys.argv[1] if len(sys.argv) > 1 else "f32"
LOG_FILE = f"cuda_test_{MODE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

OPERATORS = [
    "Erf",
    "Exp",
    "Elu",
    "Clip",
    "DivConst",
    "Einsum",
    "MaskedFill",
    "MaskRCNNBboxPooler",
    "MaskRCNNGetBboxB",
    "MaskRCNNMaskPooler",
    "MatchTemplate",
    "Max",
    "MaxPoolingIndicesBwd",
    "MaxPoolWithMask",
    "MaxUnpool",
    "MeanRstd",
    "MeanStdScale",
    "MeshGrid",
    "Min",
    "Mish",
    "Mod",
    "Pack",
    "Pow",
    "QuantizeLinear",
    "ScaleLut",
    "ScatterElements",
    "ScatterND",
    "SelectiveScan",
    "Shape",
    "ShapeSlice",
    "ShuffleChannel",
    "Sign",
    "Sin",
    "Sinh",
    "SliceAxis",
    "Softplus",
    "Softsign",
    "Sort",
    "Split",
    "Sqrt",
    "StridedSlice",
    "SwapChannel",
    "Swish",
    "Tan",
    "TopK",
    "Trilu",
    "Unpack",
    "Where",
]

total = len(OPERATORS)
passed = 0
failed = 0
failed_ops = []

with open(LOG_FILE, "w") as log:
    header = f"{'='*60}\n  批量测试 {total} 个算子 ({MODE})\n  开始: {datetime.datetime.now()}\n{'='*60}\n\n"
    log.write(header)
    print(header)

    for op in OPERATORS:
        print(f">>> 测试: {op}")
        log.write(f">>> 测试: {op}\n")
        result = subprocess.run(
            ["python", "test_onnx.py", "--chip", "bm1684x", "--mode", MODE, "--cuda", "--case", op],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        log.write(output)
        log.write("\n" + "-"*60 + "\n\n")

        if result.returncode == 0 and "Success" in output:
            print(f"    [PASS] {op}")
            passed += 1
        else:
            print(f"    [FAIL] {op} (exit={result.returncode})")
            failed += 1
            failed_ops.append(op)
        print()

    summary = f"{'='*60}\n  结果: {passed} 通过 / {failed} 失败 / {total} 总计\n  结束: {datetime.datetime.now()}\n"
    if failed_ops:
        summary += f"  失败列表: {', '.join(failed_ops)}\n"
    summary += f"  日志: {LOG_FILE}\n{'='*60}\n"
    log.write(summary)
    print(summary)
