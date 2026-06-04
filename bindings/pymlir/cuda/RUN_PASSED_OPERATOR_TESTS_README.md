# CUDA 算子验收脚本说明

本文档说明 `run_passed_operator_tests.sh` 的用途、运行方式和输出日志结构。该脚本用于批量运行 `OPERATOR_CHANGELOG.md` 中已标记为 `（通过）` 的 CUDA 算子测试 case。

## 脚本用途

`run_passed_operator_tests.sh` 会按固定顺序执行脚本内 `COMMANDS` 数组中的 51 个算子测试命令。当前这些命令来自 `OPERATOR_CHANGELOG.md` 中带全角 `（通过）` 标记的测试 case。

脚本执行每个 case 时会：

- 打印当前执行进度和命令。
- 为每个 case 生成独立日志文件。
- 记录每个 case 的通过/失败状态和耗时。
- 生成总汇总文件，便于项目验收查看。
- 记录运行环境信息，包括 Python、CUDA 编译器和 GPU 信息。

## 运行前提

运行前需要确保：

- 已完成 tpu-mlir 项目编译。
- CUDA 运行环境可用。
- 当前环境可以执行 `python/test/test_onnx.py` 和 `python/test/test_torch.py`。
- `TPU_MLIR_ROOT` 指向 tpu-mlir 项目根目录，或者项目挂载在默认路径 `/workspace/tpu-mlir`。

脚本会优先使用：

```bash
/workspace/tpu-mlir/python/test
```

如果该目录不存在，脚本会尝试根据脚本所在路径自动推导项目根目录。

## 运行方式

进入脚本所在目录：

```bash
cd /home/xjy/C-C++/DATA_docker/tpu-mlir-master/bindings/pymlir/cuda
```

直接运行：

```bash
bash run_passed_operator_tests.sh
```

如果需要显式指定 tpu-mlir 项目根目录：

```bash
TPU_MLIR_ROOT=/home/xjy/C-C++/DATA_docker/tpu-mlir-master bash run_passed_operator_tests.sh
```

如果脚本已经有执行权限，也可以运行：

```bash
./run_passed_operator_tests.sh
```

如果没有执行权限，可以先添加：

```bash
chmod +x run_passed_operator_tests.sh
```

## 输出目录

每次运行脚本都会在脚本同目录下创建一个新的日志目录：

```text
acceptance_logs/<运行时间戳>/
```

示例：

```text
acceptance_logs/20260603_153000/
```

该目录保存本次验收运行的全部输出。不同运行批次使用不同时间戳目录，不会覆盖历史日志。

## 输出文件说明

一次完整运行会生成如下文件：

```text
acceptance_logs/<运行时间戳>/
  environment.log
  summary.csv
  summary.log
  001_Abs.log
  002_AdaptiveAvgPool1d.log
  ...
  051_FAttention.log
```

### environment.log

`environment.log` 记录本次验收运行的环境信息，主要包括：

- 运行时间。
- 主机名。
- 脚本所在路径。
- `TPU_MLIR_ROOT`。
- 实际测试目录 `python/test`。
- Python 版本。
- `nvcc --version` 输出。
- `nvidia-smi` 输出。

该文件用于说明验收结果是在什么机器、什么 CUDA/GPU 环境下产生的。

### summary.log

`summary.log` 是人可读的总汇总文件，内容包括：

- 本次运行 ID。
- 总 case 数。
- 通过 case 数。
- 失败 case 数。
- 环境日志路径。
- CSV 汇总路径。
- case 日志目录路径。

验收时可以优先查看该文件。

查看方式：

```bash
cat acceptance_logs/<运行时间戳>/summary.log
```

### summary.csv

`summary.csv` 是机器可读的结果汇总文件，字段如下：

```text
index,case,status,duration_seconds,log_file,command
```

字段含义：

- `index`: case 执行序号。
- `case`: case 名称。
- `status`: 执行结果，取值为 `PASSED` 或 `FAILED`。
- `duration_seconds`: 执行耗时，单位为秒。
- `log_file`: 该 case 对应的独立日志文件路径。
- `command`: 实际执行的测试命令。

查看方式：

```bash
cat acceptance_logs/<运行时间戳>/summary.csv
```

如果只想查看失败项：

```bash
grep ',FAILED,' acceptance_logs/<运行时间戳>/summary.csv
```

### 单个 case 日志

每个算子 case 都有一个独立日志文件，例如：

```text
001_Abs.log
051_FAttention.log
```

单个 case 日志中包含：

- case 序号。
- case 名称。
- 实际执行命令。
- 开始时间。
- 测试命令的完整输出。
- 结束时间。
- 最终状态。
- 执行耗时。

如果某个 case 在 `summary.csv` 中显示 `FAILED`，应优先打开对应的 `log_file` 查看失败原因。

## 退出码说明

脚本退出码用于表示本次验收是否通过：

- `0`: 51 个 case 全部通过。
- `1`: 至少有一个 case 失败，或脚本无法找到 `python/test` 目录。

如果有失败 case，脚本会在终端和 `summary.log` 中列出失败命令。

## 验收建议

项目验收时建议保留以下文件：

```text
environment.log
summary.log
summary.csv
全部单 case 日志
```

这些文件共同构成本次 51 个 CUDA 算子测试的验收记录。

推荐验收顺序：

1. 查看 `summary.log`，确认总数、通过数和失败数。
2. 查看 `summary.csv`，确认每个 case 的状态。
3. 如果存在失败项，根据 `summary.csv` 中的 `log_file` 打开对应日志。
4. 查看 `environment.log`，确认运行环境符合项目验收要求。
