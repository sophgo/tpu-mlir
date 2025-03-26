# 命令行工具设计方案

根据你的需求，我可以设计一个类似 npm 的命令行工具，实现在初始化后创建命名空间并执行模型处理相关命令。以下是具体设计方案：

## 基本设计

```yaml:design2.yml
// ... existing code ...

cli:
  name: "bmodel"
  description: "命令行工具用于模型转换和部署"
  namespace:
    create: "init [name] [--template <template_name>]"
    description: "创建新的工作空间，可选择模板"

  commands:
    - name: "transform"
      shortcut: "t"
      description: "模型转换"
      action: "mlir_transform"

    - name: "calibrate"
      shortcut: "c"
      description: "模型校准"
      action: "mlir_calibration"

    - name: "deploy"
      shortcut: "d"
      description: "模型部署"
      action: "deploy"

    - name: "test"
      shortcut: "h"
      description: "运行测试"
      action: "harness"

    - name: "all"
      description: "执行完整流程"
      actions: ["mlir_transform", "mlir_calibration", "deploy", "harness"]

    - name: "ls"
      description: "列出当前工作空间的配置和结果"

    - name: "config"
      description: "设置或查看配置"
      subcommands:
        - name: "set <key> <value>"
          description: "设置配置项"
        - name: "get [key]"
          description: "获取配置项"
        - name: "list"
          description: "列出所有配置"
// ... existing code ...
```

## 实现思路

1. **命名空间初始化**：
   - 使用 `bmodel init myproject` 创建新的工作空间
   - 自动生成配置文件和目录结构
   - 支持模板选择，如 `bmodel init myproject --template resnet`

2. **命令执行方式**：
   - 在工作空间内运行 `bmodel transform --shape shape1 --dtype F16`
   - 支持简写：`bmodel t --shape shape1 --dtype F16`
   - 通过 `config` 设置默认参数，减少重复输入

3. **工作流支持**：
   - 使用 `bmodel all` 执行完整工作流
   - 可自定义工作流：`bmodel config set workflow.custom "transform,calibrate,deploy"`

4. **配置文件**：
   - 在初始化时自动生成 `bmodel.json`/`bmodel.yml` 存储配置
   - 包含模型信息、目标平台、数据类型等信息

## 使用示例

```bash
# 初始化工作空间
bmodel init resnet50 --template resnet

# 查看配置
bmodel config list

# 设置默认参数
bmodel config set shape shape1
bmodel config set dtype F16
bmodel config set target BM1684X

# 执行模型转换
bmodel transform

# 执行校准
bmodel calibrate

# 部署模型
bmodel deploy

# 测试模型
bmodel test

# 或者直接执行完整流程
bmodel all

# 查看结果
bmodel ls
```

这种设计的优点是：
1. 命令简洁直观
2. 工作空间隔离，避免混淆
3. 可设置默认参数，减少重复输入
4. 支持完整工作流和自定义流程
5. 结构化配置，便于扩展

是否需要我详细说明某个特定部分的实现方式？


```sh

# 初始化 workspace
ppm init xx.py::MyModule
ppm init xx.onnx
ppm init Qwen/Qwen2-7B --template huggingface

# information
ppm info
ppm shape 1x3x224x224
ppm dtype <dtype> # BM1684X,BM1688-2,BM1688,CV186X,BM1690-1,BM1690-8

# ppm compile|c <dtype> <target> [core_num] <shape>
## check if caliset is valid
ppm compile BM1684X F32
ppm c BM1684X F32
ppm compile BM1688-2 F16 # = ppm compile core2 F16 BM1688

## subwork
ppm transform
ppm verify top
ppm make caliset
ppm calibration # need ppm make caliset
ppm deploy
ppm verify tpu
ppm verify bmodel


## tpuc work
ppm list
ppm tpuc stage1
ppm tpuc stage2
ppm tpuc stage3 --dry

# 出问题打包提 jira
ppm pack

# debug 功能

ppm bmodel_dis
ppm tdb
ppm cmodel


#
```

## API

```python
from tpu_mlir.compile import cli

cli.info()
cli.shape()
cli.transform()
cli.verify()

cli.tpuc.
```

1. 将 tpu-perf 推理功能合入 tpu-mlir，将 tpu-perf 的 harness 部份单独打包（tpu-infer）
2. bmrt_test 切换为 tpuv7-runtime（重写 tpuv7-runtime）


## 工具链 2.0

 - 功能整合，代码 review，设定规范
