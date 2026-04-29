<p align="center">
  <img src="./docs/assets/sophgo_cover.png" alt="TPU-MLIR" />
</p>

<h1 align="center">TPU-MLIR</h1>

<p align="center">
  <em>面向 TPU 的开源 MLIR 机器学习编译器。</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/tpu_mlir/"><img src="https://img.shields.io/pypi/v/tpu_mlir.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/tpu_mlir/"><img src="https://img.shields.io/pypi/pyversions/tpu_mlir.svg" alt="Python"></a>
  <a href="https://github.com/sophgo/tpu-mlir/stargazers"><img src="https://img.shields.io/github/stars/sophgo/tpu-mlir?style=social" alt="Stars"></a>
  <a href="https://github.com/sophgo/tpu-mlir/issues"><img src="https://img.shields.io/github/issues/sophgo/tpu-mlir.svg" alt="Issues"></a>
  <a href="https://arxiv.org/abs/2210.15016"><img src="https://img.shields.io/badge/arXiv-2210.15016-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README_cn.md">简体中文</a> ·
  <a href="https://tpumlir.org/quick_start/index.html">快速入门</a> ·
  <a href="https://tpumlir.org/developer_manual/index.html">开发手册</a> ·
  <a href="https://github.com/sophgo/tpu-mlir/issues">Issues</a>
</p>

---

## ✨ 项目简介

**TPU-MLIR** 提供完整的工具链，将主流框架下预训练的神经网络转换为可在 SOPHGO TPU 上高效运行的 `bmodel` 文件。基于 [MLIR](https://mlir.llvm.org/) 构建，提供统一 IR、清晰的下降流水线，以及完善的量化、校准与部署工具。

```
┌──────────────────┐    model_transform.py    ┌──────────┐    model_deploy.py    ┌──────────┐
│ ONNX / PyTorch / │ ───────────────────────► │   MLIR   │ ────────────────────► │  bmodel  │
│  TFLite / Caffe  │      (前端导入)          │  (Top →  │  (lowering、量化、    │  TPU 部署 │
│   HuggingFace    │                          │   Tpu)   │   layer-group …)      │          │
└──────────────────┘                          └──────────┘                       └──────────┘
```

## 🚀 主要特性

- **多框架前端** —— PyTorch、ONNX、TFLite、Caffe（其他框架请先导出为 ONNX）。
- **LLM 一键转换** —— 通过 `llm_convert.py` 直接编译 HuggingFace 大模型（Qwen、MiniCPM-V 等）。
- **完整量化工具链** —— F32 / BF16 / F16 / INT8（对称 & 非对称），兼容 AWQ / GPTQ / AutoRound，支持校准与 QAT。
- **基于 MLIR 的流水线** —— 清晰的 Top / Tpu 方言、模式重写、layer-group 内存规划。
- **完善的辅助工具** —— `model_runner`、`model_tool`、精度验证、模拟器、可视化工具。
- **中英双语文档与活跃社区** —— 提供论文、技术手册与系列视频教程。


## 📚 目录

- [安装](#-安装)
- [快速上手 — LLM (Qwen)](#-快速上手--llm-qwen)
- [快速上手 — 视觉 (YOLOv5)](#-快速上手--视觉-yolov5)
- [辅助工具](#-辅助工具)
- [资源](#-资源)
- [引用](#-引用)
- [贡献](#-贡献)
- [许可证](#-许可证)

---

## 🔧 安装

TPU-MLIR 在指定的 Docker 镜像内运行。容器启动后，既可以直接安装预编译 wheel，也可以从源码编译。

### 1. 拉取 Docker 镜像

```shell
docker pull sophgo/tpuc_dev:latest
```

如果拉取失败，可手动下载并加载镜像包：

```shell
wget https://sophon-assets.sophon.cn/sophon-prod-s3/drive/25/04/15/16/tpuc_dev_v3.4.tar.gz
docker load -i tpuc_dev_v3.4.tar.gz
```

创建并进入容器：

```shell
docker run --privileged --name tpu-mlir -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 2a. 安装预编译 wheel（推荐）

> 要求 Python ≥ 3.10，Ubuntu 22.04（推荐直接使用上述 Docker 镜像）。

```shell
pip install tpu_mlir
```

### 2b. 从源码编译

```shell
cd /workspace/tpu-mlir
pip install -r requirements.txt
source ./envsetup.sh
./build.sh
```

---

## 🤖 快速上手 — LLM (Qwen)

将 HuggingFace 上的 LLM 编译为 TPU 上的 bmodel。

<details>
<summary><b>点击展开完整 LLM 流程</b></summary>

### 1. 下载模型

推荐使用预量化 (AWQ / GPTQ / AutoRound) 版本：

```shell
git lfs install
git clone https://huggingface.co/Intel/Qwen3.5-2B-int4-AutoRound
```

### 2. 编译为 bmodel

```shell
# 如遇 transformers / torch 版本问题：
#   pip3 install transformers torchvision -U
# --max_input_length 设置最大 prefill 长度，缺省时取 -s。
llm_convert.py \
  -m /workspace/Qwen3.5-2B-int4-AutoRound \
  --max_input_length 1024 \
  -s 2048 \
  -c bm1684x \
  --max_pixels 768,768 \
  -o qwen3.5_2b
```

`llm_convert.py` 主要参数：

| 参数            | 简写  | 必选 | 说明                                                                                                                                |
| --------------- | :---: | :--: | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model_path`    |  `m`  |  ✅  | 模型权重路径                                                                                                                        |
| `seq_length`    |  `s`  |  ✅  | 最大序列长度                                                                                                                        |
| `max_input_length` |  —  |  —   | 最大单次输入长度，缺省时取 `seq_length`（`-s`）                                                                                    |
| `quantize`      |  `q`  |  ✅  | 量化方式：`auto` / `w4bf16` / `w4f16` / `bf16` / `f16` 等（如源模型已量化可省略）                                                   |
| `q_group_size`  |  `g`  |  —   | 量化分组大小（默认 64）                                                                                                             |
| `chip`          |  `c`  |  ✅  | 目标芯片：`bm1684x` / `bm1688` / `cv186ah`                                                                                          |
| `max_pixels`    |   —   |  —   | 多模态最大分辨率 `width,height`，按 `model_type` 取默认值（qwen2_5_vl: `672,896`；minicpmv: `980,980`；其他：`768,768`）            |
| `out_dir`       |  `o`  |  ✅  | 输出目录                                                                                                                            |

### 3. 在 PCIe / SoC 环境运行

将 [`python_demo`](https://github.com/sophgo/LLM-TPU/tree/main/models/Qwen3_5/python_demo) 拷贝到设备上并编译：

```shell
mkdir build && cd build
cmake ..
make
cp *cpython*.so ..
cd ..
```

执行 bmodel：

```shell
python3 pipeline.py -m xxxx.bmodel -c config
```

运行示例：

![Qwen demo](./docs/assets/qwen3.5.png)

</details>

---

## 🖼️ 快速上手 — 视觉 (YOLOv5)

以 [`yolov5s.onnx`](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx) 为例，在 BM1684X TPU 上完成编译与部署。模型已置于 `regression/model/yolov5s.onnx`。

<details>
<summary><b>点击展开完整 YOLOv5 流程</b></summary>

### 1. 准备工作目录

```shell
mkdir model_yolov5s && cd model_yolov5s
cp ${REGRESSION_PATH}/model/yolov5s.onnx .
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
mkdir workspace && cd workspace
```

### 2. 模型转 MLIR

如果模型以图像为输入，需指定预处理；输入为 npz 文件时则无需预处理。预处理公式为：

$$y = (x - \text{mean}) \times \text{scale}$$

YOLOv5 官方输入为 RGB，每像素乘 `1/255`，因此 `mean = 0,0,0`、`scale = 0.0039216,0.0039216,0.0039216`。

```shell
model_transform.py \
  --model_name yolov5s \
  --model_def ../yolov5s.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --output_names 350,498,646 \
  --test_input ../image/dog.jpg \
  --test_result yolov5s_top_outputs.npz \
  --mlir yolov5s.mlir
```

`model_transform.py` 主要参数：

| 参数                  | 必选 | 说明                                                                                |
| --------------------- | :--: | ----------------------------------------------------------------------------------- |
| `model_name`          |  ✅  | 模型名称                                                                            |
| `model_def`           |  ✅  | 模型定义文件（`.onnx` / `.pt` / `.tflite` / `.prototxt`）                            |
| `model_data`          |  —   | Caffe 权重文件（`.caffemodel`）                                                     |
| `input_shapes`        |  —   | 输入形状，例如 `[[1,3,640,640]]`，支持多输入                                         |
| `resize_dims`         |  —   | 图像 resize 尺寸；不指定时按模型输入大小                                             |
| `keep_aspect_ratio`   |  —   | 是否保持长宽比（不足部分补 0），默认关闭                                             |
| `mean`                |  —   | 各通道均值，默认 `0,0,0`                                                            |
| `scale`               |  —   | 各通道缩放，默认 `1,1,1`                                                            |
| `pixel_format`        |  —   | `rgb` / `bgr` / `gray` / `rgbd`                                                     |
| `output_names`        |  —   | 输出名；不指定时使用模型默认输出                                                     |
| `test_input`          |  —   | 用于验证的输入文件（图像 / npy / npz），不指定则跳过验证                             |
| `test_result`         |  —   | 验证结果保存路径                                                                    |
| `excepts`             |  —   | 不参与验证的层名，逗号分隔                                                          |
| `debug`               |  —   | 保留中间文件                                                                        |
| `mlir`                |  ✅  | 输出 MLIR 文件路径                                                                  |

完成后会生成预处理后的 `${model_name}_in_f32.npz`。

### 3. MLIR → F16 bmodel

```shell
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F16 \
  --processor bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --model yolov5s_1684x_f16.bmodel
```

`model_deploy.py` 主要参数：

| 参数                  | 必选 | 说明                                                                          |
| --------------------- | :--: | ----------------------------------------------------------------------------- |
| `mlir`                |  ✅  | 输入 MLIR 文件                                                                |
| `quantize`            |  ✅  | `F32` / `BF16` / `F16` / `INT8`                                               |
| `processor`           |  ✅  | 目标芯片                                                                      |
| `calibration_table`   |  —   | 量化校准表（INT8 必填）                                                       |
| `tolerance`           |  —   | MLIR 量化与 fp32 推理结果的最小相似度                                          |
| `correctness`         |  —   | 模拟器与 MLIR 量化推理结果的最小相似度（默认 `0.99,0.90`）                    |
| `excepts`             |  —   | 不参与验证的层名，逗号分隔                                                    |
| `debug`               |  —   | 保留中间文件                                                                  |
| `model`               |  ✅  | 输出 bmodel 路径                                                              |
| `dynamic`             |  —   | 动态 codegen，支持动态 shape                                                  |

### 4. MLIR → INT8 bmodel

先做校准（一般 100~1000 张样本），优先使用对称量化。

```shell
run_calibration.py yolov5s.mlir \
  --dataset ../COCO2017 \
  --input_num 100 \
  -o yolov5s_cali_table

model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize INT8 \
  --calibration_table yolov5s_cali_table \
  --processor bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --tolerance 0.85,0.45 \
  --model yolov5s_1684x_int8.bmodel
```

### 5. 验证结果

示例脚本位于 `python/samples/detect_yolov5.py`：

```shell
# ONNX
detect_yolov5.py --input ../image/dog.jpg --model ../yolov5s.onnx          --output dog_origin.jpg
# F16 bmodel
detect_yolov5.py --input ../image/dog.jpg --model yolov5s_1684x_f16.bmodel --output dog_f16.jpg
# INT8 bmodel
detect_yolov5.py --input ../image/dog.jpg --model yolov5s_1684x_int8.bmodel --output dog_int8.jpg
```

不同模型的输出对比：

![YOLOv5 results](./docs/quick_start/assets/yolov5s.png)

</details>

---

## 🛠️ 辅助工具

### `model_runner.py` —— 通用推理工具

支持 `bmodel` / `mlir` / PyTorch / ONNX / TFLite / Caffe。

```shell
model_runner.py \
  --input  resnet18_in_f32.npz \
  --model  resnet18_1684x_f32.bmodel \
  --output resnet18_output.npz
```

### `model_tool` —— bmodel 查看与编辑

```text
model_tool
  --info     model_file                                 : 查看简要信息
  --print    model_file                                 : 查看详细信息
  --extract  model_file                                 : 将多网络 bmodel 拆分为多个单网络 bmodel
  --combine  file1 .. fileN -o new_file                 : 按文件路径合并 bmodel
  --combine_dir dir1 .. dirN -o new_dir                 : 按目录合并 bmodel
  --dump     model_file start_offset byte_size out_file : 从 bmodel dump 二进制数据
```

```shell
model_tool --info resnet18_1684x_f32.bmodel
```

---

## 📖 资源

### 文档与论文

| 类型   | 链接                                                                                |
| ------ | ----------------------------------------------------------------------------------- |
| 论文   | [TPU-MLIR (arXiv 2210.15016)](https://arxiv.org/abs/2210.15016)                     |
| 手册   | [技术参考手册](https://tpumlir.org/developer_manual/index.html)                     |
| 入门   | [快速入门](https://tpumlir.org/quick_start/index.html)                              |

### 分享会

- [TPU-MLIR 论文解读](https://www.bilibili.com/video/BV1My4y1o73Q/)
- [LayerGroup](https://www.bilibili.com/video/BV1wo4y1z7AG/)

### 视频教程

<details>
<summary><b>点击展开视频目录</b></summary>

| # | 主题 | 链接 |
|:-:|------|------|
| 01 | 什么是深度学习编译器？      | [入门](https://www.bilibili.com/video/BV1yP4y1d7gz/) |
| 02 | MLIR 入门                   | [语法 1](https://www.bilibili.com/video/BV1CP411n7fj/) · [语法 2](https://www.bilibili.com/video/BV1Gt4y1F7mt/) · [语法 3](https://www.bilibili.com/video/BV1UN4y1w72r/) · [Dialect Conversion](https://www.bilibili.com/video/BV1UG411c7nm/) · [Pattern Rewriting](https://www.bilibili.com/video/BV1R44y1d7xv/) |
| 03 | TPU-MLIR 概览               | [总览](https://www.bilibili.com/video/BV19d4y1B7eR/) · [前端转换](https://www.bilibili.com/video/BV1yv4y1S7WT/) · [Lowering](https://www.bilibili.com/video/BV1gg411z7mC/) |
| 04 | 量化                        | [总览](https://www.bilibili.com/video/BV1d8411j7t4/) · [公式推导](https://www.bilibili.com/video/BV1SW4y1H7Uu/) · [校准](https://www.bilibili.com/video/BV1qK411R75k/) · [QAT](https://www.bilibili.com/video/BV12g411J7WQ/) |
| 05 | TPU 内存                    | [Ep1](https://www.bilibili.com/video/BV1T24y1G7pu/) · [Ep2](https://www.bilibili.com/video/BV1VY4y1y7ET/) |
| 06 | TPU-MLIR 实战               | [转 ONNX](https://www.bilibili.com/video/BV1FD4y1H7pT/) · [图优化](https://www.bilibili.com/video/BV1AR4y1U7D6/) · [算子支持](https://www.bilibili.com/video/BV1tL411r71p/) · [模型支持](https://www.bilibili.com/video/BV1mM411y7Ep/) · [融合预处理](https://www.bilibili.com/video/BV1ao4y1H7m8/) · [精度验证](https://www.bilibili.com/video/BV14e4y1M79d/) |

</details>

---

## 📝 引用

如果 TPU-MLIR 对您的研究有帮助，请按下列格式引用：

```bibtex
@misc{tpumlir2022,
  title         = {TPU-MLIR: A Compiler For TPU Using MLIR},
  author        = {HuPengchao and LuMan and WangLei and JiangGuoyue},
  year          = {2022},
  eprint        = {2210.15016},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PL}
}
```

---

## 🤝 贡献

欢迎提交 Bug、功能请求与 Pull Request！

1. 提交前请先在 [Issues](https://github.com/sophgo/tpu-mlir/issues) 中检索，避免重复。
2. 较大改动建议先开 Issue 讨论设计方案。
3. PR 前请运行 `regression/` 下的回归测试。

---

## 📄 许可证

本项目采用根目录下 [LICENSE](./LICENSE) 文件中规定的开源许可证。
