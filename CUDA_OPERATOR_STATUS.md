# CUDA 算子测试报告

**日期:** 2026-05-18  
**测试环境:** bm1684x, F32, --cuda  
**项目路径:** `/home/geng/Top_to_cuda/mlir_proj/tpu-mlir/bindings/pymlir/cuda/`  
**作者:** geng (junbiao_pang@bjut.edu.cn)

---

## 概览

| 分类 | 数量 |
|------|:----:|
| 完整通过 | 34 |
| cmodel 预编译库 bug | 6 |
| 编译链缺失 | 5 |
| CPU 推理崩溃 | 3 |
| 硬件限制 | 1 |
| 代码已修/未完全修 | 2 |
| 未走完整编译链 | 1 |
| 未实现 (缺 cuda 文件) | 1 |
| Active 内确认的模式 | 1 |
| **合计** | **54** |

---

## ✅ 完整通过 (34)

Div, DivConst, Einsum, Erf, Exp, MaskedFill, MaskRCNNGetBboxB, Max, MaxConst, Min, MinConst, Mish, MaxPoolWithMask, Pack, Pad, Pow, PRelu, Shape, ShuffleChannel, Sign, Sin, Sinh, SliceAxis, Softplus, Softsign, Split, Sqrt, StridedSlice, SwapChannel, Swish, Tan, Tanh, Tile, Trilu, Unpack, Where

---

## ❌ cmodel 预编译库 bug (6，不可修)

| 算子 | F32 现象 |
|------|------|
| MaxUnpool | cos=0.096 |
| MaxPoolingIndicesBwd | cos=-0.002 |
| MeanRstd | 5/6正确, bias_new全零 |
| MeanStdScale | F16 cos=-0.003 |
| ScatterElements | cos=0.72 |
| Sort | values正确, indices全零 |

---

## ❌ 编译链缺失 (5)

| 算子 | 问题 |
|------|------|
| MaskRCNNBboxPooler | F16 lowering未实现; F32 Common层 UNREACHABLE_THIS; 静态codegen llvm_unreachable |
| MaskRCNNMaskPooler | 同 BboxPooler |
| MaskRCNNRPNGetBboxes | 无 CUDA .cpp 文件; 实现复杂度高(20+输入/30+属性/5级FPN) |
| MatchTemplate | BM1684X codegen 全 llvm_unreachable; 测试仅 CPU vs CUDA |
| MeshGrid | 测试仅 CPU vs CUDA; 无 bmodel 流程 |

---

## ❌ CPU 推理崩溃 (3)

| 算子 | 现象 | 根因分析 |
|------|------|------|
| Mod | `top.Compare` 崩 (非 Mod 本身) | compare_mode 未初始化或 Binary::run 内部问题 |
| QuantizeLinear | segfault | 非 scale 除零, 需 GDB 定位 |
| ShapeSlice | `tpu.Host2Device` 崩 | 运行时/预编译库问题 |

---

## ❌ 硬件限制 (1)

| 算子 | 现象 |
|------|------|
| ScaleLut | BM1684X 仅支持 INT8/UINT8; F16 assertion 失败: `"Only support 8bit int transform for now"` |

---

## ⚠️ 代码已修/未完全修 (2)

| 算子 | 已修复 | 剩余问题 |
|------|------|------|
| ScatterND | `GenericCpuFunc.cpp` MAX/MIN `+=` 改为 `=` | 索引计算仍有 bug (cos=0.94) |
| TopK | `TopK.cpp` cudaMemcpy `DeviceToHost` 改 `HostToDevice` | indices 仍错误 (cos=0.87) |

---

## ⚠️ 未走完整编译链 (1)

| 算子 | 说明 |
|------|------|
| SelectiveScan | 测试仅 Torch/ONNX 验证; support_modes=["f16","bf16"]; F32 未测 |

---

## ⚠️ Active 内确认的实现 (1 模式)

| 模式 | 状态 |
|------|:----:|
| ELU | ✅ 通过 |
| SIGMOID / SILU / RELU / TANH | 不确定是否用户所写 |

---

## 分类说明

- **cmodel bug**: 编译链完整 (lowering+codegen 通)，但预编译库 `libbackend_1684x.so` 计算结果错误，因无源码不可修
- **编译链缺失**: 缺少 TopToTpu lowering 或 BM1684X codegen 实现，算子无法编译到硬件
- **CPU 推理崩溃**: Top/Common 层的 CPU 推理实现有 bug 导致崩溃
- **硬件限制**: 硬件不支持该精度模式
