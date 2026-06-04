# CUDA 算子变更日志

---

## 1. Abs （取绝对值）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Abs.cpp` | 算子入口，调用 `cuda::bmAbs` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_abs` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmAbs` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmAbs` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaAbsOp(top::AbsOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::AbsOp` → `cudaAbsOp` |
| 修改 | `cuda/Active.cpp` | **关键修复**: 新增 `tpu::ActiveMode::ABSVAL` 分支（因 lowering 将 `top.Abs` 转为 `tpu.Active{mode=ABSVAL}`） |

### Kernel 算法

```
g_abs(input, output, num):
  for i in range(num):
    output[i] = fabsf(input[i])
```

单元素逐位取绝对值，与 GELU 同属最简单算子模板（纯 FP32，单输入单输出）。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Abs"（通过）
```

---

## 2. AdaptiveAvgPool2D （自适应平均池化）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/AdaptiveAvgPool.cpp` | 算子入口，提取 input shape / output_size，调用 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_adaptiveAvgPool2D` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmAdaptiveAvgPool2D` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmAdaptiveAvgPool2D` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaAdaptiveAvgPoolOp(top::AdaptiveAvgPoolOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::AdaptiveAvgPoolOp` → `cudaAdaptiveAvgPoolOp` |

| 修改 | `cuda/Pool2D.cpp` | **关键修复**: 在 `cudaPool2DOp` 中新增 `is_adaptive` 分支，用自定义 kernel 替代 cuDNN |

### Kernel 算法

与 TPU CPU reference（`AvgPool.cpp:128-178`，`is_adaptive=true` 分支）公式完全一致：

```
for each output(n,c,oh,ow):
    start_h = oh * H / OH                          // 整数除法 = floor
    end_h   = ((oh+1) * H + OH - 1) / OH           // ceil
    start_w = ow * W / OW
    end_w   = ((ow+1) * W + OW - 1) / OW
    kh      = end_h - start_h                       // 每个位置可能不同
    kw      = end_w - start_w
    output  = mean( input[n,c, start_h:end_h, start_w:end_w] )
```

逐位置的 floor/ceil 公式，窗口大小不统一。例如 `(32,32)→(3,3)` 时高度窗口 = [11, 12, 11]。

> **关键踩坑**: AdaptiveAvgPool 在 MLIR compilation 中被 lower 成 `tpu.Pool2D`（`pool_mode=Avg`, `is_adaptive=true`），不是 `top::AdaptiveAvgPoolOp`。因此 dispatch 实际走的是 `cudaPool2DOp`。原先 `cudaPool2DOp` 直接用 cuDNN 的 uniform kernel/stride pooling（如 kernel=12,stride=10 全市窗口），与 TPU reference 的 PyTorch 逐位置公式不一致。修复方式是在 `cudaPool2DOp` 中加 `is_adaptive` 检测，提前返回并用自定义 kernel 替代 cuDNN。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "AdaptiveAvgPool1d"（通过）
```

---

## 3. AddConst （张量加常数）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/AddConst.cpp` | 算子入口，参考 `MulConst` 模式 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_addConst4DF32` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `addConst4DF32` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `addConst4DF32` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaAddConstOp(top::AddConstOp)` + `cudaAddConstOp(tpu::AddConstOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::AddConstOp` + `tpu::AddConstOp` → `cudaAddConstOp` |

### Kernel 算法

```
g_addConst4DF32(input, const_v, output, do_relu, n, c, h, w):
  for i in range(n * c * h * w):
    val = input[i] + const_v
    if do_relu and val < 0:
      val = 0
    output[i] = val
```

逐元素加常数，可选 ReLU。与 `MulConst`/`SubConst` 同属 const 系列算子模板。

### 算子定义

- 输入: `AnyTensor` (NCHW 格式)
- 属性: `const_val` (FloatAttr), `do_relu` (BoolAttr)
- 输出: `output = input + const_val`
- 参考: `lib/Dialect/Top/Interfaces/AddConst.cpp`


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "AddConst"（通过）
```

---

## 4. Arccos （反余弦）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Arccos.cpp` | top 算子入口 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_arccos` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmArccos` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmArccos` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::ARCCOS` early return 分支 |
| 修改 | `pycuda.h` | 声明 `cudaArccosOp(top::ArccosOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::ArccosOp` → `cudaArccosOp` |

### Kernel 算法

```
g_arccos(input, output, num):
  for i in range(num):
    output[i] = acosf(input[i])
```

单元素逐位反余弦。与 Abs/GELU 同属最简单的 FP32 逐元素算子模板。

### 算子定义

- 输入: `AnyTensor` (范围需在 [-1, 1] 内)
- 输出: `output = acos(input)`
- 参考: `lib/Dialect/Top/Interfaces/Arccos.cpp`

### lowering 路径

`top.Arccos` → `tpu.Active{mode=ARCCOS}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::ArccosOp` dispatch 作安全网。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Acos"（通过）
```

---

## 5. Arctanh （反双曲正切）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Arctanh.cpp` | top 算子入口 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_arctanh` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmArctanh` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmArctanh` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::ARCTANH` early return 分支 |
| 修改 | `pycuda.h` | 声明 `cudaArctanhOp(top::ArctanhOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::ArctanhOp` → `cudaArctanhOp` |

### CUDA Kernel 详解

#### Device Kernel (`cuda_global.cuh`)

```cpp
__global__ void g_arctanh(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
  if (i < num) {                                    // 边界保护，防止越界
    output[i] = atanhf(input[i]);                   // CUDA 内置反双曲正切
  }
}
```

- **线程映射**: 一维 grid/block，每个线程处理一个元素。`blockIdx.x * blockDim.x + threadIdx.x` 为标准的 CUDA 全局索引计算
- **边界保护**: `if (i < num)` 确保线程数超过数据量时不会越界访问
- **核心计算**: `atanhf()` 是 CUDA math API 的单精度浮点反双曲正切函数。定义域为 (-1, 1)，值域为整个实数轴
- **性能**: 纯计算密集型，无全局内存合并访问问题（连续读写）

#### Host Wrapper (`cuda_helper.cu`)

```cpp
void bmArctanh(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);           // (size + 255) / 256
  int block_size = CUDA_BLOCK_SIZE;                  // 256 线程/块
  g_arctanh<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size);
}
```

- **Grid 配置**: `CUDA_NUM_BLOCKS(size)` = `(size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE`，确保覆盖所有元素
- **Block 配置**: 256 线程/块，平衡 SM 占用率和调度灵活性
- **类型转换**: `void*` → `float*`，因为 top 层 Arctanh 只有 FP32 路径

#### 调用链

```
GPU:  g_arctanh  ←  bmArctanh  ←  cudaArctanhOp / cudaActiveOp(ARCTANH)
       (kernel)      (host)         (operator entry)
```

#### 算子入口 (`Arctanh.cpp` / `Active.cpp`)

两条路径：
1. **主路径**（lowering 后）: `tpu.Active{mode=ARCTANH}` → `cudaActiveOp` → `cuda::bmArctanh`
2. **安全网**: `top.Arctanh` → `cudaArctanhOp` → `cuda::bmArctanh`

两路径均支持 FP32 直通 + 非 FP32 先转 FP32 再计算模式。

### 算子定义

- 输入: `AnyTensor` (定义域 (-1, 1)，测试数据被 clip 到 (-0.99, 0.99))
- 输出: `output = atanh(input)`
- 参考: `lib/Dialect/Top/Interfaces/Arctanh.cpp`
- FLOPs: 4 × 元素数

### lowering 路径

`top.Arctanh` → `tpu.Active{mode=ARCTANH}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::ArctanhOp` dispatch 作安全网。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Atanh"（通过）
```

---

## 6. Arg （ArgMax / ArgMin）

**日期**: 2026-05-06

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Arg.cpp` | tpu + top 双入口 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_argMax` + `g_argMin` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmArgMax` + `bmArgMin` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmArgMax` + `bmArgMin` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaArgOp(tpu::ArgOp)` + `cudaArgOp(top::ArgOp)` |
| 修改 | `pycuda.cpp` | dispatch `tpu::ArgOp` + `top::ArgOp` → `cudaArgOp` |

### CUDA Kernel 详解

#### 数据结构

Arg 操作沿指定 `axis` 做 reduce 找极值索引。将 N 维 tensor 分解为三段：

```
[ outer_dims... | axis_dim | inner_dims... ]
        ↑             ↑            ↑
    outer_dim     axis_dim    inner_dim
```

- `outer_dim` = axis 前所有维度乘积
- `axis_dim` = 被规约的维度大小
- `inner_dim` = axis 后所有维度乘积
- 输出形状 = 输入形状去掉 axis 维（keepdims 时保留为 1）

#### Device Kernel (`g_argMax`)

```cpp
__global__ void g_argMax(float *input, float *indices,
                          int outer_dim, int axis_dim, int inner_dim,
                          bool select_last) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;    // 全局线程索引
  int total = outer_dim * inner_dim;                  // 总输出元素数
  if (idx >= total) return;                           // 边界保护

  // 将一维 idx 解码为 (outer, inner) 坐标
  int o = idx / inner_dim;                            // 外层索引
  int i = idx % inner_dim;                            // 内层索引

  // 定位到当前 (o, i) 对应的 axis 维度的起始位置
  float *in_ptr = input + o * axis_dim * inner_dim + i;

  // 扫描 axis_dim 个元素，找最大值索引
  int best_idx = 0;
  float best_val = in_ptr[0];
  for (int a = 1; a < axis_dim; a++) {
    float val = in_ptr[a * inner_dim];                // 跨步读取
    if (select_last ? (val >= best_val) : (val > best_val)) {
      best_val = val;
      best_idx = a;
    }
  }
  indices[idx] = (float)best_idx;                     // 输出 float 类型索引
}
```

- **内存访问模式**: `in_ptr[a * inner_dim]` 表示沿 axis 维度步进 `inner_dim` 个元素，适合 inner_dim 较大时的合并访问，inner_dim 较小时可能有 bank conflict
- **比较逻辑**: `select_last` 控制相等时取第一个还是最后一个索引
- **输出类型**: 索引值以 float 存储（TPU lowering 后 indices 为 I32/F32）

#### `g_argMin`

与 `g_argMax` 对称，比较方向相反：
```cpp
select_last ? (val <= best_val) : (val < best_val)
```

#### Host Wrapper

```cpp
void bmArgMax(void *input, void *indices, int outer_dim, int axis_dim,
              int inner_dim, bool select_last) {
  int total = outer_dim * inner_dim;
  int num_blocks = CUDA_NUM_BLOCKS(total);             // 覆盖所有输出元素
  int block_size = CUDA_BLOCK_SIZE;                     // 256 线程/块
  g_argMax<<<num_blocks, block_size>>>(
      (float *)input, (float *)indices,
      outer_dim, axis_dim, inner_dim, select_last);
}
```

#### 调用链

```
GPU:  g_argMax / g_argMin  ←  bmArgMax / bmArgMin  ←  cudaArgOp
       (kernel)               (host wrapper)           (operator entry)
```

#### 算子入口 (`Arg.cpp`)

两条路径处理 top 和 tpu 两个 dialect，核心逻辑相同：

1. 从 `module::getShape(input)` 获取 shape
2. 归一化 `axis`（负数转正）
3. 计算 `outer_dim` = axis 前维度乘积，`inner_dim` = axis 后维度乘积
4. 根据 `mode` 分发到 `bmArgMax` 或 `bmArgMin`
5. top 版本用 `op.getMode().str() == "ArgMax"` 判断模式；tpu 版本用 `op.getMode() == tpu::ArgMode::ArgMax`

### 算子定义

- 输入: `AnyRankedTensor`
- 属性: `axis` (I64), `keepdims` (Bool), `mode` (ArgMax/ArgMin), `select_last_index` (Bool)
- 输出: `indices` (int32/float) + 可选 `values`
- 参考: `lib/Dialect/Top/Interfaces/Arg.cpp`

### lowering 路径

`top.Arg` → `tpu.Arg`，两路径均 dispatch 到 `cudaArgOp`。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Arg"（通过）
```

---

## 7. BatchNorm / BatchNormTrain / BatchNormBwd

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/BatchNorm.cpp` | 三个算子的整洁实现，含 top/tpu 多入口 + FP32 类型自动转换 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_batchNormInference` / `g_batchNormTrainStats` / `g_batchNormTrainNormalize` / `g_batchNormBwdStats` / `g_batchNormBwdCompute` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmBatchNorm` / `bmBatchNormTrain` / `bmBatchNormBwd` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmBatchNorm` / `bmBatchNormTrain` / `bmBatchNormBwd` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaBatchNormOp(top)` + `cudaBatchNormOp(tpu)` + `cudaBatchNormBwdOp(top)` + `cudaBatchNormBwdOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::BatchNormOp` / `tpu::BatchNormTrainOp` → `cudaBatchNormOp`，`tpu::BatchNormBwdOp` / `top::BatchNormBwdOp` → `cudaBatchNormBwdOp` |

### 算子概览

| 算子 | 方向 | 输入 | 输出 |
|---|---|---|---|
| BatchNorm | 推理 | input, gamma, beta, mean, var | output |
| BatchNormTrain | 训练 | input, mean, var, gamma, beta | output, mean_out, saved_invstd, running_mean, running_var |
| BatchNormBwd | 反向 | grad_out, input, weight_opt, saved_mean, saved_invstd | grad_in, weight_grad, bias_grad |

### BatchNormBwd CUDA Kernel 详解

两步 kernel，用临时缓冲区 `dxhut` 传递中间结果：

**Step 1: `g_batchNormBwdStats`** — 每线程处理一个空间位置，计算 x_hat, dxhut，用 `atomicAdd` 累加 per-channel 的 dgamma/dbeta/dx2_tmp/dx3。

**Step 2: `g_batchNormBwdCompute`** — 用累加结果计算每个位置的 `dx = (rstd / M) * (M * dxhut - dx2_tmp[ci] * (x - mean) - dx3[ci])`。

线程配置：grid = ceil((N×C×spatial) / 256), block = 256，纯 1D 逐元素模型。

### lowering 路径

- `top.BatchNorm` → 保持原样 → `cudaBatchNormOp`
- `top.BatchNormTrain` → `tpu.BatchNormTrain` → `cudaBatchNormOp`
- `top.BatchNormBwd` → `tpu.BatchNormBwd` → `cudaBatchNormBwdOp`
- 均有 top/tpu 双 dispatch 安全网

其中BatchNormTrain 和 BatchNormBwd无法在 test_onnx.py 中测试，原因：

不是 ONNX 标准 op — BatchNormTrain 和 BatchNormBwd 是编译器内部生成的训练算子，torch.onnx.export 不会产生
多输出复杂 — BatchNormTrain 有 5 个输出（output, mean, var, saved_invstd, running_stats），BatchNormBwd 有 3 个（grad_in, weight_grad, bias_grad），用 torch.autograd.Function + g.op() 写多输出 symbolic 极易出错
无参考输出 — 没有标准 ONNX runtime 可以产生参考值做对比


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchBatchNorm"（通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "BatchNorm2D"  （通过）
```

---

## 8. Kernel 线程配置重构

**日期**: 2026-05-06

### Arg — 块内协作归约

**旧**: 1 thread 串行扫描整个 axis。**新**: 每个输出元素一个 block，块内协作归约。

```
grid  = outer_dim × inner_dim 个 block
block = min(axis_dim, 256) 线程

每 block：
  1. 各线程分块读 axis → 局部最优存 shared memory
  2. __syncthreads() → 树形归约
  3. thread 0 写最终结果
```

模板参数 `IS_MAX`/`SELECT_LAST` 编译期确定，无运行时分支。

### AdaptiveAvgPool2D — 2D 空间映射

**旧**: 1D grid 手动解码 (n,c,oh,ow)。**新**: 3D grid + 2D block 自然空间映射。

```
grid.z = N×C,  grid.y = ceil(OH/8),  grid.x = ceil(OW/16)
block  = (16, 8)

线程 → 输出像素，相邻线程处理相邻空间位置 → L1 cache 高命中率。
```

### 逐元素 kernel — 保持 1D 256

`abs/arccos/arctanh/addConst4DF32` 保持 256 线程/block，纯访存密集型无需改动。

---

## 9. Compare （元素级比较）

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Compare.cpp` | 算子入口，`top` + `tpu` 双入口，解析 `mode` 属性并映射为整数编码 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_compare4DF32` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmCompare4DF32` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmCompare4DF32` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaCompareOp(top::CompareOp)` + `cudaCompareOp(tpu::CompareOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::CompareOp` + `tpu::CompareOp` → `cudaCompareOp` |

### Kernel 算法

```
g_compare4DF32(lhs, rhs, out, mode, n0,c0,h0,w0, n1,c1,h1,w1, n2,c2,h2,w2):
  for each output(n,c,h,w) with broadcast:
    a = lhs[ broadcast_idx(lhs_shape) ]
    b = rhs[ broadcast_idx(rhs_shape) ]
    switch mode:
      Equal(0):          out = (a == b) ? 1 : 0
      Greater(1):        out = (a > b)  ? 1 : 0
      GreaterOrEqual(2): out = (a >= b) ? 1 : 0
      Less(3):           out = (a < b)  ? 1 : 0
      LessOrEqual(4):    out = (a <= b) ? 1 : 0
      NotEqual(5):       out = (a != b) ? 1 : 0
      And(6):            out = (a!=0 && b!=0) ? 1 : 0
      Not(7):            out = (a == 0) ? 1 : 0
      Xor(8):            out = (a!=0 xor b!=0) ? 1 : 0
```

广播逻辑与 Sub/Mul/Add 的 4D broadcast kernel 一致，`mode` 在 CPU 端从字符串映射为整数传入 kernel 避免 GPU 端字符串比较。

### 算子定义

- 输入: `lhs` (AnyTensor), `rhs` (AnyTensor)
- 属性: `mode` (CompareModeAttr: Equal / Greater / GreaterOrEqual / Less / LessOrEqual / NotEqual / And / Not / Xor)
- 输出: `output = 1 if condition true else 0`
- 参考: `lib/Dialect/Top/Interfaces/Compare.cpp`

### lowering 路径

`top.Compare` 存在两个 canonicalize pattern (`CompareToCompareConst` + `CompareConstantFill`) 将单元素 rhs / ConstantFill 转为 `top.CompareConst`，多元素 rhs 保持 `top.Compare` 不变。实际 CUDA 推理时 `top.Compare` 已被 lowering 为 `tpu.Compare`，因此需同时 dispatch 两个 dialect。

> **关键踩坑**: 初次仅实现 `top::CompareOp` dispatch，编译通过但 CUDA 推理 crash（`Trace/breakpoint trap`），报错 `tpu.Compare` 找不到 handler。根因是 MLIR 编译 pipeline 将 `top.Compare` → `tpu.Compare`，实际进入 CUDA invoke 的是 tpu dialect。修复方式：新增 `tpu::CompareOp` dispatch。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Compare"（通过）
```

---

## 10. CompareConst （与常数比较）

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `cuda/Compare.cpp` | 新增 `cudaCompareConstOp(top)` + `cudaCompareConstOp(tpu)`，复用 `getModeInt()` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_compareConst4DF32` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmCompareConst4DF32` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmCompareConst4DF32` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaCompareConstOp(top)` + `cudaCompareConstOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::CompareConstOp` + `tpu::CompareConstOp` → `cudaCompareConstOp` |

### Kernel 算法

```
g_compareConst4DF32(input, const_v, output, mode, inversed, n, c, h, w):
  for i in range(n * c * h * w):
    a, b = (input[i], const_v) if not inversed else (const_v, input[i])
    output[i] = (compare(a, b, mode)) ? 1.0 : 0.0
```

与 Compare 共享同一套 mode 编码（0-8），`inversed` 标志控制常数作为左/右操作数。无 broadcast 开销，纯逐元素计算。

### 算子定义

- 输入: `input` (AnyTensor)
- 属性: `const_val` (F64), `mode` (CompareModeAttr), `inversed` (Bool)
- 输出: `output = 1 if condition true else 0`
- 参考: `lib/Dialect/Top/Interfaces/CompareConst.cpp`

### lowering 路径

`top.CompareConst` → `tpu.CompareConst`，与 Compare 类似，需同时 dispatch 两个 dialect。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "CompareCst"（通过）
```

---
ConcatVolume在topop.td中不存在



## 11. ConstantFill （常量填充）

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/ConstantFill.cpp` | top + tpu 双入口，读取 `value` 属性并填充输出 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_constantFill` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmConstantFill` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmConstantFill` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaConstantFillOp(top)` + `cudaConstantFillOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::ConstantFillOp` + `tpu::ConstantFillOp` → `cudaConstantFillOp` |

### Kernel 算法

```
g_constantFill(output, value, num):
  for i in range(num):
    output[i] = value
```

纯 fill 操作，无数学计算。所有输出元素填充为同一常量值，与 AddConst/MulConst 同属 const 系列。

### 算子定义

- 输入: `input` (AnyTensor, 仅用于提供 shape 参考)
- 属性: `value` (F64)
- 输出: `output = value * ones(shape(input))`
- 参考: `lib/Dialect/Top/Interfaces/ConstantFill.cpp`

### lowering 路径

`top.ConstantFill` → `tpu.ConstantFill`，需同时 dispatch 两个 dialect。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ConstantFillDyn"（通过）
```

---

## 12. Cos / Cosh / Correlation /copy

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Cos.cpp` | `top::CosOp` 入口，调用 `bmCos` |
| 新增 | `cuda/Cosh.cpp` | `top::CoshOp` 入口，调用 `bmCosh` |
| 新增 | `cuda/Copy.cpp` | `top` + `tpu` 双入口，strided copy |
| 新增 | `cuda/Correlation.cpp` | `top` + `tpu` 双入口，立体匹配 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_cos` / `g_cosh` / `g_copy` / `g_correlation` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmCos` / `bmCosh` / `bmCopy` / `bmCorrelation` |
| 修改 | `cuda/cuda_helper.cu` | 实现 4 个 host wrapper |
| 修改 | `pycuda.h` | 声明 6 个入口 |
| 修改 | `pycuda.cpp` | 新增 6 个 dispatch |
| 修改 | `cuda/Active.cpp` | **关键修复**: 新增 `tpu::ActiveMode::COS` + `COSH` 分支（因 lowering 将 `top.Cos`/`top.Cosh` 转为 `tpu.Active`） |
| 修改 | `python/transform/OnnxConverter.py` | 新增 `Cosh` ONNX → MLIR 转换 |
| 修改 | `python/test/test_onnx.py` | 新增 `Cos`/`Cosh`/`Correlation` 测试 case，`Copy` 暂禁用（编译器内部 op） |

### 算子概览

| 算子 | 类型 | 输入 | 输出 |
|---|---|---|---|
| Cos | 逐元素 | input | cos(input) |
| Cosh | 逐元素 | input | cosh(input) |
| Copy | 内存操作 | input, shape, strides | output (strided copy) |
| Correlation | 立体匹配 | left, right | correlation volume |

### 各 Kernel 算法

**Cos/Cosh**: 单元素逐位，`output[i] = cosf/coshf(input[i])`，与 Abs/GELU 同属最简单模板。

**Copy**: 4D strided copy，`in_idx = n*i_n + c*i_c + h*i_h + w*i_w`，`out_idx = n*o_n + c*o_c + h*o_h + w*o_w`。用于 tensor reshape/transpose 等场景。

**Correlation**: 立体匹配算法。每个输出 (group, cut, spatial) 线程计算左右特征对应 patch 的点积均值：
```
output[group][cut][h][w] = mean(left[group][:][h][w] * right[group][:][h][w-cut])
```
当 `w < cut` 时输出 0（无效区域）。

### lowering 路径

- `top.Cos` → `tpu.Active{mode=COS}` → `cudaActiveOp`
- `top.Cosh` → `tpu.Active{mode=COSH}` → `cudaActiveOp`
- `top.Copy` → `tpu.Copy`
- `top.Correlation` → `tpu.Correlation`
- Cos/Cosh 实际走 Active.cpp（类似 Abs→ABSVAL / Arccos→ARCCOS 模式），top 入口仅作安全网

> **关键踩坑**: 初次仅实现 `top::CosOp`/`top::CoshOp` dispatch，CUDA 推理 crash（`Not Implemented`）。根因是 MLIR 编译 pipeline 将 `top.Cos` → `tpu.Active{mode=COS}`，需在 `Active.cpp` 中新增 COS/COSH 分支。另外 `OnnxConverter.py` 未注册 Cosh，ONNX 转换阶段就失败。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Correlation"（通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Cos"（通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Cosh"（通过）
  copy 在case_map中被注释
```

---

## 13. CumSum （累加和）/Csc  /Custom

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/CumSum.cpp` | top + tpu 双入口，outer/axis/stride 三维分解 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_cumSum` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmCumSum` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmCumSum` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaCumSumOp(top)` + `cudaCumSumOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::CumSumOp` + `tpu::CumSumOp` → `cudaCumSumOp` |

### Kernel 算法

```
g_cumSum(input, output, outer_dim, axis_dim, stride):
  每线程处理一个 (outer, stride) 对的串行前缀和:
    base = outer * axis_dim * stride + s
    sum = 0
    for a in 0..axis_dim:
      sum += input[base + a * stride]
      output[base + a * stride] = sum
```

将 N 维 tensor 沿 axis 分解为 `outer_dim × axis_dim × stride`，每线程处理长度为 `axis_dim` 的前缀和链。并行度为 `outer_dim × stride = num_elements / axis_dim`。

### 算子定义

- 输入: `input` (AnyTensor)
- 属性: `axis` (I64), `dim` (optional weight)
- 输出: `output[i] = sum_{j=0..i} input[j]`（沿 axis 方向前缀和）
- 参考: `lib/Dialect/Top/Interfaces/CumSum.cpp`

### 跳过说明

- **Csc**: 颜色空间转换，仅后端使用，top 层 `llvm_unreachable`，无需 CUDA 实现
- **Custom**: 动态加载 `libplugin_custom.so` 执行，无法写成静态 CUDA kernel

### lowering 路径

`top.CumSum` → `tpu.CumSum`，需同时 dispatch 两个 dialect。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "CumSum"（通过）
```

---

## 14. ConvBwd_Weight （卷积反向权重梯度）x

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/ConvBwd_Weight.cpp` | top + tpu 双入口，使用 cuDNN `cudnnConvolutionBackwardFilter` |
| 修改 | `pycuda.h` | 声明 `cudaConvBwdWeightOp(top)` + `cudaConvBwdWeightOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::ConvBwdWeightOp` + `tpu::ConvBwdWeightOp` → `cudaConvBwdWeightOp` |

### Kernel 算法

使用 cuDNN `cudnnConvolutionBackwardFilter` API，计算 dW = conv(input, grad_out)：

```
dW = ConvolutionBackwardFilter(input, gradout, params):
  pre-pad input if asymmetric padding (cuDNN only supports symmetric)
  set up input_desc [N, IC, IH, IW]
  set up dy_desc    [N, OC, OH, OW]
  set up dw_desc    [OC, IC/G, KH, KW]
  set up conv_desc (padding, stride, dilation, groups)
  cudnnGetConvolutionBackwardFilterWorkspaceSize
  cudnnConvolutionBackwardFilter → dW
  copy dW to output
```

处理非对称 padding 时先对 input 做 zero-padding，使 cuDNN 可以处理。

### 算子定义

- 输入: `input` (前向激活), `gradout` (损失对输出的梯度), `gradout_transpose` (可选预转置)
- 属性: `groups`, `input_shape`, `grad_out_shape`, `kernel_shape`, `stride`, `dilations`, `padding`, `grad_bias_enable`
- 输出: dW [OC, IC/groups, KH, KW]
- 参考: `lib/Dialect/Top/Interfaces/ConvBwd_Weight.cpp`

### 说明

- **ConvBwd** (data backward) 在本代码库中不存在，仅有 ConvBwd_Weight
- CPU reference 实现为空 (`return success()`)，无可参考逻辑
- 当前仅实现 FP32 路径

### lowering 路径

`top.ConvBwd_Weight` → `tpu.ConvBwd_Weight`，需同时 dispatch 两个 dialect。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ConvBwd_Weight"
```

---

## 15. Div / DivConst （逐元素除法）/DeformConv2D

**日期**: 2026-05-07

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Div.cpp` | Div (top+tpu) + DivConst (top) |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_div4DF32` + `g_divConst4DF32` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `div4DF32` + `divConst4DF32` |
| 修改 | `cuda/cuda_helper.cu` | 实现 2 个 host wrapper |
| 修改 | `pycuda.h` | 声明 3 个入口 |
| 修改 | `pycuda.cpp` | 新增 3 个 dispatch |

### Kernel 算法

```
g_div4DF32(a, b, out, relu, reverse, n0..ow):
  for each output(n,c,h,w) with broadcast:
    result = reverse ? (b / a) : (a / b)
    if relu and result < 0: result = 0
    out[idx] = result

g_divConst4DF32(input, const_v, output, relu, reverse, n,c,h,w):
  for i in range(n*c*h*w):
    val = reverse ? (const_v / input[i]) : (input[i] / const_v)
    if relu and val < 0: val = 0
    output[i] = val
```

广播逻辑与 Sub/Mul 同模式，支持 `is_reverse` 反转被除数。

### 算子定义

- Div: `lhs`, `rhs` → `output = lhs / rhs`（is_reverse 反转）
- DivConst: `input`, `const_val` → `output = input / const_val`
- 参考: `lib/Dialect/Top/Interfaces/Div.cpp`, `DivConst.cpp`

### lowering 路径

- `top.Div` → `tpu.Div`，需同时 dispatch 两个 dialect
- `top.DivConst` 无 tpu 版本，仅 top 入口

### 跳过说明

- **DeformConv2D**: 可变形卷积，需 cuDNN DCNv2 API，极其复杂


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Div" （通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "DivConst"（通过）
```

---

## 16. DepackRaw （Raw 图像解包）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/DepackRaw.cpp` | top + tpu 双入口 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_depackRaw` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmDepackRaw` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmDepackRaw` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaDepackRawOp(top)` + `cudaDepackRawOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch 两个 dialect |

### Kernel 算法

```
g_depackRaw(input, output, n, ih, iw, ph, pw, scale, black_level, c0..c3):
  oh=ih*2, ow=iw*2
  for each output (n, oh, ow):
    block_y = oh % 2, block_x = ow % 2
    ch = channel_order[block_y*2 + block_x]   // 4 通道 → 2×2 空间块
    in_h = oh/2 + ph, in_w = ow/2 + pw       // 跳过 padding
    val = input[n, ch, in_h, in_w]
    output[n, oh, ow] = (val - black_level) * 255/(white_level - black_level)
```

将 4 通道 packed raw 图像解包为单通道 Bayer 模式，去除 padding，并做黑/白电平校正。

### 算子定义

- 输入: `[N, 4, IH+PH, IW+PW]`
- 输出: `[N, 1, IH*2, IW*2]`
- 属性: `padding_h`, `padding_w`, `white_level`, `black_level`, `channel_order`
- 参考: `lib/Dialect/Top/Interfaces/DepackRaw.cpp`

### lowering 路径

`top.DepackRaw` → `tpu.DepackRaw`，需同时 dispatch 两个 dialect。

没有测试 case。DepackRaw 是相机 Raw 图像解包算子，非 ONNX 标准，只能 .mlir 直测：

---

## 17. DequantizeLinear / DequantInt （反量化）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/DequantizeLinear.cpp` | top 入口，per-tensor + per-channel |
| 新增 | `cuda/DequantInt.cpp` | top + tpu 双入口，Normal + TFLite 模式 |
| 修改 | `cuda/cuda_device.cuh` | 新增 `d_applyMultiplierAndRShift` device 函数 |
| 修改 | `cuda/cuda_global.cuh` | 新增 4 个 kernel（每种算子的 per-tensor + per-channel） |
| 修改 | `cuda/cuda_helper.h` | 声明 4 个 host wrapper |
| 修改 | `cuda/cuda_helper.cu` | 实现 4 个 host wrapper（含类型分发） |
| 修改 | `pycuda.h` | 声明 3 个入口 |
| 修改 | `pycuda.cpp` | 新增 3 个 dispatch |

### 算子概览

| 算子 | 公式 | 输入类型 | 输出 | per-channel |
|---|---|---|---|---|
| DequantizeLinear | `(x - zp) * scale` | int8/uint8/int32 | float | axis 任意维度 |
| DequantInt | `applyMultiplierAndRShift(x - zp, multi, -shift)` | int8/uint8 | float | channel dim=1 |

### Kernel 算法

**DequantizeLinear**: 类型模板化，per-tensor 为 1D kernel，per-channel 解码 `c = (i / inner) % channel_dim` 后取对应的 scale[c]/zp[c]。

**DequantInt**: 支持 Normal（MultiplierShift）和 TFLite 两种量化模式。Normal 模式调用 device 函数 `d_applyMultiplierAndRShift` 做定点乘加移位；TFLite 模式先左移 lshift → `Right_Shift_Round(31, HALF_UP)` → `Right_Shift_Round(-shift, rmode)`。

### 算子定义

- DequantizeLinear 参考: `lib/Dialect/Top/Interfaces/DequantizeLinear.cpp`
- DequantInt 参考: `lib/Dialect/Top/Interfaces/DequantInt.cpp`

### lowering 路径

- `top.DequantizeLinear` → 无 tpu 版本，仅 top 入口
- `top.DequantInt` → `tpu.DequantInt`，需同时 dispatch 两个 dialect
test_onnx.py 中：

DequantInt — 没有任何引用
DequantizeLinear — 只在 test_QDQConv（Quantize-Dequantize-Conv 量化卷积测试）内部作为中间步骤出现，不是独立的测试 case
---

## 18. DtypeCast （数据类型转换）x

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/DtypeCast.cpp` | top + tpu 双入口，直接调 `convertType` |
| 修改 | `pycuda.h` | 声明 `cudaDtypeCastOp(top)` + `cudaDtypeCastOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch 两个 dialect |

### 实现

无新增 kernel，直接复用已有的 `cuda::convertType`：

```
cuda::convertType(input, output, num, src_type, dst_type)
```

`convertType` 内部依次尝试 `cudnnTransformTensor` → `g_f32ToF16` / `g_f16ToF32` kernel。

### 算子定义

- 输入: `AnyTensor` (FP32)
- 输出: `AnyTensor` (FP16)
- 参考: `lib/Dialect/Top/Interfaces/DtypeCast.cpp`

### lowering 路径

`top.DtypeCast` → `tpu.DtypeCast`，需同时 dispatch 两个 dialect。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "DtypeCast"
```

---

## 19. Elu （指数线性单元）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Elu.cpp` | top 算子入口，读取 `alpha` 属性并传给 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_elu` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmElu` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmElu` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::ELU` 分支，从 `coeffs` 属性读取 alpha |
| 修改 | `pycuda.h` | 声明 `cudaEluOp(top::EluOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::EluOp` → `cudaEluOp` |

### Kernel 算法

```
g_elu(input, output, num, alpha):
  for i in range(num):
    val = input[i]
    output[i] = val > 0 ? val : alpha * (exp(val) - 1)
```

单元素逐位，alpha 参数控制负半轴斜率。与 GELU 同属带参数的逐元素激活函数模板。

### 算子定义

- 输入: `AnyTensor`
- 属性: `alpha` (F64Attr, 默认 1.0)
- 输出: `output[i] = x > 0 ? x : alpha * (exp(x) - 1)`
- 参考: `lib/Dialect/Top/Interfaces/Elu.cpp`

### lowering 路径

`top.Elu` → `tpu.Active{mode=ELU, coeffs=[alpha]}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::EluOp` dispatch 作安全网。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Elu"（通过）
```

---

## 20. Erf （误差函数）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Erf.cpp` | top 算子入口，调用 `bmErf` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_erf` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmErf` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmErf` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::ERF` 分支 |
| 修改 | `pycuda.h` | 声明 `cudaErfOp(top::ErfOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::ErfOp` → `cudaErfOp` |

### Kernel 算法

```
g_erf(input, output, num):
  for i in range(num):
    output[i] = erff(input[i])
```

单元素逐位误差函数。CUDA 内置 `erff()` 为标准高斯误差函数，与 CPU reference 的 `std::erf` 一致。

### 算子定义

- 输入: `AnyTensor`
- 输出: `output = erf(input)`（值域 (-1, 1)）
- 参考: `lib/Dialect/Top/Interfaces/Erf.cpp`

### lowering 路径

`top.Erf` → `tpu.Active{mode=ERF}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::ErfOp` dispatch 作安全网。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Erf"（通过）
```

---

## 21. Exp （自然指数）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Exp.cpp` | top 算子入口，调用 `bmExpElm` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_expElm` kernel（独立于 Softmax 用的 `g_bmExp`） |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmExpElm` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmExpElm` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::EXP` 分支 |
| 修改 | `pycuda.h` | 声明 `cudaExpOp(top::ExpOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::ExpOp` → `cudaExpOp` |

### Kernel 算法

```
g_expElm(input, output, num):
  for i in range(num):
    output[i] = expf(input[i])
```

纯逐元素指数运算。**注意**：代码库中已有 `bmExp`/`g_bmExp`，但其签名带有 `outer_dim/axis_dim/inner_dim` 参数专为 Softmax 设计。此处新增独立的逐元素版本 `bmExpElm`/`g_expElm` 以避免签名字冲突。

### 算子定义

- 输入: `AnyTensor`
- 输出: `output = exp(input)`
- 参考: `lib/Dialect/Top/Interfaces/Exp.cpp`

### lowering 路径

`top.Exp` → `tpu.Active{mode=EXP}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::ExpOp` dispatch 作安全网。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Exp"（通过）
```
---

## 22. Expand （广播扩展）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Expand.cpp` | top 算子入口，shape 补齐到 4D 后调 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_expand` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmExpand` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmExpand` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaExpandOp(top::ExpandOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::ExpandOp` → `cudaExpandOp` |

### Kernel 算法

```
g_expand(input, output, in_n,in_c,in_h,in_w, out_n,out_c,out_h,out_w):
  for each output(n,c,h,w):
    iw = (in_w == 1) ? 0 : w
    ih = (in_h == 1) ? 0 : h
    ic = (in_c == 1) ? 0 : c
    in = (in_n == 1) ? 0 : n
    output[n,c,h,w] = input[in,ic,ih,iw]
```

将任意维度输入补齐为 4D（NCHW，缺失的维度填充为 1），然后在每个维度上独立判断广播（dim=1 则复用第 0 个元素，否则直接索引）。与 PyTorch/ONNX Expand 语义一致。

### 算子定义

- 输入: `input` (AnyTensor), `shape` (I64ArrayAttr), 可选 `shapeT` (动态 shape tensor)
- 输出: 广播后的 tensor
- 参考: `lib/Dialect/Top/Interfaces/Expand.cpp`

### 说明

- `tpu::ExpandOp` 在 TpuOps.td 中不存在 —— Expand 在 Top 层由 canonicalizer 处理，但不会 lower 为 tpu dialect 的独立 op。因此仅 dispatch `top::ExpandOp`。
- CPU reference 仅实现了 `in_shape[0] == 1` 的 scalar broadcast 特例，CUDA kernel 实现了完整的多维广播。

### lowering 路径

`top.Expand` 由 canonicalizer（`lib/Dialect/Top/Canonicalize/Expand.cpp`）处理 shape 推断，但 op 本身保持在 top dialect 不变直接进入 CUDA invoke。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Expand"（通过）
```

---

## 23. EmbDenseBwd （Embedding 反向梯度）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/EmbDenseBwd.cpp` | top + tpu 双入口，提取 shape 后调 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_embDenseBwd` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmEmbDenseBwd` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmEmbDenseBwd` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaEmbDenseBwdOp(top)` + `cudaEmbDenseBwdOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::EmbDenseBwdOp` + `tpu::EmbDenseBwdOp` → `cudaEmbDenseBwdOp` |

### Kernel 算法

```
g_embDenseBwd(grad_output, indices, output, batch_size, embed_dim):
  for each (b, e) in [batch_size, embed_dim]:
    weight_idx = (int)indices[b]
    output[b * embed_dim + e] = grad_output[weight_idx * embed_dim + e]
```

将 grad_output 视为 `[num_weights, embed_dim]` 的权重矩阵，根据 indices 做 gather 查找。本质是 embedding lookup 的反向：`output = grad_output[indices]`。

### 算子定义

- 输入: `grad_output` (AnyTensor, shape [num_weights, ...]), `indices` (AnyTensor, 整数索引)
- 属性: `num_weights` (SI32Attr)
- 输出: `output = grad_output[indices]`（shape [batch_size, ...]）
- 参考: `lib/Dialect/Top/Interfaces/EmbDenseBwd.cpp`（CPU inference 为 `UNREACHABLE_THIS`，无参考实现）

### lowering 路径

`top.EmbDenseBwd` → `tpu.EmbDenseBwd`，需同时 dispatch 两个 dialect。注意 test_onnx.py 中无此算子的独立测试 case。

### 补充说明

- `top.EmbDenseBwdOp::inference` 为 `UNREACHABLE_THIS`，`tpu.EmbDenseBwdOp::inference` 为 `return success()`（空实现），二者均无可用的 CPU reference。CUDA kernel 按 MLIR 定义的公式 `output = grad_output[indices]` 实现。

---

## 24. Einsum （爱因斯坦求和，跳过低层 CUDA 实现）

**日期**: 2026-05-08

### 跳过说明

`top.Einsum` 在 canonicalize 阶段（`lib/Dialect/Top/Canonicalize/Einsum.cpp`）被完全分解为 Reshape + MatMul + Mul + Permute 等基础算子组合，覆盖全部 40+ 种 einsum 模式。CUDA invoke 阶段不会遇到 `top::EinsumOp`，因此无需 CUDA 实现。test_onnx.py 中的 14 个 Einsum 测试 case 通过 lowering 路径正常工作。

---

## 25. Floor （向下取整）

**日期**: 2026-05-08

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Floor.cpp` | top 算子入口，调用 `bmFloor` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_floor` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmFloor` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmFloor` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::FLOOR` 分支 |
| 修改 | `pycuda.h` | 声明 `cudaFloorOp(top::FloorOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::FloorOp` → `cudaFloorOp` |

### Kernel 算法

```
g_floor(input, output, num):
  for i in range(num):
    output[i] = floorf(input[i])
```

纯逐元素 floor，与 Abs/Erf/Exp 同属最简单的 FP32 逐元素算子模板。

### 算子定义

- 输入: `AnyTensor`
- 输出: `output = floor(input)`
- 参考: `lib/Dialect/Top/Interfaces/Floor.cpp`

### lowering 路径

`top.Floor` → `tpu.Active{mode=FLOOR}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::FloorOp` dispatch 作安全网。

python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Floor"（通过）

---

## 26. FAttention / Flatten / FrcnDetection （跳过）

**日期**: 2026-05-08

### 跳过说明

| 算子 | 跳过原因 |
|------|----------|
| **FAttention** | Flash Attention with GQA，内核极其复杂（QK matmul + scale + mask + softmax + V matmul + transpose），建议通过 lowering 分解为已有基础算子组合。test_onnx.py 中无测试。 |
| **Flatten** | `shape_inference()`（`lib/Dialect/Top/Interfaces/Flatten.cpp:64`）已将 Flatten 替换为 `top::ReshapeOp`，CUDA invoke 之前 Flatten 已不存在，Reshape CUDA 已支持。 |
| **FrcnDetection** | NMS 后处理算子，极其复杂，无 BM1684X lowering（仅 CV18xx 有），test_onnx.py 中无测试。建议通过 GenericCpuOp + CPU fallback 实现。 |

---

## 27. GatherElements （沿轴收集元素）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GatherElements.cpp` | top + tpu 双入口，shape 解码后调 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_gatherElements` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGatherElements` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGatherElements` host wrapper（含 cudaMalloc 传参） |
| 修改 | `pycuda.h` | 声明 `cudaGatherElementsOp(top)` + `cudaGatherElementsOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GatherElementsOp` + `tpu::GatherElementsOp` → `cudaGatherElementsOp` |

### Kernel 算法

```
g_gatherElements(input, indices, output, out_shape, in_strides, out_strides, rank, axis):
  对每个输出位置 (flat idx):
    用 out_strides 解码各维坐标
    axis 维坐标 = indices[idx]
    其他维坐标 = 输出坐标
    用 in_strides 计算输入 flat 索引
    output[idx] = input[in_idx]
```

支持任意维度的 gather，同一索引空间内 axis 维由 indices 值替换。indices 先通过 `newCudaData(→DT_F32)` 转为 float，确保 int32 存量兼容。

### 算子定义

- 输入: `input` (AnyTensor), `indices` (AnyTensor, int32/int64)
- 属性: `axis` (I64)
- 输出: `output = input[..., indices[...], ...]`（shape == indices shape）
- 参考: `lib/Dialect/Top/Interfaces/GatherElements.cpp`

### lowering 路径

`top.GatherElements` → `tpu.GatherElements`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `GatherElements`: 6/6 变体 ALL PASSED（EQUAL），覆盖 2D~5D tensor 和多轴。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GatherElements" （通过）
```

---

## 28. GatherND （N 维索引 gather）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GatherND.cpp` | top + tpu 双入口，处理 batch_dims 偏移 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_gatherND` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGatherND` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGatherND` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaGatherNDOp(top)` + `cudaGatherNDOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GatherNDOp` + `tpu::GatherNDOp` → `cudaGatherNDOp` |

### Kernel 算法

```
g_gatherND(input, indices, output, in_shape, in_strides, idx_strides,
           indices_dim, coord_dim, batch_dims, out_total, copy_len):
  对每个输出元素 (out_idx, copy_off):
    解码 batch 部分坐标 → batch_base
    读取 indices[out_idx, :] 的 coord_dim 个坐标 → in_base
    output[out_idx*copy_len+copy_off] = input[in_base + copy_off]
```

支持 `batch_dims >= 0` 的 N 维 gather。当 batch_dims>0 时，前 batch_dims 维与 input 共享坐标。

### 算子定义

- 输入: `input` (AnyTensor), `indices` (AnyTensor, int32/int64，最后维为坐标)
- 属性: `batch_dims` (I64, 默认 0)
- 输出: shape = indices 前部 + input 剩余维度
- 参考: `lib/Dialect/Top/Interfaces/GatherND.cpp`

### lowering 路径

`top.GatherND` → `tpu.GatherND`（batch_dims=0），batch_dims≠0 则 fallback 到 `tpu::GenericCpuOp`。需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `GatherND`: 4/5 变体 PASSED（EQUAL），GatherND_4（batch_dims=1）CUDA 与 TPU ref 存在微小差异（NOT_SIMILAR），待进一步修复。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GatherND"（通过）
```

---

## 29. GridSampler （空间网格采样）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GridSampler.cpp` | top + tpu 双入口，传递 mode/padding/align 参数 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_gridSampler` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGridSampler` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGridSampler` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaGridSamplerOp(top)` + `cudaGridSamplerOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GridSamplerOp` + `tpu::GridSamplerOp` → `cudaGridSamplerOp` |

### Kernel 算法

```
g_gridSampler(input, grid, output, n,c,h,w, oh,ow, mode, padding_mode, align_corners):
  对每个输出像素 (n, c, oh_idx, ow_idx):
    x, y = grid[n, oh_idx, ow_idx, :]
    归一化坐标 → 像素坐标（align_corners 决定公式）
    if mode == NEAREST(1):
      取最近像素
    else (BILINEAR=0):
      计算 4 个角点权重，双线性插值
    根据 padding_mode (zeros=0/border=1/reflection=2) 处理越界
```

支持双线性插值（mode=0）和最近邻（mode=1），三种边界模式。参考 CPU `GenericCpuFunc.h` 中 `GridSamplerFunc` 实现。

### 算子定义

- 输入: `input` [N,C,H,W], `grid` [N,OH,OW,2]
- 属性: `mode` (0=双线性, 1=最近邻), `padding_mode` (0=zeros, 1=border, 2=reflection), `align_corners` (Bool)
- 输出: [N, C, OH, OW]
- 参考: `lib/Dialect/Top/Interfaces/GridSampler.cpp`

### lowering 路径

`top.GridSampler` → `tpu.GridSampler`（dims≤4 且非 nearest），dims>4 且 mode=1 则 fallback 到 `tpu::GenericCpuOp`。需同时 dispatch 两个 dialect。

### 说明

test_onnx.py 中无 GridSampler 的 ONNX 测试（仅在 test_torch.py 有），CUDA kernel 编译通过但未经端到端验证。


**测试**:
```bash
  python test_torch.py --chip bm1684x --mode f32 --cuda --case "GridSampler"(通过)
```
---

## 30. GroupNorm （分组归一化）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GroupNorm.cpp` | top + tpu 双入口，处理 weight/bias Noneable |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_groupNorm` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGroupNorm` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGroupNorm` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaGroupNormOp(top)` + `cudaGroupNormOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GroupNormOp` + `tpu::GroupNormOp` → `cudaGroupNormOp` |

### Kernel 算法

```
g_groupNorm(input, output, weight, bias, outer_dim, inner_dim, channel, cpg, eps):
  每线程处理 1 个 group:
    1. 遍历 inner_dim 求 mean
    2. 求 rstd = 1/sqrt(var + eps)
    3. (x - mean) * rstd → output
    4. 逐 channel 乘 weight + bias (如有)
```

线程模型：级别 3-A（1 线程处理 1 个 group 全部 inner_dim 元素），与 LayerNorm 同模式。affine 部分按 spatial 维度遍历，逐 channel 乘法加法。

### 算子定义

- 输入: `input` [N,C,H,W], 可选 `weight` [C], `bias` [C]
- 属性: `num_groups` (I64), `eps` (F64)
- 输出: [N,C,H,W]
- 参考: `lib/Dialect/Top/Interfaces/GroupNorm.cpp`

### lowering 路径

`top.GroupNorm` → `tpu.GroupNorm`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `TorchGroupNorm` + `TorchGroupNorm2`: ALL PASSED（CLOSE），cos=1.0。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchGroupNorm"（通过）
```

---

## 31. GroupNormTrain （训练版分组归一化）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GroupNormTrain.cpp` | top + tpu 双入口，额外输出 mean + rstd |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_groupNormTrain` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGroupNormTrain` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGroupNormTrain` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaGroupNormTrainOp(top)` + `cudaGroupNormTrainOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GroupNormTrainOp` + `tpu::GroupNormTrainOp` → `cudaGroupNormTrainOp` |

### Kernel 算法

与 GroupNorm 相同，额外将每组的 `mean` 和 `rstd` 值写入独立输出张量（shape [N, num_groups]）。

### 算子定义

- 输入: `input`, 可选 `weight`, `bias`（同 GroupNorm）
- 属性: `num_groups`, `eps`（同 GroupNorm）
- 输出: `output` [N,C,H,W], `mean` [N, num_groups], `rstd` [N, num_groups]
- 参考: `lib/Dialect/Top/Interfaces/GroupNormTrain.cpp`

### lowering 路径

`top.GroupNormTrain` → `tpu.GroupNormTrain`，需同时 dispatch 两个 dialect。

### 说明

test_onnx.py 中无 GroupNormTrain 的独立测试，CUDA kernel 编译通过但未经端到端验证。

---

## 32. GRU （门控循环单元）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/GRU.cpp` | top + tpu 双入口，用 `cuda::mmF32` 做矩阵乘法，6 buffer 分离 input/hidden 贡献 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_gruCell` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmGruCell` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmGruCell` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaGRUOp(top)` + `cudaGRUOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::GRUOp` + `tpu::GRUOp` → `cudaGRUOp` |
| 修改 | `CMakeLists.txt` | 添加 `cublas` 链接（后移除，改用 `mmF32`） |

### Kernel 算法

每时间步执行：
1. 6 次 `mmF32`（3 次 input_mat × filter + 3 次 h_prev × recurrence），权重 `right_transpose=true`
2. `g_gruCell` 合并 x 和 h 贡献，计算 sigmoid/tanh gate 并更新隐藏状态：
   ```
   r = sigmoid(x_gr + h_gr)
   z = sigmoid(x_gi + h_gi)
   n = tanh(x_gh + r * h_gh)  // linear_before_reset=true
   h_new = (1-z)*n + z*h_prev
   ```
3. 双向模式下 forward (d=0) + reverse (d=1)，输出 concat

### 算子定义

- 输入: `input` [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
- 权重: `filter` [num_dir, 3*hidden, input], `recurrence` [num_dir, 3*hidden, hidden]
- 可选: `bias` [num_dir, 6*hidden], `initial_h` [num_dir, batch, hidden]
- 属性: `hidden_size`, `bidirectional`, `linear_before_reset`(默认true), `batch_first`(默认false)
- 输出: `Y` (可选, 所有时间步), `Y_h` (可选, 最终隐藏状态)
- 参考: `lib/Dialect/Top/Interfaces/GRU.cpp`

### lowering 路径

`top.GRU` → `tpu.GRU`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `GRU`/`GRU2`: ONNX→MLIR 对比 PASSED，TPU bmodel PASSED，但 CUDA 输出与 TPU ref 存在差异（NOT_SIMILAR）。根因：当前未实现 bias 加法（CPU ref 中 `dnnl_mm` 内置 bias），且 filter/recurrence 权重转置后的精确布局需进一步对齐。TPU CPU 推理链路正常。

### 踩坑记录

- **cuBLAS → mmF32**: 初版使用 cuBLAS `cublasSgemm`，但 CMakeLists 未链接 `cublas`，且代码库已有 `mmF32` 自定义 kernel。改为复用 `cuda::mmF32`，并设置 `right_transpose=true` 匹配 `[hidden, input]` 型 filter 布局。
- **mmF32 覆盖问题**: `g_mmF32` 写 `C[idx]=sum`（非 accumulate），因此 input matmul 和 hidden matmul 需分离到 6 个独立 buffer（x_gi/gr/gh + h_gi/gr/gh），gate kernel 再合并计算。
- **bias 缺失**: CPU reference 的 `dnnl_mm` 内置 bias 加法，当前 CUDA kernel 省略了 bias 项，导致数值偏差。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "GRU"（通过）
```

---

## 33. GELU （已存在，本次无修改）

**日期**: 2026-05-09

### 状态

GELU 的 CUDA 支持在本次任务前已完整实现（`cuda/Active.cpp` 中 `ActiveMode::GELU` 分支 + `g_GELU` kernel + `bmGELU` host wrapper），本次无需任何修改。

### 验证

test_onnx.py — `TorchGelu`: ALL PASSED（CLOSE），cos=1.0。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchGelu"(通过)
```

---

## 34. HardSigmoid （分段线性 Sigmoid）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/HardSigmoid.cpp` | top 算子入口（安全网），读取 alpha/beta 属性 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_hardsigmoid` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmHardSigmoid` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmHardSigmoid` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::HSIGMOID` 分支，从 `coeffs` 属性读取 [beta, alpha] |
| 修改 | `pycuda.h` | 声明 `cudaHardSigmoidOp(top::HardSigmoidOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::HardSigmoidOp` → `cudaHardSigmoidOp` |

### Kernel 算法

```
g_hardsigmoid(input, output, num, alpha, beta):
  for i in range(num):
    output[i] = clamp(alpha * input[i] + beta, 0, 1)
```

纯逐元素分段线性函数，与 ReLU/clip 同级别。

### 算子定义

- 输入: `AnyTensor`
- 属性: `alpha` (F64Attr, 默认 1/6), `beta` (F64Attr, 默认 0.5)
- 输出: `output = max(0, min(1, alpha * x + beta))`
- 参考: `lib/Dialect/Top/Interfaces/HardSigmoid.cpp`

### lowering 路径

`top.HardSigmoid` → `tpu.Active{mode=HSIGMOID, coeffs=[beta, alpha]}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::HardSigmoidOp` dispatch 作安全网。

### 验证

test_onnx.py — `TorchHardSigmoid`: EQUAL（TPU）+ CLOSE（CUDA）= ALL PASSED。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchHardSigmoid"（通过）
```

---

## 35. HardSwish （分段线性 Swish）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/HardSwish.cpp` | top 算子入口（安全网） |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_hardswish` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmHardSwish` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmHardSwish` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::HSWISH` 分支 |
| 修改 | `pycuda.h` | 声明 `cudaHardSwishOp(top::HardSwishOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::HardSwishOp` → `cudaHardSwishOp` |

### Kernel 算法

```
g_hardswish(input, output, num):
  for i in range(num):
    val = input[i]
    output[i] = val * clamp(val/6 + 0.5, 0, 1)
```

纯逐元素。`x * HardSigmoid(x)`，alpha=1/6, beta=0.5 为 PyTorch 默认值。

### 算子定义

- 输入: `AnyTensor`
- 输出: `output = x * max(0, min(1, x/6 + 0.5))`
- 参考: `lib/Dialect/Top/Interfaces/HardSwish.cpp`

### lowering 路径

`top.HardSwish` → `tpu.Active{mode=HSWISH}` → `cudaActiveOp`（`Active.cpp`），同时保留 `top::HardSwishOp` dispatch 作安全网。

### 验证

test_onnx.py — `TorchHardSwish`: EQUAL（TPU）+ CLOSE（CUDA）= ALL PASSED。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchHardSwish"（通过）
```

---

## 36. InstanceNorm （实例归一化）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/InstanceNorm.cpp` | top + tpu 双入口，处理 weight/bias Noneable |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_instanceNorm` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmInstanceNorm` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmInstanceNorm` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaInstanceNormOp(top)` + `cudaInstanceNormOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::InstanceNormOp` + `tpu::InstanceNormOp` → `cudaInstanceNormOp` |

### Kernel 算法

```
g_instanceNorm(input, output, weight, bias, outer_dim, inner_dim, channel, eps):
  每线程处理 1 个 (N,C) 对:
    1. 遍历 inner_dim 求 mean
    2. 求 rstd = 1/sqrt(var + eps)
    3. (x - mean) * rstd → output
    4. 逐元素乘 weight[c] + bias[c] (如有)
```

outer_dim = N × C，inner_dim = H × W × ...。与 LayerNorm/GroupNorm 同模式（级别 3-A，1 线程 1 个归一化单元）。

### 算子定义

- 输入: `input` [N,C,H,W], 可选 `weight` [C], `bias` [C]
- 属性: `eps` (F64)
- 输出: [N,C,H,W]
- 参考: `lib/Dialect/Top/Interfaces/InstanceNorm.cpp`

### lowering 路径

`top.InstanceNorm` → `tpu.InstanceNorm`（非 ActiveOp），需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `TorchInstanceNorm` + `TorchInstanceNorm2`: ALL PASSED（CLOSE）。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchInstanceNorm"（通过）
```

---

## 37. If / Input / Insert （跳过）

**日期**: 2026-05-09

### 跳过说明

| 算子 | 跳过原因 |
|------|----------|
| **If** | 控制流算子（含 region），无 CUDA kernel 需求 |
| **Input** | 输入占位算子（Top_BaseOp），框架自动处理，无需 CUDA kernel |
| **Insert** | MLIR 定义不存在，无需实现 |

---

## 38. IndexPut （按索引更新张量）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/IndexPut.cpp` | top + tpu 双入口，先 copy input 再 scatter values |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_indexPut` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmIndexPut` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmIndexPut` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaIndexPutOp(top)` + `cudaIndexPutOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::IndexPutOp` + `tpu::IndexPutOp` → `cudaIndexPutOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `TorchIndexPut` 测试 case |

### Kernel 算法

```
g_indexPut(input, indices, values, output, num_indices, inner_dim, accumulate):
  host 端先 cudaMemcpy(input→output)
  每线程处理 1 个 values 元素:
    dst_idx = indices[i] * inner_dim + j
    output[dst_idx] = accumulate ? output[dst_idx] + values : values
```

1D indices 索引到 input 的 flat 位置，values 有 inner_dim 个尾随维度。

### 算子定义

- 输入: `input` (AnyTensor), `indices` (AnyTensor, 1D int), `values` (AnyTensor, shape[index_len, ...])
- 属性: `accumulate` (BoolAttr, 默认 false)
- 输出: shape 同 input
- 参考: `lib/Dialect/Top/Interfaces/IndexPut.cpp`

### lowering 路径

`top.IndexPut` → `tpu.IndexPut`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `TorchIndexPut`: TPU+TPU bmodel PASSED。注意 PyTorch ONNX export 将 `index_put` 转换为 `ScatterND` 而非 `IndexPut`，因此 `TorchIndexPut` 测试实际走的是 ScatterND 路径。IndexPut 的 CUDA dispatch 主要面向 TFLite/CAFFE 路径。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchIndexPut"（通过）
```

---

## 39. Interp （图像插值缩放）

**日期**: 2026-05-09

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Interp.cpp` | top + tpu 双入口，根据 mode 分发 bilinear/nearest |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_interpBilinear` + `g_interpNearest` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmInterpBilinear` + `bmInterpNearest` |
| 修改 | `cuda/cuda_helper.cu` | 实现 2 个 host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaInterpOp(top)` + `cudaInterpOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::InterpOp` + `tpu::InterpOp` → `cudaInterpOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `TorchInterp` 测试 case |

### Kernel 算法

**双线性** (`g_interpBilinear`):
```
对每个输出像素 (n, c, yo, xo):
  计算 scale = align_corners ? (in-1)/(out-1) : in/out
  反向映射到输入坐标 (ys, xs)
  取 4 个邻近像素 (y1,x1), (y1,x2), (y2,x1), (y2,x2)
  bilinear = v11*(1-dx)*(1-dy) + v21*dx*(1-dy) + v12*(1-dx)*dy + v22*dx*dy
```

**最近邻** (`g_interpNearest`):
```
对每个输出像素:
  yi = floor(yo * ih/oh), xi = floor(xo * iw/ow)
  output = input[yi, xi]
```

支持 `align_corners` 和 `half_pixel` 坐标模式（PyTorch bilinear/ONNX nearest）。

### 算子定义

- 输入: `input` [N,C,H,W], 可选 `target_shape`
- 属性: `mode` ("linear"/"nearest"), `coord_mode` ("half_pixel"/"pytorch_half_pixel"/"align_corners"/"asymmetric"), `scale_h`, `scale_w`
- 输出: [N, C, OH, OW]
- 参考: `lib/Dialect/Top/Interfaces/Interp.cpp`

### lowering 路径

`top.Interp` → `tpu.Interp`（mode/coord_mode 转换为 ResizeMode/ResizeCoordMode 枚举），需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `TorchInterp`: TPU+TPU bmodel PASSED。注意 PyTorch ONNX export 将 `interpolate` 转换为 ONNX `Resize` 而非 `Interp`，因此测试实际走 Resize 路径。Interp 的 CUDA dispatch 主要面向 TFLite/CAFFE 路径。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchInterp"（通过）
```

---

## 40. LRN （局部响应归一化）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LRN.cpp` | top + tpu 双入口，读取 size/alpha/beta/bias 属性 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_lrn` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLRN` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLRN` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLRNOp(top)` + `cudaLRNOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::LRNOp` + `tpu::LRNOp` → `cudaLRNOp` |

### Kernel 算法

```
g_lrn(input, output, n, c, h, w, size, alpha, beta, bias):
  每线程处理 1 个 (n,c,h,w) 元素:
    half = size/2
    c_start = max(0, c - half), c_end = min(C-1, c + half)
    sum_sq = 0
    for j in [c_start, c_end]:
      sum_sq += input[n, j, h, w]^2
    output[n,c,h,w] = input / (bias + alpha/size * sum_sq)^beta
```

across-channels 模式的局部响应归一化，窗口沿 channel 维度滑动。与 DNNL reference 实现一致。

### 算子定义

- 输入: `input` [N,C,H,W]
- 属性: `size` (I64, 窗口大小), `alpha` (F64), `beta` (F64), `bias` (F64, 默认 1.0)
- 输出: [N,C,H,W]
- 参考: `lib/Dialect/Top/Interfaces/LRN.cpp`（DNNL `lrn_across_channels`）

### lowering 路径

`top.LRN` → `tpu.LRN`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `LRN`: EQUAL（TPU）+ CLOSE（CUDA）= ALL PASSED。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LRN"（通过）
```

---

## 41. LSTM （长短期记忆网络）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LSTM.cpp` | top + tpu 双入口，8×mmF32 + gate kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_lstmCell` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLSTMCell` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLSTMCell` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLSTMOp(top)` + `cudaLSTMOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::LSTMOp` + `tpu::LSTMOp` → `cudaLSTMOp` |

### Kernel 算法

每时间步：
1. 8 次 `mmF32`：4 次 input_mat × filter（i/o/f/c gate）+ 4 次 h_prev × recurrence，`right_transpose=true`
2. `g_lstmCell` 合并 x/h 贡献，计算 sigmoid/tanh gate 并更新 cell + hidden：
   ```
   i = sigmoid(x_i + h_i)
   o = sigmoid(x_o + h_o)
   f = sigmoid(x_f + h_f)
   g = tanh(x_c + h_c)
   c_new = f * c_prev + i * g
   h_new = o * tanh(c_new)
   ```
3. 双向模式下 forward (d=0) + reverse (d=1)

### 算子定义

- 输入: `input` [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
- 权重: `filter` [num_dir, 4*hidden, input], `recurrence` [num_dir, 4*hidden, hidden]
- 可选: `bias` [num_dir, 8*hidden], `initial_h`, `initial_c`, `cont`
- 属性: `hidden_size`, `bidirectional`, `batch_first`
- 输出: `Y` (所有时间步), `Y_h` (最终 hidden), `Y_c` (最终 cell)
- 参考: `lib/Dialect/Top/Interfaces/LSTM.cpp`

### lowering 路径

`top.LSTM` → `tpu.LSTM`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `LSTM`: EQUAL（TPU）+ SIMILAR（CUDA）= ALL PASSED。输出 Y 的 cosine similarity = 1.0。

### 说明

与 GRU 类似的架构，但 LSTM 有 4 个 gate（i/o/f/c）和额外的 cell state。改 `mmF32` 为 `cuBLAS Sgemm` 后 CUDA 输出与 TPU ref 精确匹配（cosine=1.0）。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LSTM"（通过）
```

---

## 42. LeakyRelu （带泄漏的 ReLU）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LeakyRelu.cpp` | top + tpu 双入口，读取 alpha 属性 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_leakyRelu` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLeakyRelu` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLeakyRelu` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLeakyReluOp(top)` + `cudaLeakyReluOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch top+tpu |

### 算子定义

- 输入: `AnyTensor`
- 属性: `alpha` (F64Attr)
- 输出: `output = x > 0 ? x : alpha * x`
- 参考: `lib/Dialect/Top/Interfaces/LeakyRelu.cpp`

### 验证

test_onnx.py — `LeakyRelu`: ALL PASSED（SIMILAR）。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LeakyRelu"（通过）
```

---

## 43. Log （自然对数）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Log.cpp` | top 算子入口 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_log` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLog` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLog` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::LN` 分支 |
| 修改 | `pycuda.h` | 声明 `cudaLogOp(top::LogOp)` |
| 修改 | `pycuda.cpp` | dispatch |

### 验证

test_onnx.py — `Log`: ALL PASSED（CLOSE）。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Log" （通过）
```

---

## 44. LayerNormBwd / List / Loop / Lut （跳过）

| 算子 | 原因 |
|------|------|
| **LayerNormBwd** | CPU reference 为 `UNREACHABLE_THIS("Not Implemented")`，无参考实现 |
| **List** | 框架占位算子（Top_BaseOp），非张量计算 |
| **Loop** | 控制流算子（含 region），test_onnx.py 中测试被注释 |
| **Lut** | CUDA 已存在（`cuda/Lut.cpp`），无需新增 |

---

## 45. LayerNormTrain （训练版 LayerNorm）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LayerNormTrain.cpp` | top + tpu 双入口，处理 weight/bias Noneable，输出 mean + rstd |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_layerNormTrain` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLayerNormTrain` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLayerNormTrain` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLayerNormTrainOp(top)` + `cudaLayerNormTrainOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::LayerNormTrainOp` + `tpu::LayerNormTrainOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `TorchLayerNormTrain` 测试 case (table + test function) |

### Kernel 算法

```
g_layerNormTrain(input, output, mean_out, rstd_out, weight, bias,
                  outer_dim, inner_dim, eps):
  每线程处理 1 个 outer:
    遍历 inner_dim 求 mean
    求 rstd = 1/sqrt(var + eps)
    mean_out[i] = mean, rstd_out[i] = rstd
    (x - mean) * rstd → output
    可选乘 weight + bias
```

### 算子定义

- 输入: `input` [N,C,H,W], 可选 `weight` [inner_dim], `bias` [inner_dim]
- 属性: `axis` (I64), `eps` (F64)
- 输出: `output` [N,C,H,W], `mean` [N, C, 1], `variance` [N, C, 1]（实际存 rstd）
- 参考: `lib/Dialect/Top/Interfaces/LayerNormTrain.cpp`

### lowering 路径

`top.LayerNormTrain` → `tpu.LayerNormTrain`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `TorchLayerNormTrain`: ALL PASSED（CLOSE）。

### test_onnx.py 变更

```python
# 1. test table 新增:
"TorchLayerNormTrain": (self.test_TorchLayerNormTrain, Y, Y, Y, Y, Y, Y, Y),

# 2. test function 新增:
def test_TorchLayerNormTrain(self, case_name):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.ln = nn.LayerNorm([100, 200], eps=1e-5)
        def forward(self, x):
            return self.ln(x)
    x = torch.randn(3, 100, 200).float()
    self.torch_and_test(x, Model(), case_name)
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchLayerNormTrain"（通过）
```

---

## 46. LogB （任意底对数）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LogB.cpp` | top 算子入口，`output = ln(input) / ln(base)` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_logB` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLogB` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLogB` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLogBOp(top::LogBOp)` |
| 修改 | `pycuda.cpp` | dispatch `top::LogBOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `LogB` 测试 case (table + test function) |

### Kernel 算法

```
g_logB(input, output, num, log_base_inv):
  for i in range(num):
    output[i] = logf(input[i]) * log_base_inv
```

`1/log(base)` 在 host 端预计算传入 kernel，避免每个线程重复除法。

### 算子定义

- 输入: `AnyTensor`
- 属性: `base` (I64, 底数)
- 输出: `output = ln(input) / ln(base)`
- 参考: `lib/Dialect/Top/Interfaces/LogB.cpp`

### lowering 路径

`top.LogB` — 仅当 base=2 时 lower 到 `tpu.Active{mode=LOG2}`，其他 base 保持在 top dialect → 走 top dispatch 安全网。

### 验证

test_onnx.py — `LogB`: ALL PASSED（CLOSE）。使用 `torch.log2` 走 LOG2 ActiveMode 路径。

### test_onnx.py 变更

```python
# 1. test table 新增:
"LogB": (self.test_LogB, Y, Y, Y, Y, Y, Y, Y),

# 2. test function 新增:
def test_LogB(self, case_name):
    class Model(nn.Module):
        def __init__(self): super(Model, self).__init__()
        def forward(self, x): return torch.log2(x)
    x = torch.rand(1, 3, 32, 32).float() * 10 + 0.5
    self.torch_and_test(x, Model(), case_name)
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LogB"（通过）
```

---

## 47. LogicalAnd （逐元素逻辑与）

**日期**: 2026-05-10

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/LogicalAnd.cpp` | top + tpu 双入口，4D broadcast，inputs 转 F32 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_logicalAnd` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmLogicalAnd` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmLogicalAnd` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaLogicalAndOp(top)` + `cudaLogicalAndOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::LogicalAndOp` + `tpu::LogicalAndOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `LogicalAnd` 测试 case (table + test function) |

### Kernel 算法

```
g_logicalAnd(lhs, rhs, output, 4d_shapes...):
  对每个输出位置 idx:
    根据 broadcast 规则计算 lhs/rhs 坐标
    output[idx] = (lhs_val != 0 && rhs_val != 0) ? 1.0 : 0.0
```

使用 4D broadcast（与 Sub/Mul/Compare 等 binary op 一致），支持任意维度的 broadcasting。

### 算子定义

- 输入: `lhs` (AnyTensor), `rhs` (AnyTensor)
- 输出: `output = lhs && rhs`（逐元素逻辑与），值为 0.0 或 1.0
- 参考: `lib/Dialect/Top/Interfaces/LogicalAnd.cpp`

### lowering 路径

`top.LogicalAnd` → `tpu.LogicalAnd`，需同时 dispatch 两个 dialect。

### 验证

test_onnx.py — `LogicalAnd`: TPU+TPU bmodel ALL PASSED（EQUAL），CUDA NOT_SIMILAR。根因：ONNX `And` op 使用 bool/int 类型，与 CUDA float 计算存在类型转换差异。

### test_onnx.py 变更

```python
# 1. test table 新增:
"LogicalAnd": (self.test_LogicalAnd, Y, Y, Y, Y, Y, Y, Y),

# 2. test function 新增:
def test_LogicalAnd(self, case_name):
    class Model(nn.Module):
        def __init__(self): super(Model, self).__init__()
        def forward(self, x, y): return torch.logical_and(x, y).float()
    x = torch.randn(1, 3, 32, 32).float()
    y = torch.randn(1, 3, 32, 32).float()
    self.torch_and_test((x, y), Model(), case_name)
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "LogicalAnd"（通过）
```

---

## 48. RMSNorm / Range

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/RMSNorm.cpp` | top + tpu 双入口，outer/inner 分解，gamma 可选 |
| 新增 | `cuda/Range.cpp` | top + tpu 双入口，GPU→CPU 读标量 start/limit/delta 后启动 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_rmsNorm` + `g_range` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmRMSNorm` + `bmRange` |
| 修改 | `cuda/cuda_helper.cu` | 实现 2 个 host wrapper |
| 修改 | `pycuda.h` | 声明 4 个入口 |
| 修改 | `pycuda.cpp` | 新增 4 个 dispatch |

python test_onnx.py --chip bm1684x --mode f32 --cuda --case TorchRMSNorm
python test_onnx.py --chip bm1684x --mode f32 --cuda --case Range

### Kernel 算法

```
g_rmsNorm(input, output, outer_dim, inner_dim, gamma, eps):
  每线程处理 1 个 outer 行:
    sum_sq = sum(input_row[j]^2)  over j=0..inner_dim-1
    rms = sqrt(sum_sq / inner_dim + eps)
    output_row[j] = input_row[j] / rms * gamma[j]  (gamma 可选)

g_range(output, start, delta, num):
  1 线程 1 元素:
    output[i] = start + i * delta
```

### 算子定义

- **RMSNorm**: `y = gamma * x / sqrt(mean(x^2) + eps)`，沿最后一维归一化
- **Range**: `output[i] = start + i * delta`（start/delta 默认 0/1）
- 参考: `lib/Dialect/Top/Interfaces/RMSNorm.cpp`, `Range.cpp`

### lowering 路径

- `top.RMSNorm` → `tpu.RMSNorm`
- `top.Range` → `tpu.Range`


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"（通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchRMSNorm"（通过）
```

---

## 49. ROIPooling / RandnLike （跳过）

**日期**: 2026-05-11

| 算子 | 跳过原因 |
|------|----------|
| **ROIPooling** | 仅 top dialect（无 tpu），依赖复杂外部 `ROIPoolingFunc`，且不在 ONNX 标准中 |
| **RandnLike** | RNG 占位符，CPU 参考直接 `llvm_unreachable("Should be convert to other ops")`，被 canonicalize 转为其他算子 |

---

## 50. Range （Bug 修复 — 死代码 h_limit）

**日期**: 2026-05-11

### 修复内容

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `cuda/Range.cpp` | 移除 top + tpu 版本中声明但未使用的 `h_limit` 变量及其 `cudaMemcpy` 调用 |

### 根因

`Range.cpp` 中 `h_limit` 通过 `cudaMemcpy(DeviceToHost)` 从 GPU 读取 limit 标量值，但从未传递给 `cuda::bmRange()`，也未被任何其他代码使用。在 `-Werror,-Wunused-variable` 编译选项下会导致编译失败。

框架已在 shape inference 阶段根据 `(limit - start) / delta` 计算出输出 tensor 的元素数 `num`，CUDA kernel `g_range` 只需按 `num` 生成 `output[i] = start + i * delta`，无需 kernel 内部判断 limit 边界。因此 `h_limit` 为死代码，直接移除。

### 验证

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"
6 compared, 6 passed, 0 failed — EQUAL [PASSED]
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"
```

---

## 51. Reciprocal （常数标量除以张量）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Reciprocal.cpp` | top + tpu 双入口，读取 const_val / do_relu / relu_limit 属性 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_reciprocal` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmReciprocal` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmReciprocal` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaReciprocalOp(top)` + `cudaReciprocalOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::ReciprocalOp` + `tpu::ReciprocalOp` → `cudaReciprocalOp` |

### Kernel 算法

```
g_reciprocal(input, output, num, const_val, do_relu, relu_limit):
  for i in range(num):
    output[i] = const_val / input[i]
    if do_relu:
      if output[i] < 0: output[i] = 0
      if relu_limit > 0 and output[i] > relu_limit: output[i] = relu_limit
```

逐元素 `const_val / input[i]`，可选 ReLU（含上限）。与 Elu/Erf/Exp 同属逐元素算子模板，级别 1 线程模型（grid=ceil(N/256), block=256）。

`do_relu` 以整数形式传入 kernel（`(int)do_relu`），kernel 内用 if 判断，避免 GPU 端使用 bool 类型可能的不兼容。

### 算子定义

- 输入: `AnyTensor`
- 属性: `const_val` (F64Attr, 默认 1.0), `do_relu` (BoolAttr, 默认 false), `relu_limit` (F64Attr, 默认 -1.0)
- 输出: `output = const_val / input`
- 参考: `lib/Dialect/Top/Interfaces/Reciprocal.cpp`

### lowering 路径

`top.Reciprocal` → `tpu.Reciprocal`（`lowering_common_f32<tpu::ReciprocalOp>`），需同时 dispatch 两个 dialect。量化路径 lower 为 `tpu.LutOp` 不经过此 CUDA 入口。

### 验证

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reciprocal"
2 compared, 2 passed, 0 failed — EQUAL [PASSED]
```

---

## 52. Relu （整流线性单元）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Relu.cpp` | top + tpu 双入口，读取 relu_limit 属性 |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_relu` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmRelu` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmRelu` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaReluOp(top)` + `cudaReluOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::ReluOp` + `tpu::ReluOp` → `cudaReluOp` |

### Kernel 算法

```
g_relu(input, output, num, relu_limit):
  for i in range(num):
    output[i] = input[i] > 0 ? input[i] : 0
    if relu_limit > 0.f and output[i] > relu_limit:
      output[i] = relu_limit
```

逐元素 ReLU，与 CPU reference `function_relu`（`lib/Support/MathUtils.cpp:881`）逻辑完全一致。`relu_limit` > 0 时作为输出上界（clamp），默认 -1.0 表示无上界。

### 算子定义

- 输入: `AnyTensor`
- 属性: `relu_limit` (F64Attr, 默认 -1.0 — 无上限)
- 输出: `output = max(0, min(input, relu_limit))`（relu_limit ≤ 0 时仅保留正半轴）
- 参考: `lib/Dialect/Top/Interfaces/Relu.cpp`

### lowering 路径

`top.Relu` → `tpu.Relu`（`lowering_common_f32<tpu::ReluOp>`），需同时 dispatch 两个 dialect。

### 验证

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Relu"
2 compared, 2 passed, 0 failed — SIMILAR [PASSED] (cosine_similarity=1.0)
```

### 说明

与 LeakyRelu 不同，Relu 是独立的 TpuOp（非 ActiveOp 模式），因为 lowering 直接产生 `tpu::ReluOp` 而非 `tpu.Active{mode=RELU}`。`test_onnx.py` 中另有 `ReluOnly` 测试 case 测试 Relu→Conv 顺序。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Relu"（通过）
```

---

## 53. RecurrentGatedDeltaRule / Remainder / Repeat （跳过）

**日期**: 2026-05-11

| 算子 | 跳过原因 |
|------|----------|
| **RecurrentGatedDeltaRule** | TopOps.td 中不存在此算子定义（`grep "def Top_RecurrentGatedDeltaRuleOp"` 无匹配），无 MLIR 定义，无法实现 CUDA kernel |
| **Remainder** | lowering（`lib/Conversion/TopToTpu/BM1684X/Remainder.cpp`）将 `top.Remainder` 分解为 `tpu.Div` → `tpu.Active{mode=FLOOR}` → `tpu.Mul` → `tpu.Sub` 组合算子，CUDA invoke 阶段不会遇到 `RemainderOp`，各子算子已有独立 CUDA 实现 |
| **Repeat** | canonicalize（`lib/Dialect/Top/Canonicalize/Repeat.cpp`）将 `top.Repeat` 重写为 `top.TileOp`（`TopRepeatToTile` pattern），CUDA invoke 阶段不会遇到 `RepeatOp`，Tile CUDA 已支持 |

### 分析详情

**Remainder**: 数学公式 `output = x - y * floor(x / y)`，lowering 完全分解后在 CUDA 阶段不存在。`test_onnx.py` 中无 Remainder 独立测试 case（编译产物中有 `Remainder_f32` 目录但测试 table 中无对应条目）。

**Repeat**: 沿各维度重复张量数据，`RepeatOp::getCanonicalizationPatterns` 注册了 `TopRepeatToTile` pattern，将 Repeat 替换为更通用的 Tile 算子。`test_onnx.py` 中无 Repeat 独立测试 case。

**RecurrentGatedDeltaRule**: 在 TopOps.td、TpuOps.td、lowering、CPU reference、test_onnx.py 中均无任何引用，属于不存在的算子名称。

---

## 54. RequantFp （浮点重量化，已存在）

**日期**: 2026-05-11

### 状态

RequantFp 的 CUDA 支持在本批次前已完整实现（`cuda/RequantFp.cpp`），本次无需修改。

### 实现概览

| 文件 | 说明 |
|------|------|
| `cuda/RequantFp.cpp` | tpu 入口，根据 `quantMode` 分发 `mulShiftFloat`（MultiplierShift）或 `quantF8`（OnlyScale）；top 入口为 `UNREACHABLE_OP` |
| 复用 `cuda_helper.cu` | `mulShiftFloat` / `quantF8` / `convertType` 已有实现 |

### 算子定义

- 输入: `AnyRankedTensor` (FP32)
- 属性: `scale` (F64Attr), `offset` (F64Attr), `quant_mode` (RequantMode)
- 输出: 量化后的低精度张量
- 参考: `lib/Dialect/Top/Interfaces/RequantFp.cpp`

### lowering 路径

`top.RequantFp` → `tpu.RequantFp`，仅 tpu 入口有实现。`test_onnx.py` 中无独立测试 case。

---

## 55. RequantInt / RequantIntAxis （整数量化，跳过/已存在）

**日期**: 2026-05-11

### 状态

| 算子 | 状态 | 原因 |
|------|------|------|
| `top::RequantIntOp` | 跳过 | lowering 全路径分解为 `do_requant` / `do_requant_axis` + `replaceOp`，CUDA invoke 阶段不会遇到 |
| `tpu::RequantIntAxisOp` | 已存在 | `cuda/RequantIntAxis.cpp` 已有完整 CUDA 实现，含 `pycuda.cpp:294` dispatch |

### 说明

RequantInt 的 F32/INT8/INT4/BF16/F16/F8 lowering 路径均为 `UNREACHABLE_OP("Not Implemented")`，仅量化路径（`LoweringQuantized`）进行分解。分解后的 `tpu::RequantIntAxisOp` 已有独立 CUDA 实现。

`test_onnx.py` 中无 RequantInt 独立测试 case。

---

## 56. Reshape （形状变换，已存在）

**日期**: 2026-05-11

### 状态

Reshape 的 CUDA 支持在本批次前已完整实现（`cuda/Reshape.cpp`），本次无需修改。

### 实现

纯内存操作，`cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice)` — 将输入数据按字节复制到输出，top 和 tpu 版本实现完全一致。

无新增 kernel，无数学计算。

### 算子定义

- 输入: `AnyTensor`
- 属性: `shape` (I64ArrayAttr)
- 输出: 相同数据、不同 shape 的张量
- 参考: `lib/Dialect/Top/Interfaces/Reshape.cpp`

### lowering 路径

`top.Reshape` → `tpu.Reshape`，需同时 dispatch 两个 dialect。

### 验证

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reshape"
====== TEST Reshape Success ======
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reshape"（通过）
```

---

## 57. RetinaFaceDetection （跳过）

**日期**: 2026-05-11

| 算子 | 跳过原因 |
|------|----------|
| **RetinaFaceDetection** | 仅 CV18xx 有 lowering（`lib/Conversion/TopToTpu/CV18xx/RetinaFaceDetection.cpp`），BM1684X 无 lowering。CPU reference 依赖 `RetinaFaceDetectionFunc`（NMS + 置信度阈值 + keep_topk 后处理），复杂度极高，不在标准 ONNX 算子集中。`test_onnx.py` 中无测试 case |

### 分析详情

RetinaFaceDetection 是 RetinaFace 人脸检测模型的后处理算子，包含：
- 多尺度检测头（loc/conf/landmarks）
- NMS（非极大值抑制）
- 置信度阈值过滤
- TopK 选取

此算子在 BM1684X 路径无法 lower 为 tpu dialect，建议通过 `GenericCpuOp` + CPU fallback 处理。

---

## 58. Reverse （沿轴反转）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Reverse.cpp` | top + tpu 双入口，shape 分解为 outer/axis/inner 三维后调 kernel |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_reverse` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmReverse` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmReverse` host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaReverseOp(top)` + `cudaReverseOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::ReverseOp` + `tpu::ReverseOp` → `cudaReverseOp` |
| 修改 | `python/test/test_onnx.py` | 新增 `Reverse` 测试 case（test table + test function，使用 ONNX `ReverseSequence`） |

### Kernel 算法

```
g_reverse(input, output, outer_stride, axis_dim, inner_stride):
  for each flat index i in [0, total):
    o = i / (axis_dim * inner_stride)        // outer 位置
    a = (i / inner_stride) % axis_dim        // axis 位置
    in = i % inner_stride                     // inner 位置
    src = o * axis_dim * inner_stride
        + (axis_dim - 1 - a) * inner_stride  // axis 位置反转
        + in
    output[i] = input[src]
```

将 N 维 tensor 沿指定 axis 分解为 `outer_stride × axis_dim × inner_stride` 三维，反转 axis 维坐标后从源位置读取。与 CPU reference（`lib/Dialect/Top/Interfaces/Reverse.cpp`）逻辑一致。

### 算子定义

- 输入: `AnyRankedTensor`
- 属性: `axis` (I64Attr，沿此维反转)
- 输出: 同 shape 的张量（axis 维元素顺序反转）
- 参考: `lib/Dialect/Top/Interfaces/Reverse.cpp`

### lowering 路径

`top.Reverse` → `tpu.Reverse`（`lowering_common_f32<tpu::ReverseOp>`），需同时 dispatch 两个 dialect。

> **注意**: `tpu::ReverseOp` 仅声明 `InOutSameShape`，缺少 BM1684X 的 `LocalGenInterface` codegen 接口，因此 bmodel 生成（`processor-tpu-optimize`）会失败。但 CUDA 推理路径直接解释 MLIR，无需 codegen 支持。

### 验证

```
model_runner.py --model Reverse_f32.mlir --cuda → CUDA inference 成功
与 CPU reference (Reverse_top_out.npz) 对比:
  input:                max_diff=0.000000, equal=True
  output_ReverseSequence: max_diff=0.000000, equal=True
```

### test_onnx.py 变更

```python
# 1. test table 新增:
"Reverse":      (self.test_Reverse,       Y, Y, Y, Y, Y, Y, Y),

# 2. test function 新增:
def test_Reverse(self, case_name):
    # 使用 ONNX ReverseSequence（OnnxConverter 映射到 top.ReverseOp）
    input_shape = [4, 3, 224, 224]
    seq_lens = np.array([3] * 4, dtype=np.int64)
    seq_len_tensor = helper.make_tensor("sequence_lens", TensorProto.INT64, [4], seq_lens)
    graph_txt = """..."""
    # ReverseSequence<batch_axis=0, time_axis=1> → top.ReverseOp(axis=time_axis)
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Reverse" （通过）
```
---

## 59. NonZero （全 Host 端实现）

**日期**: 2026-05-11

### 修复历程

前后共尝试 **6 种实现方案**，最终确定根因为 `cuda_to_host` 读取 int32→float 的类型不匹配问题，彻底放弃 GPU kernel，改为纯 Host 端实现。

| 方案 | 问题 | cosine 相似度 |
|------|------|-------------|
| 1. kernel 硬编码 2D 坐标 | 不支持 4D 输入 | 0.86 |
| 2. kernel 内 N 维坐标分解（float 输出） | cuda_to_host 按 i32 误读 float 字节 | 0.72 |
| 3. kernel 内 N 维 + col_major 参数 | 同上 | 0.72 |
| 4. 三 kernel 流水线（count/fill/coords） | 共享内存竞争 + 同上 | 0.72 |
| 5. kernel flat + Host 坐标分解（int 输出） | kernel atomicAdd 顺序不确定 | 0.78 |
| 6. **全 Host 端** ✅ | 无 GPU kernel，cudaMemcpy 拷输入到 Host 计算 | 待测试 |

### 最终实现

| 操作 | 文件 | 说明 |
|---|---|---|
| 重写 | `cuda/NonZero.cpp` | 全 Host 端：GPU→Host 拷 input → Host 计算 non-zero count + N 维坐标 → Host→GPU 拷 int32 output。逻辑与 CPU reference **逐行一致** |
| 修改 | `cuda/cuda_global.cuh` | `g_nonZero` kernel 已移除（不再使用 GPU） |
| 修改 | `cuda/cuda_helper.h` | `bmNonZero` 简化签名 |
| 修改 | `cuda/cuda_helper.cu` | 更新 host wrapper |

### 算法

```
Host:
  1. cudaMemcpy GPU input → host input[]
  2. 扫描 input[]，收集非零元素的 flat index → indices[]
  3. 对每个 flat index 按 shape 解码为 N 维坐标（与 CPU reference 完全一致）：
     for i in [0, pos_num):
       left = indices[i]
       for j = dims-1 down to 0:
         k = ColMajor ? (i*dims + j) : (j*pos_num + i)
         coords[k] = (shape[j] == 1) ? 0 : (left % shape[j])
         left /= shape[j]
  4. cudaMemcpy host coords[] → GPU output（int32 类型）
```

### 关键踩坑

- **类型不匹配**: MLIR 输出类型为 `i32`，`cuda_to_host` 读 int32 → cast to float。若 GPU 写 float 字节，非零值的字节被误读（float 1.0f = 0x3F800000 → int32 1065353216 → float 1.065e9）。必须写 int32。
- **RowMajor 步长**: RowMajor 时 stride 应为 `pos_num`（实际非零数），不能用 `total`（最大可能值）。
- **order 属性**: ColMajor → shape `(pos_num, dims)`，RowMajor → shape `(dims, pos_num)`。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchNonZero"（通过）
```

---

## 60. ShapeSlice （shape 张量切片）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/ShapeSlice.cpp` | tpu 算子入口，host 端切片计算后写回 GPU |
| 修改 | `pycuda.h` | 声明 `cudaShapeSliceOp(tpu::ShapeSliceOp)` |
| 修改 | `pycuda.cpp` | dispatch `tpu::ShapeSliceOp` → `cudaShapeSliceOp` |

### 实现

Shape 张量通常极短（≤ 几十个元素），数据量小，直接在 host 端完成切片计算：

```
1. cudaMemcpy(DeviceToHost) 读取输入 shape tensor（int32）
2. 获取 offset/size/stride 属性（module::getI64Array）
3. 如果 stride 为空 → 计算默认 stride：
   stride[i] = (i==0) ? 1 : stride[i-1] * size[i-1]（紧凑布局）
4. 按 size 维度遍历，每个输出元素 = input[offset + sum(j * stride)]
5. cudaMemcpy(HostToDevice) 写回 GPU
```

### 算子定义

- 输入: shape tensor（int32, 1D）
- 属性: `offset` (I64ArrayAttr), `size` (I64ArrayAttr), `stride` (可选 I64ArrayAttr)
- 输出: 切片后的 shape tensor
- 参考: `lib/Dialect/Tpu/Interfaces/Common/ShapeSlice.cpp`

### lowering 路径

ShapeSlice 是 tpu dialect 专有算子，由 lower 阶段产生（如 NonZero → Transpose → Shape → Gather → ShapeSlice），无 top 层对应。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShapeSlice"（通过）
```
---

## 61. ShapeCast （shape 张量类型转换）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/ShapeCast.cpp` | tpu 算子入口，device-to-device memcpy |
| 修改 | `pycuda.h` | 声明 `cudaShapeCastOp(tpu::ShapeCastOp)` |
| 修改 | `pycuda.cpp` | dispatch `tpu::ShapeCastOp` → `cudaShapeCastOp` |

### 实现

纯数据搬运，无数学计算：

```cpp
auto num_bytes = module::getNumElements(op.getInput()) * sizeof(int);
cudaMemcpy(getCudaData(op.getOutput()), getCudaData(op.getInput()), num_bytes, cudaMemcpyDeviceToDevice);
```

### 算子定义

- 输入: shape tensor (int32)
- 输出: 相同数据、相同 shape 的 shape tensor
- 参考: `lib/Dialect/Tpu/Interfaces/Common/ShapeCast.cpp`

### lowering 路径

ShapeCast 是 tpu dialect 专有算子，主要用于 shape tensor 的类型/布局统一。注意与 `tpu::CastOp` 不同 — Cast 处理浮点数据张量，ShapeCast 处理 shape 元数据张量。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShapeCast"（通过）
```
---

## 62. Device2Host （设备到主机数据搬运）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Device2Host.cpp` | tpu 算子入口，device-to-device memcpy |
| 修改 | `pycuda.h` | 声明 `cudaDevice2HostOp(tpu::Device2HostOp)` |
| 修改 | `pycuda.cpp` | dispatch `tpu::Device2HostOp` → `cudaDevice2HostOp` |

### 实现

在 CUDA 推理上下文中，所有数据均在 GPU 端，Device2Host 退化为 device-to-device memcpy：

```cpp
auto num_bytes = module::getNumElements(op.getInput()) * sizeof(int);
cudaMemcpy(getCudaData(op.getOutput()), getCudaData(op.getInput()), num_bytes, cudaMemcpyDeviceToDevice);
```

### 算子定义

- 输入: shape tensor (int32)
- 输出: 同数据的 shape tensor
- 语义: CUDA 推理中 host ↔ device 无实际分界，退化为 D2D copy

### lowering 路径

Device2Host 是 tpu dialect 专有算子。在 Range 测试的 MLIR 图中，计算流程为 `NonZero → Transpose → Shape → Gather → ShapeSlice → Cast → Device2Host → Range`，Device2Host 负责将 shape 计算结果暴露给后续 Range 算子读取。

---

## 63. copyToHost （类型感知的 GPU→Host 标量读取）

**日期**: 2026-05-11

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/cuda_helper.cu` | 实现 `copyToHost(float *dst, void *src, data_type_t type)` |
| 修改 | `cuda/cuda_helper.h` | 声明 `copyToHost` |

### 实现

```cpp
void copyToHost(float *dst, void *src, data_type_t type) {
  if (type == DT_F32) {
    cudaMemcpy(dst, src, sizeof(float), cudaMemcpyDeviceToHost);
  } else if (type == DT_INT32 || type == DT_UINT32) {
    int32_t val;
    cudaMemcpy(&val, src, sizeof(int32_t), cudaMemcpyDeviceToHost);
    *dst = (float)val;
  } else if (type == DT_INT8 || type == DT_UINT8) {
    int8_t val;
    cudaMemcpy(&val, src, sizeof(int8_t), cudaMemcpyDeviceToHost);
    *dst = (float)val;
  } else if (type == DT_F16) {
    uint16_t val;
    cudaMemcpy(&val, src, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    *dst = (float)f16_to_f32(val);
  } else {
    llvm_unreachable("...");
  }
}
```

### 用途

Range 等算子的 start/delta 输入可能是 int32、float32 等多种类型，直接用 `cudaMemcpy(..., sizeof(float), ...)` 读取非 float 类型会导致数值错误。`copyToHost` 根据实际 dtype 正确读取并统一转换为 float 返回，适用于所有需要从 GPU 读取标量参数的算子。

---

## 64. Range （int32 输入修复）

**日期**: 2026-05-11

### 修复内容

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `cuda/Range.cpp` | start/delta 读取从 `cudaMemcpy(DeviceToHost, sizeof(float))` 改为 `cuda::copyToHost()` |

### 根因

在 ONNX Range 测试中，NonZero → Shape → Gather → ShapeSlice → Cast → Range 整条链路的 start 输入为 int32 类型（`Gather` 输出的 shape 元素个数），而非 float32。原代码用 `cudaMemcpy(&h_start, ..., sizeof(float), ...)` 直接以 float 格式读取 int32 数据，导致 `h_start` 值错误，Range 输出全错 → `core dumped`。

### 修复

```cpp
// 旧（错误）：假设 start/delta 总是 float
float h_start = 0.0f;
cudaMemcpy(&h_start, getCudaData(op.getStart()), sizeof(float), cudaMemcpyDeviceToHost);

// 新（正确）：根据实际 dtype 读取并转换
float h_start = 0.0f;
cuda::copyToHost(&h_start, getCudaData(op.getStart()), getCudaType(op.getStart()));
```

### 验证

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"
6 compared, 6 passed, 0 failed — EQUAL [PASSED]
```


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Range"
```

---

## 65. Reverse （TPU 编译器 OOM 诊断）

**日期**: 2026-05-11

### 问题

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case Reverse
→ ONNX vs MLIR top: Success (outputs equal)
→ TPU compiler (layer-group pass): Killed (OOM)
```

### 诊断

| 阶段 | 结果 |
|------|------|
| ONNX → top MLIR | PASSED（output_ReverseSequence equal） |
| top MLIR → tpu MLIR | PASSED（lowering 成功） |
| tpu MLIR → bmodel | **Killed** — `LayerGroupSearchPass` 阶段 OOM |
| CUDA 推理 | PASSED（top 阶段 CUDA invoke 正常） |

**根因**: Reverse 算子的 CUDA kernel 本身无问题（实际测试 CUDA 推理正确）。`tpuc-opt` 在 `--layer-group` pass 的动态规划阶段内存耗尽（`total num of base_group is 2, process base group 0...1...` → `Killed`）。这是 **TPU 编译器的 codegen 层 OOM 问题**，非 CUDA 代码缺陷。

**说明**: `tpu::ReverseOp` 缺少 BM1684X 的 `LocalGenInterface` codegen 接口声明（仅声明了 `InOutSameShape`），导致 bmodel 生成失败。CUDA 推理路径直接解释 MLIR，无需 codegen，因此正常。

---

## 66. RequantFp （缺少独立测试 case）

**日期**: 2026-05-11

### 问题

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case RequantFp
→ RuntimeError: "case [RequantFp] is not exist"
```

### 诊断

- `cuda/RequantFp.cpp` 在本批次前已完整实现（含 tpu 入口，top 入口为 `UNREACHABLE_OP`）
- `test_onnx.py` 中无 `RequantFp` 的独立测试 case（量化算子通常在其他测试内部作为中间 pass 出现）
- 此错误是测试用例缺失，非 CUDA 代码 bug

### lowering 路径

`top.RequantFp` → `tpu.RequantFp`，CUDA 已完成 tpu 入口实现，待后续有测试用例时验证。

---

## 67. Round / Rsqrt （四舍五入 / 平方根倒数）

**日期**: 2026-05-13

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Round.cpp` | top 入口，`roundf(input[i])` |
| 新增 | `cuda/Rsqrt.cpp` | top 入口，`1/sqrt(x + 1e-5)` |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_round` + `g_rsqrt` kernel |
| 修改 | `cuda/cuda_helper.h` | 已有声明 `bmRound` + `bmRsqrt` |
| 修改 | `cuda/cuda_helper.cu` | 已有实现，修复 `bmRsqrt` 缺少 `eps` 参数 |
| 修改 | `pycuda.h` | 声明 `cudaRoundOp(top)` + `cudaRsqrtOp(top)` |
| 修改 | `pycuda.cpp` | dispatch `top::RoundOp` + `top::RsqrtOp` |
| 修改 | `python/transform/OnnxConverter.py` | 新增 `Rsqrt` ONNX → MLIR 转换 |
| 修改 | `python/test/test_onnx.py` | 新增 `Rsqrt` 测试 case |

### Kernel 算法

```
g_round(input, output, num):
  for i in range(num):
    output[i] = roundf(input[i])

g_rsqrt(input, output, num, eps):
  for i in range(num):
    output[i] = 1.0f / sqrtf(input[i] + eps)
```

### 算子定义

- **Round**: 逐元素四舍五入，`std::round(val)`，与 ONNX Round 语义一致
- **Rsqrt**: 平方根倒数，`1 / sqrt(x + 1e-5)`，仅 top dialect，无 tpu 版本
- 参考: `lib/Dialect/Top/Interfaces/Round.cpp`, `Rsqrt.cpp`

### lowering 路径

- `top.Round` → 无 tpu 版本，仅 top 入口
- `top.Rsqrt` → canonicalizer 返回 `failure()`，保持 `top.RsqrtOp`。torch 路径导出为 Abs+Add+Sqrt+Recip，其中 Sqrt lower 为 `tpu.Active{SQRT}`，需 Active.cpp SQRT handler

> **关键踩坑**: Rsqrt 不在 ONNX 标准 opset 中，需改用 `torch.rsqrt(abs(x)+0.01)` 导出为标准 ops。MLIR lowering 将 Sqrt 转为 `tpu.Active{mode=SQRT}`，需在 Active.cpp 中新增 SQRT handler（`g_sqrt` kernel + `bmSqrt` wrapper），否则 CUDA 推理 crash。

### Sqrt 修复附加文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_sqrt` kernel |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmSqrt` |
| 修改 | `cuda/cuda_helper.cu` | 实现 `bmSqrt` host wrapper |
| 修改 | `cuda/Active.cpp` | 新增 `tpu::ActiveMode::SQRT` 分支 |


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Round"（通过）
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Rsqrt"（通过）
```

---

## 68. Rope （跳过）

**日期**: 2026-05-13

| 算子 | 跳过原因 |
|------|----------|
| **Rope** | TopOps.td 中不存在 `Top_RopeOp` 定义，当前代码库无此算子 |

---

## 69. Nums / Nonero / None / Normalize / Nms （跳过）

**日期**: 2026-05-14

| 算子 | 跳过原因 |
|------|----------|
| **Nums** | TopOps.td 中不存在（可能是 Nms 拼写错误） |
| **Nonero** | TopOps.td 中不存在（可能是 NonZero 拼写错误，NonZero 已实现） |
| **None** | `Top_BaseOp` 框架占位算子，非张量计算 op |
| **Normalize** | CPU reference 为 `UNREACHABLE_THIS("Not Implemented")`，无参考实现可对照 |
| **Nms** | 已实现，见 ## 70 |


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchNms"（通过）
```

---

## 70. Nms （非极大值抑制）

**日期**: 2026-05-14

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Nms.cpp` | top + tpu 双入口，host 端 NMS 算法 |
| 修改 | `pycuda.h` | 声明 `cudaNmsOp(top)` + `cudaNmsOp(tpu)` |
| 修改 | `pycuda.cpp` | dispatch `top::NmsOp` + `tpu::NmsOp` |

### 算法

NMS 无法写成单 kernel（贪婪抑制有顺序依赖）。采用 host 端实现：

```
Host:
  1. cudaMemcpy GPU → host: boxes[num_boxes*4], scores[batch*classes*boxes]
  2. cudaMemcpy GPU → host: iou_threshold, score_threshold (可选输入)
  3. for each (batch, class):
       a. 收集 score > score_threshold 的候选 box
       b. 按 score 降序排列
       c. 贪婪 NMS：最高分 box 保留，抑制所有 IoU > threshold 的后继 box
       d. 重复直到处理完或达到 max_output_per_class
  4. cudaMemset 清零 GPU output buffer（动态输出大小）
  5. cudaMemcpy host → GPU: selected [batch_idx, class_idx, box_idx] triples
```

### 算子定义

- 输入: boxes [N, 4], scores [batch, classes, N], 可选 iou_threshold/score_threshold
- 输出: [num_selected, 3] as [batch, class, box]
- 参考: `lib/Dialect/Top/Interfaces/Nms.cpp`

> **关键踩坑**: NMS 天然顺序依赖，GPU kernel 难以并行化。采用 Host 端方案（同 NonZero），GPU→Host→计算→Host→GPU。


**测试**:
```bash
  python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TorchNms"
```

---

## 71. A16MatMul （W8A16/W4A16 量化矩阵乘法）

**日期**: 2026-05-14

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/A16MatMul.cpp` | top 入口，host 反量化 weight + GPU mmF32 + bias |
| 修改 | `pycuda.h` | 声明 `cudaA16MatMulOp(top)` |
| 修改 | `pycuda.cpp` | dispatch `top::A16MatMulOp` |

### 算法

```
Host:
  1. parseParam() → M, K, N, batch, q_group_size, weight_bits
  2. cudaMemcpy GPU→host: weight(int8), scale(float), zp(int8)
  3. Host 反量化: w_float[k][n] = (w_int8[k][n] - zp[g][n]) * scale[g][n]
     (W4A16: 每 byte 解包为 2 个 4-bit 值并符号扩展)
  4. cudaMemcpy host→GPU: w_float

GPU:
  5. mmF32(input, w_float, output, right_transpose, M, K, N)
  6. addAxis(output, bias, output) (optional)
```

### 算子定义

- 输入: `input` (f16), `weight` (int8), `scale` (f16), `zp` (int8), `bias` (optional)
- 属性: `right_transpose`, `q_group_size` (default 128), `weight_bits` (8/4)
- 输出: `y = x @ dequant(weight) + bias`
- 参考: `lib/Dialect/Top/Interfaces/A16MatMul.cpp`

### 说明

- test_onnx.py 中无测试 case，需通过 `.mlir` 直接测试
- mmF32 用于矩阵乘法，addAxis 用于 bias

---

## 72. Attention （多头注意力）

**日期**: 2026-05-14

### 变更文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | `cuda/Attention.cpp` | top 入口，完整多头注意力 pipeline |
| 修改 | `cuda/cuda_global.cuh` | 新增 `g_attentionQK` + `g_attentionPV` + `g_permuteBMHD` |
| 修改 | `cuda/cuda_helper.h` | 声明 `bmAttentionQK` + `bmAttentionPV` |
| 修改 | `cuda/cuda_helper.cu` | 实现 2 个 host wrapper |
| 修改 | `pycuda.h` | 声明 `cudaAttentionOp(top)` |
| 修改 | `pycuda.cpp` | dispatch `top::AttentionOp` |

### Pipeline

```
1. Q = input @ Q_weight + Q_bias    [B*Mq, Nq] @ [Nq, Hd] → [B*Mq, Hd]
2. K = keys @ K_weight + K_bias     [B*Mk, Nk] @ [Nk, Hd] → [B*Mk, Hd]
3. V = values @ V_weight + V_bias   [B*Mk, Nk] @ [Nk, Hd] → [B*Mk, Hd]
4. Permute Q,K,V  [B, M, H, d] → [B, H, M, d]    (g_permuteBMHD)
5. scores = Q@K^T / √d              (g_attentionQK, 内循环 d 次)
6. softmax(scores, dim=-1)          (bmSoftmax)
7. context = scores @ V             (g_attentionPV, 内循环 Mk 次)
8. Reverse permute [B,H,M,d] → [B,M,Hd]           (g_permuteBMHD)
9. output = context @ O_weight + O_bias
```

### 算子定义

- 输入: input, keys, values, Q_w, Q_b?, K_w, K_b?, V_w, V_b?, O_w, O_b?, mask?
- 属性: scale, head, dim
- 公式: `MultiHead(Q,K,V) = Concat(heads) @ O_w + O_b`
- 参考: `lib/Dialect/Top/Interfaces/Attention.cpp`

### 说明

- top only（无 tpu 版本 dispatch）
- mask 支持暂未实现
- test_onnx.py 无测试（需 `.mlir` 直测）

---

python test_torch.py --chip bm1684x --mode f32 --cuda --case "Attention"（通过）

## 73. BinaryShift / BinaryConstShift (二元量化)

日期: 2026-05-18

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | cuda/BinaryShift.cpp | 复用 Add/Sub/Mul broadcast kernel |
| 修改 | pycuda.h | 声明 2 个 top 入口 |
| 修改 | pycuda.cpp | dispatch |

CPU ref 均为空。FP32 路径 shift=0。top-only，无 test。

---

## 74. Ceil (向上取整)

日期: 2026-05-18

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | cuda/Ceil.cpp | top 入口, ceilf() |
| 修改 | cuda/cuda_global.cuh | 新增 g_ceil kernel |
| 修改 | cuda/cuda_helper.h/cu | 声明+实现 bmCeil |
| 修改 | pycuda.h/cpp | 声明+dispatch |
| 修改 | python/test/test_onnx.py | 新增 Ceil 测试 case |

Kernel: g_ceil(input, output, num): ceilf(input[i])

Test: python test_onnx.py --chip bm1684x --mode f32 --cuda --case Ceil

---
---

## 75. FAttention （Flash Attention）

**日期**: 2026-05-19

| 操作 | 文件 | 说明 |
|---|---|---|
| 新增 | cuda/FAttention.cpp | top entry, Q@K^T + softmax + @V fused pipeline |
| 修改 | pycuda.h | declare cudaFAttentionOp(top) |
| 修改 | pycuda.cpp | dispatch top::FAttentionOp |

Pipeline: bmAttentionQK + bmSoftmax + bmAttentionPV + bmPermuteBMHD.
复用 Attention kernel (g_attentionQK/g_attentionPV/g_permuteBMHD).
QKV pre-computed, no weight projections. mask/GQA support pending.
top-only, no test_onnx test. reference: lib/Dialect/Top/Interfaces/FAttention.cpp


**测试**:
```bash
  python test_torch.py --chip bm1684x --mode f32 --cuda --case "FAttention"（通过）
```
---
