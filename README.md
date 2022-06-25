![](./design/assets/sophgo_chip.png)

# TPU-MLIR

本项目是算能智能AI芯片的TPU编译器工程。该工程提供了一套完整的工具链，其可以将不
同框架下预训练的神经网络，转化为可以在算能TPU上高效运算的二进制文件`bmodel`。

算能承续了比特大陆在AI领域沉淀多年的技术、专利、产品和客户，以成为全球领先的通用
算力提供商为愿景，智算赋能数字世界为使命，专注于人工智能芯片、RISC-V指令集高性能
CPU服务器以及相关产品的研发与销售。旗下算丰全系列人工智能产品包括智算芯片、智算
模组、智算盒子、智算卡、智算服务器等，丰富的产品形态为各型数据中心提供高效能的计
算平台。公司具备全球领先的先进制程设计能力，现已成功量产云端、边端人工智能芯片并
规模化商业落地。

更多关于**TPU-MLIR**的信息可以参考[TPU-MLIR设计文档](./design/design.md)。

# 编译工程

* 从[dockerhub](https://hub.docker.com/r/sophgo/sophgo_dev)下载所需的镜像。

``` shell
docker pull sophgo/sophgo_dev:1.2-ubuntu-18.04

# myname1234 just a example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/work -it sophgo/sophgo_dev:1.1-ubuntu-18.04
```

* 编译代码

克隆本工程，并在工程目录下运行以下命令：

``` shell
source ./envsetup.sh
./build.sh
```

# 代码验证

``` shell
pushd regression
./run.sh
popd
```

# 使用方法

工具链相关串通流程可以参考：[regression](./regression/basic/run_basic.sh)

## 将外部框架的模型转化为`mlir`(top)模型

``` shell
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --model_def  ../resnet18.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ../image/cat.jpg \
    --test_result resnet18_top_outputs.npz \
    --mlir resnet18.mlir
```

## 部署

将`mlir`模型转化为`bmodel`流程

### 转化为 fp32 `bmodel`

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet18_1684x_f32.bmodel
```

### 转化为int8 `bmodel`

该过程需要经过量化。

准备输入数据：将图片放置到单独文件夹下，此处为 "dataset"，然后进行量化标定。

``` shell
run_calibration.py resnet18.mlir \
    --dataset ../image \
    --input_num 2 \
    -o resnet18_cali_table
```

使用上述的量化表，将模型转化为int8 `bmodel`。

### 对称量化

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --calibration_table resnet18_cali_table \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.95,0.72 \
  --correctness 0.99,0.85 \
  --model resnet18_1684x_int8_sym.bmodel
```

### 非对称量化

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table resnet18_cali_table \
  --chip bm1684x \
  --test_input resnet18_in_f32.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.97,0.75 \
  --correctness 0.99,0.85 \
  --model resnet18_1684x_int8_asym.bmodel
```

## 辅助工具

### 模型推理工具`model_runner.py`

支持 bmodel/mlir/onnx/tflite

``` shell
model_runner.py \
  --input resnet18_in_f32.npz \
  --model resnet18_1684x_f32.bmodel \
  --output resnet18_output.npz
```

### `bmodel`模型工具

可以通过`model_tool`工具来查看和编辑`bmodel`文件, 用法参考以下列表:

```
  model_tool
    --info model_file : show brief model info
    --print model_file : show detailed model info
    --extract model_file : extract one multi-net bmodel to multi one-net bmodels
    --combine file1 .. fileN -o new_file: combine bmodels to one bmodel by filepath
    --combine_dir dir1 .. dirN -o new_dir: combine bmodels to one bmodel by directory path
    --dump model_file start_offset byte_size out_file: dump binary data to file from bmodel
```

例如, 获取`bmodel`的基本信息：

``` shell
model_tool --info resnet18_1684x_f32.bmodel
```
