![](./docs/assets/sophgo_chip.png)

# TPU-MLIR

本项目是算能智能AI芯片的TPU编译器工程。该工程提供了一套完整的工具链，其可以将不
同框架下预训练的神经网络，转化为可以在算能TPU上高效运算的二进制文件`bmodel`。

算能致力于成为全球领先的通用算力提供商。算能专注于AI、RISC-V CPU等算力产品的研发和推广应用，以自研产品为核心打造了覆盖“云、边、端”的全场景应用矩阵 ，为城市大脑、智算中心、智慧安防、智慧交通、安全生产、工业质检、智能终端等应用提供算力产品及整体解决方案 。公司在北京、上海、深圳、青岛、厦门等国内 10 多个城市及美国、新加坡等国家设有研发中心。

目前该工程直接支持的AI框架包括PyTorch、ONNX、TFLite和Caffe，其他框架模型需要转成ONNX。

# 资源

以下资源可以帮助你更好地了解TPU-MLIR：

| 序列 | 文档 |
| :---: | --- |
| 01 | [TPU-MLIR 论文](https://arxiv.org/abs/2210.15016) |
| 02 | [TPU-MLIR 开发参考手册](https://tpumlir.org/docs/developer_manual/index.html) |
| 03 | [TPU-MLIR 快速入门指南](https://tpumlir.org/docs/quick_start/index.html) |

| 序列 | 分享会 |
| :---: | --- |
| 01 | [TPU-MLIR 论文讲解](https://www.bilibili.com/video/BV1My4y1o73Q/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |
| 02 | [LayerGroup 讲解](https://www.bilibili.com/video/BV1wo4y1z7AG/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |

| 序列 | 主题 | 视频链接 |
| :---: | --- | --- |
| 01 | 什么是AI编译器？ | [AI编译器简介](https://www.bilibili.com/video/BV1yP4y1d7gz/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8)|
| 02 | MLIR 简介 | [基本语法(一)](https://www.bilibili.com/video/BV1CP411n7fj/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [基本语法(二)](https://www.bilibili.com/video/BV1Gt4y1F7mt/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [基本语法(三)](https://www.bilibili.com/video/BV1UN4y1w72r/?share_source=copy_web&vd_source=90fd7c624ed0c40dbaf08684), [Dialect Conversion](https://www.bilibili.com/video/BV1UG411c7nm/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [Pattern Rewriting](https://www.bilibili.com/video/BV1R44y1d7xv/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |
| 03 | TPU-MLIR 介绍 | [概述](https://www.bilibili.com/video/BV19d4y1B7eR/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [前端转换](https://www.bilibili.com/video/BV1yv4y1S7WT/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [Lowering](https://www.bilibili.com/video/BV1gg411z7mC/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |
| 04 | 量化 | [概述](https://www.bilibili.com/video/BV1d8411j7t4/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [公式推导](https://www.bilibili.com/video/BV1SW4y1H7Uu/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [校准](https://www.bilibili.com/video/BV1qK411R75k/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [QAT](https://www.bilibili.com/video/BV12g411J7WQ/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8)|
| 05 | TPU 内存 | [Ep1](https://www.bilibili.com/video/BV1T24y1G7pu/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [Ep2](https://www.bilibili.com/video/BV1VY4y1y7ET/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |
| 06 | TPU-MLIR 实践 | [转Onnx格式](https://www.bilibili.com/video/BV1FD4y1H7pT/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [图优化](https://www.bilibili.com/video/BV1AR4y1U7D6/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [算子支持](https://www.bilibili.com/video/BV1tL411r71p/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [模型支持](https://www.bilibili.com/video/BV1mM411y7Ep/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8), [融合预处理](https://www.bilibili.com/video/BV1ao4y1H7m8/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8ddhttpe8), [精度验证](https://www.bilibili.com/video/BV14e4y1M79d/?share_source=copy_web&vd_source=90fd7c624ed0c40af96748bd0b8dd3e8) |

此外，我们还为任何对我们的项目感兴趣并愿意与我们一同参与开发工作的朋友发布了一系列任务：

| 序列 | 任务列表 |
| :---: | --- |
| 01 | [Rewrite Patterns for PermuteOp](https://github.com/sophgo/tpu-mlir/issues/93) |
| 02 | [Shape Inference Implement](https://github.com/sophgo/tpu-mlir/issues/81) |
| 03 | [mlir2onnx tool Optimize](https://github.com/sophgo/tpu-mlir/issues/80) |

更多任务请查看[Issue](https://github.com/sophgo/tpu-mlir/issues)。

如果你在完成上述任务时有任何疑问，可以在我们的[问答平台](https://ask.tpumlir.org/questions)中提问或查看现有答案。

# 编译工程

克隆本工程代码后，需要在docker中编译。

* 从[dockerhub](https://hub.docker.com/r/sophgo/tpuc_dev)下载所需的镜像。

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 just a example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

容器建立后，代码在docker中的目录为`/workspace/tpu-mlir`。

* 编译代码

在工程目录下运行以下命令：

``` shell
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
# 使用方法

以`yolov5s.onnx`为例，介绍如何编译迁移一个onnx模型至BM1684X TPU平台运行。

该模型来在yolov5的官网: <https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx>。

在本工程已经放在`regression/model/yolov5s.onnx`。

## 准备模型和数据

建立`model_yolov5s`目录，注意是与本工程同级目录；并把模型文件和图片文件都放入`model_yolov5s`目录中。

操作如下：

``` shell
mkdir model_yolov5s && cd model_yolov5s
cp ${REGRESSION_PATH}/model/yolov5s.onnx .
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
mkdir workspace && cd workspace
```
## 将模型转化MLIR

如果模型是图片输入，在转模型之前我们需要了解模型的预处理。如果模型用预处理后的npz文件做输入，则不需要考虑预处理。
预处理过程用公式表达如下（x代表输入)：
$$
y = （x - mean） \times scale
$$

官网yolov5的图片是rgb，每个值会乘以`1/255`，转换成mean和scale对应为`0.0,0.0,0.0`和`0.0039216,0.0039216,0.0039216`。

模型转换命令如下：

``` shell
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

`model_transform.py`支持的参数如下:

| **参数名**           | 必选？ | **说明**            |
| ------------------- | ----- | ------------------- |
| model_name          | 是    | 指定模型名称          |
| model_def           | 是    | 指定模型定义文件，比如`.onnx`或`.pt`或`.tflite`或`.prototxt`文件 |
| model_data          | 否    | 指定模型权重文件，caffe模型需要，对应`.caffemodel`文件 |
| input_shapes        | 否    | 指定输入的shape，例如`[[1,3,640,640]]`；二维数组，可以支持多输入情况 |
| resize_dims         | 否    | 原始图片需要resize之后的尺寸；如果不指定，则resize成模型的输入尺寸 |
| keep_aspect_ratio   | 否    | 在Resize时是否保持长宽比，默认为false；设置时会对不足部分补0 |
| mean                | 否    | 图像每个通道的均值，默认为0.0,0.0,0.0                    |
| scale               | 否    | 图片每个通道的比值，默认为1.0,1.0,1.0                    |
| pixel_format        | 否    | 图片类型，可以是rgb、bgr、gray、rgbd四种情况              |
| output_names        | 否    | 指定输出的名称，如果不指定，则用模型的输出；指定后用该指定名称做输出 |
| test_input          | 否    | 指定输入文件用于验证，可以是图片或npy或npz；可以不指定，则不会正确性验证 |
| test_result         | 否    | 指定验证后的输出文件                                         |
| excepts             | 否    | 指定需要排除验证的网络层的名称，多个用,隔开                      |
| debug | 否 | 指定后保留中间临时文件；否则会清理掉中间临时文件 |
| mlir                | 是    | 指定输出的mlir文件路径                                       |

转成mlir文件后，会生成一个`${model_name}_in_f32.npz`文件，该文件是模型的输入文件。它是通过对图片输入进行预处理后得到的数据。


## MLIR转F16模型

将mlir文件转换成f16的bmodel，操作方法如下：

``` shell
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F16 \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --model yolov5s_1684x_f16.bmodel
```

model_deploy.py的相关参数说明如下：

| **参数名**           | 必选？ | **说明**                       |
| ------------------- | ----- | ----------------------------- |
| mlir                | 是    | 指定mlir文件                                              |
| quantize            | 是    | 指定默认量化类型，支持F32/BF16/F16/INT8                     |
| chip                | 是    | 指定模型将要用到的平台，支持bm1684x（后续会支持多款TPU平台）     |
| calibration_table   | 否    | 指定量化表路径，当存在INT8量化的时候需要量化表                 |
| tolerance           | 否    | 表示 MLIR 量化后的结果与 MLIR fp32推理结果相似度的误差容忍度 |
| correctnetss        | 否    | 表示仿真器运行的结果与MLIR量化后的结果相似度的误差容忍度，默认0.99,0.99 |
| excepts             | 否    | 指定需要排除验证的网络层的名称，多个用,隔开 |
| debug | 否 | 指定后保留中间临时文件；否则会清理掉中间临时文件 |
| model               | 是    | 指定输出的model文件路径                                  |
| dynamic             | 否    | 动态编译，支持动态shape                           |

## MLIR转INT8模型

转INT8模型前需要跑calibration，得到量化表；输入数据的数量根据情况准备100~1000张左右。

然后用量化表，生成对称或非对称bmodel。如果对称符合需求，一般不建议用非对称，因为非对称的性能会略差与对称模型。

这里用现有的100张来自COCO2017的图片举例，执行calibration：

``` shell
run_calibration.py yolov5s.mlir \
  --dataset ../COCO2017 \
  --input_num 100 \
  -o yolov5s_cali_table
```


转成INT8对称量化模型，执行如下命令：

``` shell
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize INT8 \
  --calibration_table yolov5s_cali_table \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --tolerance 0.85,0.45 \
  --model yolov5s_1684x_int8.bmodel
```



## 效果对比

本工程有用python写好的yolov5用例，源码路径`python/samples/detect_yolov5.py`，用于对图片进行目标检测。阅读该代码可以了解模型是如何使用的：先预处理得到模型的输入，然后推理得到输出，最后做后处理。以下用该代码分别来验证onnx/f32/int8的执行结果。

onnx模型的执行方式如下，得到`dog_onnx.jpg`：

``` shell
detect_yolov5.py \
  --input ../image/dog.jpg \
  --model ../yolov5s.onnx \
  --output dog_onnx.jpg
```


f16 bmodel的执行方式如下，得到`dog_f16.jpg`：

``` shell
detect_yolov5.py \
  --input ../image/dog.jpg \
  --model yolov5s_1684x_f16.bmodel \
  --output dog_f16.jpg
```


int8 **对称**bmodel的执行方式如下，得到dog_int8_sym.jpg：

``` shell
detect_yolov5.py \
  --input ../image/dog.jpg \
  --model yolov5s_1684x_int8.bmodel \
  --output dog_int8.jpg
```

三张图片对比如下：

![](./docs/quick_start/assets/yolov5s.png)



# 辅助工具

## 模型推理工具`model_runner.py`

支持 bmodel/mlir/onnx/tflite

``` shell
model_runner.py \
  --input resnet18_in_f32.npz \
  --model resnet18_1684x_f32.bmodel \
  --output resnet18_output.npz
```

## `bmodel`模型工具

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



# 相关链接

* [官方网站](https://tpumlir.org)
* [问答平台](https://ask.tpumlir.org)
* [教学视频](https://space.bilibili.com/1829795304/channel/collectiondetail?sid=734875)
