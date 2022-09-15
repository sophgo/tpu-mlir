MLIR定义
============

本章介绍MLIR各个元素的定义，包括Dialect、Interface等等

Top Dialect
---------------

Operations
~~~~~~~~~~~~~~~

AddOp
^^^^^^^^^^^^^^^

:简述:
    加法操作，:math:`Y = coeff_0 * X_0 + coeff_1 * X_1`

:输入:
    - inputs: tensor数组，对应2个或多个输入tensor

:输出:
    - output: tensor

:属性:
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限
    - coeff: 对应每个tensor的系数，默认为1.0

:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "tpu.Add"(%0, %1) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("add")


AvgPoolOp
^^^^^^^^^^^^^^^

:简述:
    将输入的tensor进行均值池化，:math:`S=\frac{1}{width\ *\ height}\sum_{i,j}a_{ij}` 。大小给定的滑动窗口会依次将输入tensor进行池化

    其中 :math:`width` 和 :math:`height` 表示kernel_shape的宽度和高度。 :math:`\sum_{i,j}a_{ij}` 则表示对kernel_shape进行求和
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - kernel_shape: 控制均值池化滑动窗口的大小
    - strides: 步长，控制滑动窗口每次滑动的距离
    - pads: 控制填充形状，方便池化
    - pad_value: 填充内容，常数，默认为0
    - count_include_pad: 结果是否需要对填充的pad进行计数
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %90 = "top.AvgPool"(%89) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool1_fwd_GlobalAveragePool")

Depth2SpaceOp
^^^^^^^^^^^^^^^
(待补充)

BatchNormOp
^^^^^^^^^^^^^^^
(待补充)

CastOp
^^^^^^^^^^^^^^^
(待补充)

ClipOp
^^^^^^^^^^^^^^^
(待补充)

ConcatOp
^^^^^^^^^^^^^^^
(待补充)

ConvOp
^^^^^^^^^^^^^^^
(待补充)

DeconvOp
^^^^^^^^^^^^^^^
(待补充)

DivOp
^^^^^^^^^^^^^^^
(待补充)

InputOp
^^^^^^^^^^^^^^^
(待补充)

LeakyReluOp
^^^^^^^^^^^^^^^
(待补充)

LSTMOp
^^^^^^^^^^^^^^^
(待补充)

LogOp
^^^^^^^^^^^^^^^
(待补充)

MaxPoolOp
^^^^^^^^^^^^^^^
:简述:
    将输入的tensor进行最大池化
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - kernel_shape: 控制均值池化滑动窗口的大小
    - strides: 步长，控制滑动窗口每次滑动的距离
    - pads: 控制填充形状，方便池化
    - pad_value: 填充内容，常数，默认为0
    - count_include_pad: 结果是否需要对填充的pad进行计数
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %8 = "top.MaxPool"(%7) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool0_fwd_MaxPool")

MatMulOp
^^^^^^^^^^^^^^^
(待补充)

MulOp
^^^^^^^^^^^^^^^
(待补充)

MulConstOp
^^^^^^^^^^^^^^^
(待补充)

PermuteOp
^^^^^^^^^^^^^^^
(待补充)

ReluOp
^^^^^^^^^^^^^^^
(待补充)

ReshapeOp
^^^^^^^^^^^^^^^
:简述:
    Reshape算子，返回一个给定形状的tensor，该tensor的类型和内部的值与输入tensor相同。reshape可能会对tensor的任何一行进行操作。在reshape过程中不会有任何数据的值被修改
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    无

:接口:
    无

:范例:
    .. code-block:: console

      %133 = "top.Reshape"(%132) : (tensor<1x255x20x20xf32>) -> tensor<1x3x85x20x20xf32> loc("resnetv22_flatten0_reshape0_Reshape")

ScaleOp
^^^^^^^^^^^^^^^
(待补充)

SigmoidOp
^^^^^^^^^^^^^^^
(待补充)

SiLUOp
^^^^^^^^^^^^^^^
(待补充)

SliceOp
^^^^^^^^^^^^^^^
(待补充)

SoftmaxOp
^^^^^^^^^^^^^^^
(待补充)

SqueezeOp
^^^^^^^^^^^^^^^
:简述:
    对输入tensor进行指定维度的裁剪并返回裁剪后的tensor
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - axes: 指定需要裁剪的维度，0代表第一个维度，-1代表最后一个维度

:接口:
    无

:范例:
    .. code-block:: console

      %133 = "top.Squeeze"(%132) {axes = [-1]} : (tensor<1x255x20x20xf32) -> tensor<1x255x20xf32> loc(#loc278)

UpsampleOp
^^^^^^^^^^^^^^^

:简述:
    上采样op，将输入tensor进行nearest上采样并返回tensor

:输入:
    tensor

:属性:
    - scale_h: 目标图像与原图像的高度之比
    - scale_w: 目标图像与原图像的宽度之比
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:输出:
    - output: tensor

:接口:
    无

:范例:
    .. code-block:: console

      %179 = "top.Upsample"(%178) {scale_h = 2 : i64, scale_w = 2 : i64} : (tensor<1x128x40x40xf32>) -> tensor<1x128x80x80xf32> loc("268_Resize")

WeightOp
^^^^^^^^^^^^^^^

:简述:
    权重op，包括权重的读取和创建，权重会存到npz文件中。权重的location与npz中的tensor名称是对应关系。

:输入:
    无

:属性:
    无

:输出:
    - output: 权重Tensor

:接口:
    - read: 读取权重数据，类型由模型指定
    - read_as_float: 将权重数据转换成float类型读取
    - read_as_byte: 将权重数据按字节类型读取
    - create: 创建权重op
    - clone_bf16: 将当前权重转换成bf16，并创建权重Op
    - clone_f16: 将当前权重转换成f16，并创建权重Op

:范例:
    .. code-block:: console

      %1 = "top.Weight"() : () -> tensor<32x16x3x3xf32> loc("filter")

