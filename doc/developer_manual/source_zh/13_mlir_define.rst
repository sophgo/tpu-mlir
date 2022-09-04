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
(待补充)

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
(待补充)

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
(待补充)

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
(待补充)

UpsampleOp
^^^^^^^^^^^^^^^
(待补充)

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

