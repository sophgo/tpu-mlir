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

      %2 = "top.Add"(%0, %1) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("add")


AvgPoolOp
^^^^^^^^^^^^^^^
(待补充)

Depth2SpaceOp
^^^^^^^^^^^^^^^
(待补充)

BatchNormOp
^^^^^^^^^^^^^^^

:简述:
    在一个四维输入tensor上执行批标准化(Batch Normalization)。关于批标准化的更多细节可以参考论文《`Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 》。

    具体计算公式如下：

    .. math::

      y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

:输入:
    - input: 四维输入tensor
    - mean: input的均值tensor
    - variance: input的方差tensor
    - gamma: 公式中的 :math:`\gamma` tensor, 可以为None
    - beta: 公式中的 :math:`\beta` tensor, 可以为None

:输出:
    - output: 结果tensor

:属性:
    - epsilon: 公式中的 :math:`\epsilon` 常量，默认为1e-05
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %5 = "top.BatchNorm"(%0, %1, %2, %3, %4) {epsilon = 1e-05, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x27x27xf32> loc("BatchNorm")

CastOp
^^^^^^^^^^^^^^^
(待补充)

ClipOp
^^^^^^^^^^^^^^^
(待补充)

ConcatOp
^^^^^^^^^^^^^^^

:简述:
    将给定的tensor序列在给定的维度上连接起来。所有的输入tensor或者都具有相同的shape(待连接的维度除外)，或者都为空。

:输入:
    - inputs: tensor数组，对应2个或多个输入tensor

:输出:
    - output: 结果tensor

:属性:
    - axis: 待连接的维度的下标
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "top.Concat"(%0, %1) {axis = 1, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>)  -> tensor<1x6x27x27xf32> loc("Concat")

ConvOp
^^^^^^^^^^^^^^^

:简述:
    对输入tensor执行二维卷积操作。

    简单来说，给定输入大小为 :math:`(N, C_{\text{in}}, H, W)`，输出 :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` 的计算方法为：

    .. math::

      \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    其中 :math:`\star` 是有效的cross-correlation操作， :math:`N` 是batch的大小， :math:`C` 是channel的数量， :math:`H, W` 是输入图片的高和宽。

:输入:
    - input: 输入tensor
    - filter: 参数tensor，其形状为 :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`:
    - bias: 可学习的偏差tensor，形状为 :math:`(out_channels)`.

:输出:
    - output: 结果tensor

:属性:
    - kernel_shape: 卷积核的尺寸
    - strides: 卷积的步长
    - pads: 输入的每一条边补充0的层数
    - group: 从输入通道到输出通道的阻塞连接数，默认为1
    - dilations: 卷积核元素之间的间距，可选
    - inserts: 可选
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "top.Conv"(%0, %1) {kernel_shape = [3, 5], strides = [2, 1], pads = [4, 2]} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Conv")

DeconvOp
^^^^^^^^^^^^^^^

:简述:

    对输入tensor执行反卷积操作。

:输入:
    - input: 输入tensor
    - filter: 参数tensor，其形状为 :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`:
    - bias: 可学习的偏差tensor，形状为 :math:`(out_channels)`.

:输出:
    - output: 结果tensor

:属性:
    - kernel_shape: 卷积核的尺寸
    - strides: 卷积的步长
    - pads: 输入的每一条边补充0的层数
    - group: 从输入通道到输出通道的阻塞连接数，默认为1
    - dilations: 卷积核元素之间的间距，可选
    - inserts: 可选
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "top.Deconv"(%0, %1) {kernel_shape = (3, 5), strides = (2, 1), pads = (4, 2)} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Deconv")


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

:简述:
    Scale操作 :math:`Y = X * S + B`，其中X/Y的shape为[N, C, H, W]，S/B的shape为[1, C, 1, ,1]。

:输入:
    - input: 输入tensor
    - scale: 保存input的放大倍数
    - bias: 放大后加上的bias

:输出:
    - output: 结果tensor

:属性:
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限

:接口:
    无

:范例:
    .. code-block:: console

      %3 = "top.Scale"(%0, %1, %2) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x27x27xf32> loc("Scale")


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


