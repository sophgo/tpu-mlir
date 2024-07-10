MLIR定义
============

本章介绍MLIR各个元素的定义, 包括Dialect、Interface等等

Top Dialect
---------------

Operations
~~~~~~~~~~~~~~~

AddOp
^^^^^^^^^^^^^^^

:简述:
    加法操作, :math:`Y = coeff_0 * X_0 + coeff_1 * X_1`

:输入:
    - inputs: tensor数组, 对应2个或多个输入tensor

:输出:
    - output: tensor

:属性:
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限
    - coeff: 对应每个tensor的系数, 默认为1.0

:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Add"(%0, %1) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("add")


AvgPoolOp
^^^^^^^^^^^^^^^

:简述:
    将输入的tensor进行均值池化, :math:`S=\frac{1}{width\ *\ height}\sum_{i,j}a_{ij}` 。大小给定的滑动窗口会依次将输入tensor进行池化

    其中 :math:`width` 和 :math:`height` 表示kernel_shape的宽度和高度。 :math:`\sum_{i,j}a_{ij}` 则表示对kernel_shape进行求和
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - kernel_shape: 控制均值池化滑动窗口的大小
    - strides: 步长, 控制滑动窗口每次滑动的距离
    - pads: 控制填充形状, 方便池化
    - pad_value: 填充内容, 常数, 默认为0
    - count_include_pad: 结果是否需要对填充的pad进行计数
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %90 = "top.AvgPool"(%89) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool1_fwd_GlobalAveragePool")

Depth2SpaceOp
^^^^^^^^^^^^^^^

:简述:
    深度转空间操作, :math:`Y = Depth2Space(X)`

:输入:
    - inputs: tensor

:输出:
    - output: tensor

:属性:
    - block_h: tensor 高度改变的参数, i64类型
    - block_w: tensor 宽度改变的参数, i64类型
    - is_CRD: column-row-depth, 如果true, 则数据沿深度方向的排布按照HWC, 否则为CHW, bool类型
    - is_inversed: 如果true, 那么结果的形状为:  :math:`[n, c * block_h * block_w, h / block_h, w / block_w]`,
                    否则结果的形状为: :math:`[n, c / (block_h * block_w), h * block_h, w * block_w]`

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Depth2Space"(%0) {block_h = 2, block_w = 2, is_CRD = true, is_inversed = false} : (tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32> loc("add")


BatchNormOp
^^^^^^^^^^^^^^^

:简述:
    在一个四维输入tensor上执行批标准化(Batch Normalization)。关于批标准化的更多细节可以参考论文《`Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 》。

    具体计算公式如下:

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
    - epsilon: 公式中的 :math:`\epsilon` 常量, 默认为1e-05
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %5 = "top.BatchNorm"(%0, %1, %2, %3, %4) {epsilon = 1e-05, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x27x27xf32> loc("BatchNorm")

CastOp
^^^^^^^^^^^^^^^
(待补充)

ClipOp
^^^^^^^^^^^^^^^
:简述:
      将给定输入限制在一定范围内

:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - min: 给定的下限
    - max: 给定的上限

:输出:
    - output: 输出tensor
:接口:
    无

:范例:
    .. code-block:: shell

      %3 = "top.Clip"(%0) {max = 1%: f64,min = 2%: f64} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Clip")

ConcatOp
^^^^^^^^^^^^^^^

:简述:
    将给定的tensor序列在给定的维度上连接起来。所有的输入tensor或者都具有相同的shape(待连接的维度除外), 或者都为空。

:输入:
    - inputs: tensor数组, 对应2个或多个输入tensor

:输出:
    - output: 结果tensor

:属性:
    - axis: 待连接的维度的下标
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Concat"(%0, %1) {axis = 1, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>)  -> tensor<1x6x27x27xf32> loc("Concat")

ConvOp
^^^^^^^^^^^^^^^

:简述:
    对输入tensor执行二维卷积操作。

    简单来说, 给定输入大小为 :math:`(N, C_{\text{in}}, H, W)`, 输出 :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` 的计算方法为:

    .. math::

      \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    其中 :math:`\star` 是有效的cross-correlation操作,  :math:`N` 是batch的大小,  :math:`C` 是channel的数量,  :math:`H, W` 是输入图片的高和宽。

:输入:
    - input: 输入tensor
    - filter: 参数tensor, 其形状为 :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`:
    - bias: 可学习的偏差tensor, 形状为 :math:`(out\_channels)`.

:输出:
    - output: 结果tensor

:属性:
    - kernel_shape: 卷积核的尺寸
    - strides: 卷积的步长
    - pads: 输入的每一条边补充0的层数
    - group: 从输入通道到输出通道的阻塞连接数, 默认为1
    - dilations: 卷积核元素之间的间距, 可选
    - inserts: 可选
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Conv"(%0, %1) {kernel_shape = [3, 5], strides = [2, 1], pads = [4, 2]} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Conv")

DeconvOp
^^^^^^^^^^^^^^^

:简述:

    对输入tensor执行反卷积操作。

:输入:
    - input: 输入tensor
    - filter: 参数tensor, 其形状为 :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`:
    - bias: 可学习的偏差tensor, 形状为 :math:`(out\_channels)`.

:输出:
    - output: 结果tensor

:属性:
    - kernel_shape: 卷积核的尺寸
    - strides: 卷积的步长
    - pads: 输入的每一条边补充0的层数
    - group: 从输入通道到输出通道的阻塞连接数, 默认为1
    - dilations: 卷积核元素之间的间距, 可选
    - inserts: 可选
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Deconv"(%0, %1) {kernel_shape = (3, 5), strides = (2, 1), pads = (4, 2)} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Deconv")


DivOp
^^^^^^^^^^^^^^^

:简述:
    除法操作, :math:`Y = X_0 / X_1`

:输入:
    - inputs: tensor数组, 对应2个或多个输入tensor

:输出:
    - output: tensor

:属性:
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限
    - multiplier: 量化用的乘数, 默认为1
    - rshift: 量化用的右移, 默认为0

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Div"(%0, %1) {do_relu = false, relu_limit = -1.0, multiplier = 1, rshift = 0} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("div")


InputOp
^^^^^^^^^^^^^^^
(待补充)

LeakyReluOp
^^^^^^^^^^^^^^^
:简述:
    tensor中每个元素执行LeakyRelu函数, 函数可表示为: f(x) = alpha * x for x < 0, f(x) = x for x >= 0
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - alpha:对应每个tensor的系数

:接口:
    无

:范例:
    .. code-block:: shell

      %4 = "top.LeakyRelu"(%3) {alpha = 0.67000001668930054 : f64} : (tensor<1x32x100x100xf32>) -> tensor<1x32x100x100xf32> loc("LeakyRelu")


LSTMOp
^^^^^^^^^^^^^^^
:简述:
    执行RNN 的LSTM操作

:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - filter:卷积核
    - recurrence: 循环单元
    - bias: LSTM的参数: 偏置
    - initial_h: LSTM中的每句话经过当前cell后会得到一个state,state 是个tuple(c, h), 其中h=[batch_size, hidden_size]
    - initial_c: c=[batch_size, hidden_size]
    - have_bias: 是否设置偏置bias, 默认为false
    - bidirectional: 设置双向循环的LSTM, 默认为false
    - batch_first: 是否将batch放在第一维, 默认为false

:接口:
    无

:范例:
    .. code-block:: shell

     %6 = "top.LSTM"(%0, %1, %2, %3, %4, %5) {batch_first = false, bidirectional = true, have_bias = true} : (tensor<75x2x128xf32>,tensor<2x256x128xf32>, tensor<2x256x64xf32>, tensor<2x512xf32>, tensor<2x2x64xf32>, tensor<2x2x64xf32>) -> tensor<75x2x2x64xf32> loc("LSTM")

LogOp
^^^^^^^^^^^^^^^
:简述:
    按元素计算给定输入张量的自然对数

:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    无

:接口:
    无

:范例:
    .. code-block:: shell

     %1 = "top.Log"(%0) : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Log")

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
    - strides: 步长, 控制滑动窗口每次滑动的距离
    - pads: 控制填充形状, 方便池化
    - pad_value: 填充内容, 常数, 默认为0
    - count_include_pad: 结果是否需要对填充的pad进行计数
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %8 = "top.MaxPool"(%7) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool0_fwd_MaxPool")

MatMulOp
^^^^^^^^^^^^^^^

:简述:
    二维矩阵乘法操作, :math:`C = A * B`

:输入:
    - input: tensor: m*k 大小的矩阵
    - right: tensor: k*n 大小的矩阵

:输出:
    - output: tensor m*n 大小的矩阵

:属性:
    - bias: 偏差, 量化的时候会根据bias计算 bias_scale, 可以为空
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.MatMul"(%0, %1) {do_relu = false, relu_limit = -1.0} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32> loc("matmul")


MulOp
^^^^^^^^^^^^^^^

:简述:
    乘法操作, :math:`Y = X_0 * X_1`

:输入:
    - inputs: tensor数组, 对应2个或多个输入tensor

:输出:
    - output: tensor

:属性:
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限
    - multiplier: 量化用的乘数, 默认为1
    - rshift: 量化用的右移, 默认为0

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Mul"(%0, %1) {do_relu = false, relu_limit = -1.0, multiplier = 1, rshift = 0} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("mul")


MulConstOp
^^^^^^^^^^^^^^^

:简述:
    和常数做乘法操作, :math:`Y = X * Const_Val`

:输入:
    - inputs: tensor

:输出:
    - output: tensor

:属性:
    - const_val: f64类型的常量
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %1 = arith.constant 4.7 : f64
      %2 = "top.MulConst"(%0) {do_relu = false, relu_limit = -1.0} : (tensor<1x3x27x27xf64>, %1) -> tensor<1x3x27x27xf64> loc("mulconst")


PermuteOp
^^^^^^^^^^^^^^^
:简述:
    改变tensor布局, 变化tensor数据维度的顺序, 将输入的tensor按照order给定的顺序重新布局

:输入:
    - inputs: tensor数组, 任意类型的tensor


:属性:
    - order: 指定重新布局tensor的顺序


:输出:
    - output: 输出tensor, 按order的顺序重新布局后的tensor

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Permute"(%1) {order = [0, 1, 3, 4, 2]} : (tensor<4x3x85x20x20xf32>) -> tensor<4x3x20x20x85xf32> loc("output_Transpose")



ReluOp
^^^^^^^^^^^^^^^
:简述:
    tensor中每个元素执行ReLU函数, 如果极限为零, 则不使用上限
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
   - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限。

:接口:
    无

:范例:
    .. code-block:: shell

      %1 = "top.Relu"(%0) {relu_limit = 6.000000e+00 : f64} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Clip")

ReshapeOp
^^^^^^^^^^^^^^^
:简述:
    Reshape算子, 返回一个给定形状的tensor, 该tensor的类型和内部的值与输入tensor相同。reshape可能会对tensor的任何一行进行操作。在reshape过程中不会有任何数据的值被修改
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    无

:接口:
    无

:范例:
    .. code-block:: shell

      %133 = "top.Reshape"(%132) : (tensor<1x255x20x20xf32>) -> tensor<1x3x85x20x20xf32> loc("resnetv22_flatten0_reshape0_Reshape")

ScaleOp
^^^^^^^^^^^^^^^

:简述:
    Scale操作 :math:`Y = X * S + B`, 其中X/Y的shape为[N, C, H, W], S/B的shape为[1, C, 1, ,1]。

:输入:
    - input: 输入tensor
    - scale: 保存input的放大倍数
    - bias: 放大后加上的bias

:输出:
    - output: 结果tensor

:属性:
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:接口:
    无

:范例:
    .. code-block:: shell

      %3 = "top.Scale"(%0, %1, %2) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x27x27xf32> loc("Scale")


SigmoidOp
^^^^^^^^^^^^^^^
:简述:
    激活函数, 将tensor中元素映射到特定区间, 默认映射到[0, 1], 计算方法为:

    .. math::
        Y = \frac{scale}{1 + e^{-X}} + bias

:输入:
    - inputs: tensor数组, 任意类型的tensor


:属性:
    - scale: 倍数, 默认是1
    - bias: 偏置, 默认是0


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: shell

      %2 = "top.Sigmoid"(%1) {bias = 0.000000e+00 : f64, scale = 1.000000e+00 : f64} : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Sigmoid")



SiLUOp
^^^^^^^^^^^^^^^
:简述:
    激活函数, :math:`Y = \frac{X}{1 + e^{-X}}` 或 :math:`Y = X * Sigmoid(X)`

:输入:
    - input: tensor数组, 任意类型的tensor


:属性:
    无


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: shell

        %1 = "top.SiLU"(%0) : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Mul")



SliceOp
^^^^^^^^^^^^^^^
:简述: tensor切片, 将输入的tensor的各个维度, 根据offset和steps数组中的偏移和步长进行切片, 生成新的tensor


:输入:
    - input: tensor数组, 任意类型的tensor


:属性:
    - offset: 存储切片偏移的数组, offset数组的索引和输入tensor的维度索引对应
    - steps: 存储切片步长的数组, steps数组的索引和输入tensor维度索引对应


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: shell

        %1 = "top.Slice"(%0) {offset = [2, 10, 10, 12], steps = [1, 2, 2, 3]} : (tensor<5x116x64x64xf32>) -> tensor<3x16x16x8xf32> loc("output_Slice")




SoftmaxOp
^^^^^^^^^^^^^^^
:简述:
    对输入tensor, 在指定axis的维度上计算归一化指数值, 计算的方法如下:

    .. math::
        \sigma(Z)_i = \frac{e^{\beta{Z_i}}}{\sum_{j=0}^{K-1}{e^{\beta{Z_j}}}}

    其中,  :math:`\sum_{j=0}^{K-1}{e^{\beta{Z_j}}}` , 在axis维度上做指数值求和, j从0到K-1, K是输入tensor在axis维度上的尺寸。

    例如: 输入tensor的尺寸为 :math:`(N, C, W, H)`,在axis=1的通道上计算Softmax, 计算方法为:

    .. math::
        Y_{n,i,w,h} = \frac{e^{\beta{X_{n,i,w,h}}}}{\sum_{j=0}^{C-1}{e^{\beta{X_{n,j,w,h}}}}}
:输入:
    - input: tensor数组, 任意类型的tensor


:属性:
    - axis: 维度索引, 用于指定对输入tensor执行Softmax对应的维度, axis可以取值[-r,  r-1], r 为输入tensor维度的数量, 当axis为负数时, 表示倒序维度
    - beta: tflite模型中对输入的缩放系数, 非tflite模型无效, 默认值为1.0


:输出:
    - output: 输出tensor, 在指定维度做归一化指数值后的tensor

:接口:
    无

:范例:
    .. code-block:: shell

      %1 = "top.Softmax"(%0) {axis = 1 : i64} : (tensor<1x1000x1x1xf32>) -> tensor<1x1000x1x1xf32> loc("output_Softmax")


SqueezeOp
^^^^^^^^^^^^^^^
:简述:
    对输入tensor进行指定维度的裁剪并返回裁剪后的tensor
:输入:
    - input: tensor

:输出:
    - output: tensor

:属性:
    - axes: 指定需要裁剪的维度, 0代表第一个维度, -1代表最后一个维度

:接口:
    无

:范例:
    .. code-block:: shell

      %133 = "top.Squeeze"(%132) {axes = [-1]} : (tensor<1x255x20x20xf32) -> tensor<1x255x20xf32> loc(#loc278)

UpsampleOp
^^^^^^^^^^^^^^^

:简述:
    上采样op, 将输入tensor进行nearest上采样并返回tensor

:输入:
    tensor

:属性:
    - scale_h: 目标图像与原图像的高度之比
    - scale_w: 目标图像与原图像的宽度之比
    - do_relu: 结果是否做Relu, 默认为False
    - relu_limit: 如果做Relu, 指定上限值, 如果是负数, 则认为没有上限

:输出:
    - output: tensor

:接口:
    无

:范例:
    .. code-block:: shell

      %179 = "top.Upsample"(%178) {scale_h = 2 : i64, scale_w = 2 : i64} : (tensor<1x128x40x40xf32>) -> tensor<1x128x80x80xf32> loc("268_Resize")

WeightOp
^^^^^^^^^^^^^^^

:简述:
    权重op, 包括权重的读取和创建, 权重会存到npz文件中。权重的location与npz中的tensor名称是对应关系。

:输入:
    无

:属性:
    无

:输出:
    - output: 权重Tensor

:接口:
    - read: 读取权重数据, 类型由模型指定
    - read_as_float: 将权重数据转换成float类型读取
    - read_as_byte: 将权重数据按字节类型读取
    - create: 创建权重op
    - clone_bf16: 将当前权重转换成bf16, 并创建权重Op
    - clone_f16: 将当前权重转换成f16, 并创建权重Op

:范例:
    .. code-block:: shell

      %1 = "top.Weight"() : () -> tensor<32x16x3x3xf32> loc("filter")


