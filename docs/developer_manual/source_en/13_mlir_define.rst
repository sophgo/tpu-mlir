MLIR Definition
===============

This chapter introduces the definition of each element of MLIR, including Dialect, Interface, etc.

Top Dialect
---------------

Operations
~~~~~~~~~~~~~~~

AddOp
^^^^^^^^^^^^^^^

:Brief intro:
    Add operation, :math:`Y = coeff_0 * X_0 + coeff_1 * X_1`

:Input:
    - inputs: tensor array, corresponding to 2 or more input tensors

:Output:
    - output: tensor

:Attributes:
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number
    - coeff: the coefficient corresponding to each tensor, 1.0 by default

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Add"(%0, %1) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("add")


AvgPoolOp
^^^^^^^^^^^^^^^

:Brief intro:
    Perform average pooling on the input tensor, :math:`S=\frac{1}{width\ *\ height}\sum_{i,j}a_{ij}`, where :math:`width` and :math:`height` represent the width and height of the kernel_shape. :math:`\sum_{i,j}a_{ij}` means to sum the kernel_shape. A sliding window of a given size will sequentially pool the input tensor

:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - kernel_shape: controls the size of the sliding window
    - strides: step size, controlling each step of the sliding window
    - pads: controls the shape of the padding
    - pad_value: padding content, constant, 0 by default
    - count_include_pad: whether the result needs to count the pads filled
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %90 = "top.AvgPool"(%89) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool1_fwd_GlobalAveragePool")

Depth2SpaceOp
^^^^^^^^^^^^^^^

:Brief intro:
    Depth to space operation, :math:`Y = Depth2Space(X)`

:Input:
    - inputs: tensor

:Output:
    - output: tensor

:Attributes:
    - block_h: tensor block size of h dimension, i64 type
    - block_w: tensor block size of w dimension, i64 type
    - is_CRD: column-row-depth. If true, the data is arranged in the depth direction according to the order of HWC, otherwise it is CHW, bool type
    - is_inversed: if true, the shape of the result is: :math:`[n, c * block_h * block_w, h / block_h, w / block_w]`, otherwise it is: :math:`[n, c / (block_h * block_w), h * block_h, w * block_w]`, bool type

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Depth2Space"(%0) {block_h = 2, block_w = 2, is_CRD = true, is_inversed = false} : (tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32> loc("add")


BatchNormOp
^^^^^^^^^^^^^^^

:Brief intro:
    Perform Batch Normalization on a 4D input tensor. More details on batch normalization can be found in the paper "`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    The specific calculation formula is as follows:

    .. math::

      y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

:Input:
    - input: 4D input tensor
    - mean: mean of the input tensor
    - variance: variance of the input tensor
    - gamma: :math:`\gamma` tensor in the formula, can be None
    - beta: :math:`\beta` tensor in the formula, can be None

:Output:
    - output: tensor

:Attributes:
    - epsilon: constant :math:`\epsilon` in formula, 1e-05 by default
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %5 = "top.BatchNorm"(%0, %1, %2, %3, %4) {epsilon = 1e-05, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x27x27xf32> loc("BatchNorm")

CastOp
^^^^^^^^^^^^^^^
(To be implemented)

ClipOp
^^^^^^^^^^^^^^^
:Brief intro:
      Constrain the given input to a certain range

:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - min: the lower limit
    - max: the upper limit

:Output:
    - output: tensor
:Interface:
    None

:Example:
    .. code-block:: shell

      %3 = "top.Clip"(%0) {max = 1%: f64,min = 2%: f64} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Clip")

ConcatOp
^^^^^^^^^^^^^^^

:Brief intro:
    Concatenates the given sequence of tensors in the given dimension. All input tensors either have the same shape (except the dimension to be concatenated) or are all empty.

:Input:
    - inputs: tensor array, corresponding to 2 or more input tensors

:Output:
    - output: tensor

:Attributes:
    - axis: the subscript of the dimension to be concatenated
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Concat"(%0, %1) {axis = 1, do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>)  -> tensor<1x6x27x27xf32> loc("Concat")

ConvOp
^^^^^^^^^^^^^^^

:Brief intro:
    Perform 2D convolution operation on the input tensor.

    In simple terms, the size of the given input is :math:`(N, C_{\text{in}}, H, W)`. The output :math:`(N, C_{\text{out}}, H_{ \text{out}}, W_{\text{out}})` is calculated as:

    .. math::

      \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k),

    where :math:`\star` is a valid cross-correlation operation, :math:`N` is the batch size, :math:`C` is the number of channels, :math:`H, W` is the input image height and width.

:Input:
    - input: tensor
    - filter: parameter tensor. The shape is

    :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`

    - bias: learnable bias tensor with the shape of :math:`(out\_channels)`

:Output:
    - output: tensor

:Attributes:
    - kernel_shape: the size of the convolution kernel
    - strides: strides of convolution
    - pads: the number of layers to add 0 to each side of the input
    - group: the number of blocked connections from the input channel to the output channel, the default is 1
    - dilations: the spacing between convolution kernel elements, optional
    - inserts: optional
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Conv"(%0, %1) {kernel_shape = [3, 5], strides = [2, 1], pads = [4, 2]} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Conv")

DeconvOp
^^^^^^^^^^^^^^^

:Brief intro:

    Perform a deconvolution operation on the input tensor.

:Input:
    - input: tensor
    - filter: parameter tensor. The shape is

    :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size[0]}, \text{kernel\_size[1]})`

    - bias: learnable bias tensor with the shape of :math:`(out\_channels)`

:Output:
    - output: tensor

:Attributes:
    - kernel_shape: the size of the convolution kernel
    - strides: strides of convolution
    - pads: the number of layers to add 0 to each side of the input
    - group: the number of blocked connections from the input channel to the output channel, the default is 1
    - dilations: the spacing between convolution kernel elements, optional
    - inserts: optional
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Deconv"(%0, %1) {kernel_shape = (3, 5), strides = (2, 1), pads = (4, 2)} : (tensor<20x16x50x100xf32>, tensor<33x3x5xf32>)  -> tensor<20x33x28x49xf32> loc("Deconv")


DivOp
^^^^^^^^^^^^^^^

:Brief intro:
    Division operation, :math:`Y = X_0 / X_1`

:Input:
    - inputs: tensor array, corresponding to 2 or more input tensors

:Output:
    - output: tensor

:Attributes:
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number
    - multiplier: the multiplier for quantization, the default is 1
    - rshift: right shift for quantization, 0 by default

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Div"(%0, %1) {do_relu = false, relu_limit = -1.0, multiplier = 1, rshift = 0} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("div")


InputOp
^^^^^^^^^^^^^^^
(To be implemented)

LeakyReluOp
^^^^^^^^^^^^^^^
:Brief intro:
    Apply the LeakyRelu function on each element in the tensor. The function can be expressed as: f(x) = alpha * x for x < 0, f(x) = x for x >= 0
:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - alpha: the coefficients corresponding to each tensor

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %4 = "top.LeakyRelu"(%3) {alpha = 0.67000001668930054 : f64} : (tensor<1x32x100x100xf32>) -> tensor<1x32x100x100xf32> loc("LeakyRelu")


LSTMOp
^^^^^^^^^^^^^^^
:Brief intro:
    Perform the LSTM operation of the RNN

:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - filter: convolution kernel
    - recurrence: recurrence unit
    - bias: parameter of LSTM
    - initial_h: Each sentence in LSTM will get a state after the current cell. The state is a tuple(c, h), where h=[batch_size, hidden_size]
    - initial_c: c=[batch_size, hidden_size]
    - have_bias: whether to set bias, the default is false
    - bidirectional: set the LSTM of the bidirectional loop, the default is false
    - batch_first: whether to put the batch in the first dimension, the default is false

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

     %6 = "top.LSTM"(%0, %1, %2, %3, %4, %5) {batch_first = false, bidirectional = true, have_bias = true} : (tensor<75x2x128xf32>,tensor<2x256x128xf32>, tensor<2x256x64xf32>, tensor<2x512xf32>, tensor<2x2x64xf32>, tensor<2x2x64xf32>) -> tensor<75x2x2x64xf32> loc("LSTM")

LogOp
^^^^^^^^^^^^^^^
:Brief intro:
    Perform element-wise logarithm on the given input tensor

:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    None

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

     %1 = "top.Log"(%0) : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Log")

MaxPoolOp
^^^^^^^^^^^^^^^
:Brief intro:
    Perform max pool on the given input tensor
:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - kernel_shape: controls the size of the sliding window
    - strides: step size, controlling each step of the sliding window
    - pads: controls the shape of the padding
    - pad_value: padding content, constant, 0 by default
    - count_include_pad: whether the result needs to count the pads filled
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %8 = "top.MaxPool"(%7) {do_relu = false, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32> loc("resnetv22_pool0_fwd_MaxPool")

MatMulOp
^^^^^^^^^^^^^^^

:Brief intro:
    2D matrix multiplication operation, :math:`C = A * B`

:Input:
    - input: tensor: matrix of size m*k
    - right: tensor: matrix of size k*n

:Output:
    - output: tensor: matrix of size m*n

:Attributes:
    - bias: the bias_scale will be calculated according to the bias during quantization (can be empty)
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.MatMul"(%0, %1) {do_relu = false, relu_limit = -1.0} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32> loc("matmul")


MulOp
^^^^^^^^^^^^^^^

:Brief intro:
    multiplication operation, :math:`Y = X_0 * X_1`

:Input:
    - inputs: tensor array, corresponding to 2 or more input tensors

:Output:
    - output: tensor

:Attributes:
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number
    - multiplier: the multiplier for quantization, the default is 1
    - rshift: right shift for quantization, default is 0

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Mul"(%0, %1) {do_relu = false, relu_limit = -1.0, multiplier = 1, rshift = 0} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("mul")


MulConstOp
^^^^^^^^^^^^^^^

:Brief intro:
    Multiply with a constant, :math:`Y = X * Const_Val`

:Input:
    - inputs: tensor

:Output:
    - output: tensor

:Attributes:
    - const_val: constants of type f64
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %1 = arith.constant 4.7 : f64
      %2 = "top.MulConst"(%0) {do_relu = false, relu_limit = -1.0} : (tensor<1x3x27x27xf64>, %1) -> tensor<1x3x27x27xf64> loc("mulconst")


PermuteOp
^^^^^^^^^^^^^^^
:Brief intro:
    Change the tensor layout. Change the order of tensor data dimensions, and rearrange the input tensor according to the given order

:Input:
    - inputs: tensor array, tensor of any types


:Attributes:
    - order: the order in which tensors are rearranged


:Output:
    - output: rearranged tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Permute"(%1) {order = [0, 1, 3, 4, 2]} : (tensor<4x3x85x20x20xf32>) -> tensor<4x3x20x20x85xf32> loc("output_Transpose")



ReluOp
^^^^^^^^^^^^^^^
:Brief intro:
    Performs the ReLU function on each element in the input tensor, if the limit is zero, the upper limit is not used
:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
   - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %1 = "top.Relu"(%0) {relu_limit = 6.000000e+00 : f64} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc("Clip")

ReshapeOp
^^^^^^^^^^^^^^^
:Brief intro:
    Reshape operator, which returns a tensor of the given shape with the same type and internal values as the input tensor. Reshape may operate on any row of the tensor. No data values will be modified during the reshaping process
:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    None

:Interface:
    None

:Example:
    .. code-block:: shell

      %133 = "top.Reshape"(%132) : (tensor<1x255x20x20xf32>) -> tensor<1x3x85x20x20xf32> loc("resnetv22_flatten0_reshape0_Reshape")

ScaleOp
^^^^^^^^^^^^^^^

:Brief intro:
    Scale operation :math:`Y = X * S + B`, where the shape of X/Y is [N, C, H, W], and the shape of S/B is [1, C, 1, , 1].

:Input:
    - input: tensor
    - scale: the magnification of the input
    - bias: the bias added after scaling

:Output:
    - output: tensor

:Attributes:
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Interface:
    None

:Example:
    .. code-block:: shell

      %3 = "top.Scale"(%0, %1, %2) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x27x27xf32> loc("Scale")


SigmoidOp
^^^^^^^^^^^^^^^
:Brief intro:
    The activation function, which maps elements in the tensor to a specific interval, [0, 1] by default. The calculation method is:

    .. math::
        Y = \frac{scale}{1 + e^{-X}} + bias

:Input:
    - inputs: tensor array, tensor of any types


:Attributes:
    - scale: the magnification of the input, 1 by default
    - bias: default is 0


:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %2 = "top.Sigmoid"(%1) {bias = 0.000000e+00 : f64, scale = 1.000000e+00 : f64} : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Sigmoid")



SiLUOp
^^^^^^^^^^^^^^^
:Brief intro:
    The activation function, :math:`Y = \frac{X}{1 + e^{-X}}` or :math:`Y = X * Sigmoid(X)`

:Input:
    - input: tensor array, tensor of any types


:Attributes:
    None


:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

        %1 = "top.SiLU"(%0) : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Mul")



SliceOp
^^^^^^^^^^^^^^^
:Brief intro: Tensor slice, slicing each dimension of the input tensor according to the offset and step size in the offset and steps arrays to generate a new tensor


:Input:
    - input: tensor array, tensor of any types


:Attributes:
    - offset: an array for storing slice offsets. The index of the offset array corresponds to the dimension index of the input tensor
    - steps: an array that stores the step size of the slice. The index of the steps array corresponds to the index of the input tensor dimension


:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

        %1 = "top.Slice"(%0) {offset = [2, 10, 10, 12], steps = [1, 2, 2, 3]} : (tensor<5x116x64x64xf32>) -> tensor<3x16x16x8xf32> loc("output_Slice")




SoftmaxOp
^^^^^^^^^^^^^^^
:Brief intro:
    For the input tensor, the normalized index value is calculated on the dimension of the specified axis. The calculation method is as follows:

    .. math::
        \sigma(Z)_i = \frac{e^{\beta{Z_i}}}{\sum_{j=0}^{K-1}{e^{\beta{Z_j}}}},

    where :math:`\sum_{j=0}^{K-1}{e^{\beta{Z_j}}}` does the exponential summation on the axis dimension. j ranges from 0 to K-1 and K is the size of the input tensor in the axis dimension.

    For example, the size of the input tensor is :math:`(N, C, W, H)`, and the Softmax is calculated on the channel of axis=1. The calculation method is:

    .. math::
        Y_{n,i,w,h} = \frac{e^{\beta{X_{n,i,w,h}}}}{\sum_{j=0}^{C-1}{e^{\beta{X_{n,j,w,h}}}}}
:Input:
    - input: tensor array, tensor of any types


:Attributes:
    - axis: dimension index, which is used to specify the dimension to perform softmax. It can take the value from [-r, r-1], where r is the number of dimensions of the input tensor. When axis is negative, it means the reverse order dimension
    - beta: The scaling factor for the input in the tflite model, invalid for non-tflite models, 1.0 by default.


:Output:
    - output: the tensor on which the softmax is performed.

:Interface:
    None

:Example:
    .. code-block:: shell

      %1 = "top.Softmax"(%0) {axis = 1 : i64} : (tensor<1x1000x1x1xf32>) -> tensor<1x1000x1x1xf32> loc("output_Softmax")


SqueezeOp
^^^^^^^^^^^^^^^
:Brief intro:
    Crop the input tensor with the specified dimension and return the cropped tensor
:Input:
    - input: tensor

:Output:
    - output: tensor

:Attributes:
    - axes: specifies the dimension to be cropped. 0 represents the first dimension and -1 represents the last dimension

:Interface:
    None

:Example:
    .. code-block:: shell

      %133 = "top.Squeeze"(%132) {axes = [-1]} : (tensor<1x255x20x20xf32) -> tensor<1x255x20xf32> loc(#loc278)

UpsampleOp
^^^^^^^^^^^^^^^

:Brief intro:
    Upsampling op, upsampling the input tensor nearest and returning the tensor

:Input:
    tensor

:Attributes:
    - scale_h: the ratio of the height of the target image to the original image
    - scale_w: the ratio of the width of the target image to the original image
    - do_relu: whether to perform Relu operation on the result, False by default
    - relu_limit: specify the upper limit value if doing Relu. There is no upper limit if it is a negative number

:Output:
    - output: tensor

:Interface:
    None

:Example:
    .. code-block:: shell

      %179 = "top.Upsample"(%178) {scale_h = 2 : i64, scale_w = 2 : i64} : (tensor<1x128x40x40xf32>) -> tensor<1x128x80x80xf32> loc("268_Resize")

WeightOp
^^^^^^^^^^^^^^^

:Brief intro:
    The weight op, including the reading and creation of weights. Weights will be stored in the npz file. The location of the weight corresponds to the tensor name in npz.

:Input:
    None

:Attributes:
    None

:Output:
    - output: weight Tensor

:Interface:
    - read: read weight data, the type is specified by the model
    - read_as_float: convert the weight data to float type for reading
    - read_as_byte: read the weight data in byte type
    - create: create weight op
    - clone_bf16: convert the current weight to bf16 and create a weight Op
    - clone_f16: convert the current weight to f16 and create a weight Op

:Example:
    .. code-block:: shell

      %1 = "top.Weight"() : () -> tensor<32x16x3x3xf32> loc("filter")


