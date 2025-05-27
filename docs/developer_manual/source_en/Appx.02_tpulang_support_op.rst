Appendix 02: Basic Elements of TpuLang
=============================================



This chapter will introduce the basic elements of TpuLang programs: Tensor, Scalar, Control Functions, and Operator.

.. _tensor:

Tensor
---------------

In TpuLang, the properties of a Tensor, including its name, data, data type, and tensor type, can only be declared or set at most once.

Generally, it is recommended to create a Tensor without specifying a name to avoid potential issues arising from identical names.
Only when it is necessary to specify a name should you provide one during the creation of the Tensor.

For Tensors that serve as the output of an Operator, you can choose not to specify the shape since the Operator will deduce it automatically.
Even if you do specify a shape, when the Tensor is the output of an Operator, the Operator itself will deduce and modify the shape accordingly.


The definition of Tensor in TpuLang is as follows:

   .. code-block:: python

      class Tensor:

         def __init__(self,
                     shape: list = [],
                     name: str = None,
                     ttype="neuron",
                     data=None,
                     dtype: str = "float32",
                     scale: Union[float, List[float]] = None,
                     zero_point: Union[int, List[int]] = None)
               #pass


As shown above, a Tensor in TpuLang has five parameters:

* shape: The shape of the Tensor, a List[int]. For Tensors that serve as the output of an Operator, the shape can be left unspecified with a default value of [].
* Name: The name of the Tensor, a string or None. It is recommended to use the default value None to avoid potential issues arising from identical names.
* ttype: The type of the Tensor, which can be "neuron," "coeff," or None. The initial value is "neuron."
* data: The input data for the Tensor.ndarray or None,the default value is None, the Tensor will be initialized with all zeros based on the specified shape.If ttype == "coeff", data **must** be provided (cannot be None).  If data is an ndarray, its shape and dtype must match the declared shape and dtype.
* dtype: The data type of the Tensor, with a default value of "float32." Other possible values include "float32," "float16," "int32," "uint32," "int16," "uint16," "int8," and "uint8."
* scale:  The quantization scale parameter of Tensor, float or List[float], default value is None;
* zero_point:  The quantization zero-point parameter, also known as the offset parameter of Tensor, int or List[int], default value is None;

Example of declaring a Tensor:

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      #weight
      weight = tpul.Tensor(dtype='float32', shape=[3,4], data=np.random.uniform(0,1,shape).astype('float32'), ttype="coeff")

Tensor Preprocessing (Tensor.preprocess)
---------------------------------------------

In TpuLang, if a Tensor is an input and requires preprocessing, you can call this function.

The definition of Tensor.preprocess in TpuLang is as follows:

   .. code-block:: python

      class Tensor:

         def preprocess(self,
                        mean : List[float] = [0, 0, 0],
                        scale : List[float] = [1.0, 1.0, 1.0],
                        pixel_format : str = 'bgr',
                        channel_format : str = 'nchw',
                        resize_dims : List[int] = None,
                        keep_aspect_ratio : bool = False,
                        keep_ratio_mode : str = 'letterbox',
                        pad_value : int = 0,
                        pad_type : str = 'center',
                        white_level : float = 4095,
                        black_level : float = 112):
               #pass

As shown above, Tensor.preprocess in TpuLang has the following parameters:

* mean: The average value of each channel of Tensor. Default = [0, 0, 0]
* scale: The scale value of each channel of the Tensor. Default = [1, 1, 1]
* pixel_format: The pixel format of Tensor. Default = 'bgr', Choices:'rgb', 'bgr', 'gray', 'rgba','gbrg', 'grbg', 'bggr', 'rggb'.
* channel_format: The data format of Tensor, i.e. whether channel is first or last. Default = 'nchw'.Choices: 'nchw', 'nhwc'.
* resize_dims: [h, w] of the Tensor after resizing. The default value is None, which means taking the h and w of the Tensor.
* keep_aspect_ratio: Parameter of resize operation that determines whether to maintain the same scaling ratio, bool, default = False
* keep_ratio_mode: Parameter of resize operation that specifies the mode when keep_aspect_ratio is enabled, default = 'letterbox'. Choices: 'letterbox', 'short_side_scale'.
* pad_value:Parameter of resize operation that sets the value when padding, int, default = 0.
* pad_type: The padding strategy when resizing, str, default = 'center'. Choices: 'normal', 'center'.
* white_level: The white-level parameter for raw image processing, str, default = 4095
* black_level: The black-level parameter for raw image processing, str, default = 112

Example of declaring Tensor.preprocess:

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      input.preprocess(mean=[123.675,116.28,103.53], scale=[0.017,0.017,0.017])
      # pass


.. _scalar:

Scalar
---------------

Define a scalar Scalar. A Scalar is a constant specified during declaration and cannot be modified afterward.

   .. code-block:: python

      class Scalar:

            def __init__(self, value, dtype=None):
                #pass

The Scalar constructor has two parameters:

* value: Variable type, i.e., int/float type, with no default value, and must be specified.
* dtype: The data type of the Scalar. If the default value None is used, it is equivalent to "float32."
 Otherwise, it can take values such as "float32," "float16," "int32," "uint32," "int16," "uint16," "int8," and "uint8."


Example of usage:

   .. code-block:: python

      pad_val = tpul.Scalar(1.0)
      pad = tpul.pad(input, value=pad_val)

Control Functions
--------------------

Control functions mainly involve controlling the initialization of TpuLang, starting the compilation process to generate target files, and other related operations.

Control functions are commonly used before and after the definition of Tensors and Operators in a TpuLang program.
For example, initialization might be necessary before writing Tensors and Operators,
and compilation and deinitialization might be performed after completing the definitions of Tensors and Operators.

.. _init:

Initialization Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Initialization Function is used before constructing a network in a program.

The interface for the initialization function is as follows, where you choose the processor:

    .. code-block:: python

      def init(device):
          #pass

* The device parameter is of type string and can take values from the range "BM1684X"\|"BM1688"\|"CV183X".

.. _compile:
compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interface definition
:::::::::::::::::::::::::::::::::::::::::::::::::

    .. code-block:: python

        def compile(name: str,
            inputs: List[Tensor],
            outputs: List[Tensor],
            cmp=True,
            refs=None,
            mode='f32',         # unused
            dynamic=False,
            asymmetric=False,
            no_save=False,
            opt=2,
            mlir_inference=True,
            bmodel_inference=True,
            log_level="normal",
            embed_debug_info=False):
            #pass


Description of the function
:::::::::::::::::::::::::::::::::::::::::::::::::

The function for comipling TpuLang model to bmodel.

Explanation of parameters
:::::::::::::::::::::::::::::::::::::::::::::::::

* name: A string. Model name.
* inputs: List of Tensors, representing all input Tensors for compiling the network.
* outputs: List of Tensors, representing all output Tensors for compiling the network.
* cmp: A boolean. True indicates result verification is needed, False indicates compilation only. 'cmp' parameter is useless when 'mlir_inference' set to False.
* refs: List of Tensors, representing all Tensors requiring verification in the compiled network.
* mode: A string. Indicates the type of model, supporting "f32" and "int8".
* dynamic: A boolean. Whether to do dynamic compilation.
* no_save: A boolean. It indicates whether to temporarily store intermediate files in shared memory and release them along with the process. When this option is enabled, the compile function will return the generated 'bmodel' file as a bytes-like object, which the user needs to receive and do some further process, for example, by saving it using 'f.write(bmodel_bin).'.
* asymmetric: A boolean. This parameter indicates whether it is for asymmetric quantization.
* opt: An integer type representing the compiler group optimization level. 0 indicates no need for layer group; 1 indicates grouping as much as possible; 2 indicates grouping based on dynamic programming.
* mlir_inference: A boolean. Whether to do mlir inference. 'cmp' parameter is useless when 'mlir_inference' set to False.
* bmodel_inference: A boolean. Whether to do bmodel inference.
* log_level is used to control the log level. Currently it supports only-pass, only-layer-group, normal, and quiet:
    - simple: Mainly prints graph to optimize pattern matching.
    - only-layer-group: mainly prints layer group information.
    - normal: The logs compiled and generated by bmodel will be printed out.
    - quiet: print nothing
* embed_debug_info: A boolean. Whether to enable profile.

.. _deinit:


Deinitialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After constructing the network, it is necessary to perform deinitialization to conclude the process.
Only after deinitialization, the TPU executable target generated by the previously initiated compilation will be saved to the specified output directory.

    .. code-block:: python

       def deinit():
          #pass

.. _reset_default_graph:

Reset Default Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before constructing a network, it is necessary to reset the default graph. If the input graph is None, after resetting the default graph, the current graph will be an empty graph.
If a specific graph is provided, it will be set as the default graph.
If there is only one subgraph, explicitly calling reset_default_graph is optional because the init function will invoke this method automatically.

    .. code-block:: python

       def reset_default_graph(graph = None):
          #pass

.. _get_default_graph:

Get Current Default Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After building the network, if you need to obtain the default subgraph, call this function to retrieve the default graph.

    .. code-block:: python

       def get_default_graph():
          #pass

.. _reset_graph:

Reset Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To clear a graph and its stored Tensor information, call this function. If graph is None, it clears the information of the current default graph.

    .. code-block:: python

       def reset_graph(graph = None):
          #pass

Note: If the Tensors in the graph are still used by other graphs, do not call this function to clear the graph's information.

.. _RoundingMode:

Rounding Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rounding is the process of discarding extra digits beyond a certain point according to specific rules, yielding a shorter, unambiguous numerical representation.
Given x, the rounded result is y. The following rounding modes are available:

Half to Even
"""""""""""""""""""""""""""""""""

    Round to nearest; when the fractional part is 0.5, round to the nearest even number. Corresponds to :cpp:enumerator:`half_to_even`.

Half Away From Zero
"""""""""""""""""""""""""""""""""

    Round to nearest; positive values toward +∞, negative values toward -∞. Corresponds to :cpp:enumerator:`half_away_from_zero`. Formula:

    .. math:: \mathsf{y = \mathrm{sign}(x)\left\lfloor|x| + 0.5\right\rfloor = -\mathrm{sign}(x)\left\lceil-|x| - 0.5\right\rceil}

Towards Zero
"""""""""""""""""""""""""""""""""

    Unconditional truncation toward zero. Corresponds to :cpp:enumerator:`towards_zero`. Formula:

    .. math:: \mathsf{y = \mathrm{sign}(x)\left\lfloor|x|\right\rfloor = -\mathrm{sign}(x)\left\lceil-|x|\right\rceil} = {\begin{cases}\mathsf{\lfloor x\rfloor}&{\text{if}}\mathsf{\ \ x > 0,}\\ \mathsf{\lceil x\rceil}&{\text{otherwise}}.\end{cases}}

Down
"""""""""""""""""""""""""""""""""

    Round toward -∞. Corresponds to :cpp:enumerator:`down`. Formula:

    .. math:: \mathsf{y = \lfloor x\rfloor = -\lceil-x\rceil}

Up
"""""""""""""""""""""""""""""""""

    Round toward +∞. Corresponds to :cpp:enumerator:`up`. Formula:

    .. math:: \mathsf{y = \lceil x\rceil = -\lfloor-x\rfloor}

Half Up
"""""""""""""""""""""""""""""""""

    Round to nearest; when the fractional part is 0.5, round toward +∞. Corresponds to :cpp:enumerator:`half_up`. Formula:

    .. math:: \mathsf{y = \lceil x + 0.5\rceil = -\lfloor-x - 0.5\rfloor = \left\lceil\frac{\lfloor 2x\rfloor}{2}\right\rceil}

Half Down
"""""""""""""""""""""""""""""""""

    Round to nearest; when the fractional part is 0.5, round toward -∞. Corresponds to :cpp:enumerator:`half_down`. Formula:

    .. math:: \mathsf{y = \lfloor x - 0.5\rfloor = -\lceil-x + 0.5\rceil = \left\lfloor\frac{\lceil 2x\rceil}{2}\right\rfloor}

Examples
"""""""""""""""""""""""""""""""""

The table below shows the mapping from x to y under different rounding modes.

.. math::
    \begin{array}{|c|c|c|c|c|c|c|c|}
    \hline
    ~ & \textsf{Half to} & \textsf{Half Away} & \textsf{Towards} & \textsf{Down} & \textsf{ Up } & \textsf{Half Up} & \textsf{Half Down}\\
    ~ & \textsf{Even}    & \textsf{From Zero} & \textsf{Zero}    & ~           & ~         & ~              & ~               \\ \hline
    +1.8 & +2 & +2 & +1 & +1 & +2 & +2 & +2\\ \hline
    +1.5 & +2 & +2 & +1 & +1 & +2 & +2 & +1\\ \hline
    +1.2 & +1 & +1 & +1 & +1 & +2 & +1 & +1\\ \hline
    +0.8 & +1 & +1 &  0 &  0 & +1 & +1 & +1\\ \hline
    +0.5 &  0 & +1 &  0 &  0 & +1 & +1 &  0\\ \hline
    +0.2 &  0 &  0 &  0 &  0 & +1 &  0 &  0\\ \hline
    -0.2 &  0 &  0 &  0 & -1 &  0 &  0 &  0\\ \hline
    -0.5 &  0 & -1 &  0 & -1 &  0 &  0 & -1\\ \hline
    -0.8 & -1 & -1 &  0 & -1 &  0 & -1 & -1\\ \hline
    -1.2 & -1 & -1 & -1 & -2 & -1 & -1 & -1\\ \hline
    -1.5 & -2 & -2 & -1 & -2 & -1 & -1 & -2\\ \hline
    -1.8 & -2 & -2 & -1 & -2 & -1 & -2 & -2\\ \hline
    \end{array}

.. _rounding mode of right-shift:


.. _operator:

Operator
---------------


In order to optimize performance in TpuLang programming, operators are categorized into Local Operator, Limited Local Operator, and Global Operator.

* Local Operator: During compilation, local operators can be merged and optimized with other local operators, ensuring that the data between operations only exists in the local storage of the TPU.
* Limited Local Operator: Limited local operators can be merged and optimized with other local operators under certain conditions.
* Global Operator: Global operators cannot be merged and optimized with other operators. The input and output data of these operators need to be placed in the TPU's global storage.

Many of the following operations are element-wise operations, requiring input and output Tensors to have the same number of dimensions.

When an operation has two input Tensors, there are two categories based on whether shape broadcasting is supported or not.
Support for shape broadcasting means that the shape values of tensor_i0 (input 0) and tensor_i1 (input 1) for the same dimension can be different.
In this case, one of the tensor's shape values must be 1, and the data will be broadcasted to match the shape of the other tensor.
Not supporting shape broadcasting requires the shape values of tensor_i0 (input 0) and tensor_i1 (input 1) to be identical.






NN/Matrix Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conv
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv(input: Tensor,
               weight: Tensor,
               bias: Tensor = None,
               stride: List[int] = None,
               dilation: List[int] = None,
               pad: List[int] = None,
               group: int = 1,
               out_dtype: str = None,
               out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional convolution operation. You can refer to the definitions of 2D convolution in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""

* input: Tensor type, representing the input Tensor in 4D NCHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 4D NCHW format.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1], where oc represents the number of output channels.
* stride: List of integers, representing the stride size along each spatial axis. If None, it is [1, 1]. If not None, it requires a length of 2.
* dilation: List of integers, representing the dilation size along each spatial axis. If None, it is [1, 1]. If not None, it requires a length of 2.
* pad: List of integers, representing the padding size along each spatial axis, which follows the order of [x1_begin, x2_begin…x1_end, x2_end,…]. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 4.
* groups: An integer, representing the number of groups in the convolution layer.
* out_dtype: str or None. If None, the output tensor's data type matches the input's. Choices: "float32" or "float16".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.
* BM1684X: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.


conv_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv_int(input: Tensor,
                   weight: Tensor,
                   bias: Tensor = None,
                   stride: List[int] = None,
                   dilation: List[int] = None,
                   pad: List[int] = None,
                   group: int = 1,
                   input_zp: Union[int, List[int]] = None,
                   weight_zp: Union[int, List[int]] = None,
                   out_dtype: str = None,
                   out_name: str = None):
          # pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional convolution operation. You can refer to the definitions of 2D convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    output = (input - izp) Conv (weight - wzp) + bias[c];

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, the input tensor in 4-D NCHW format.
* weight: Tensor type, the convolution kernel in 4-D [oc, ic, kh, kw] format, where
    oc = number of output channels
    ic = number of input channels
    kh = kernel height
    kw = kernel width
* bias: Tensor type or None. If None, no bias is applied; otherwise shape must be [1, oc, 1, 1]. Data type is int32.
* stride: List[int] or None, the stride for each spatial dimension. Defaults to [1, 1] if None; if provided, length must be 2.
* dilation: List[int] or None, the dilation for each spatial dimension. Defaults to [1, 1] if None; if provided, length must be 2.
* pad: List[int] or None, the padding for each spatial dimension in [x1_begin, x2_begin, x1_end, x2_end] order. Defaults to [0, 0, 0, 0] if None; if provided, length must be 4.
* groups: int, number of convolution groups. If ic = oc = groups, performs depthwise convolution.
* input_zp: int or List[int] or None, the zero-point for input. Defaults to 0 if None; if a list is provided its length must equal ic. (List mode not supported currently.)
* weight_zp: int or List[int] or None, the zero-point for weight. Defaults to 0 if None; if a list is provided its length must equal ic (the number of input channels).
* out_dtype: string or None, the output tensor's data type. Defaults to int32 if None. Valid values: "int32", "uint32".
* out_name: string or None, the name of the output tensor. If None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8 or UINT8. The bias data type must be INT32.
* BM1684X: The input data type can be INT8 or UINT8. The bias data type must be INT32.



conv_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv_quant(input: Tensor,
                   weight: Tensor,
                   bias: Tensor = None,
                   stride: List[int] = None,
                   dilation: List[int] = None,
                   pad: List[int] = None,
                   group: int = 1,
                   input_scale: Union[float, List[float]] = None,
                   weight_scale: Union[float, List[float]] = None,
                   output_scale: Union[float, List[float]] = None,
                   input_zp: Union[int, List[int]] = None,
                   weight_zp: Union[int, List[int]] = None,
                   output_zp: Union[int, List[int]] = None,
                   out_dtype: str = None,
                   out_name: str = None):
          # pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional convolution operation. You can refer to the definitions of 2D convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    conv_i32 = (input - izp) Conv (weight - wzp) + bias[c];
    output = requant_int(conv_i32, mul, shift) + ozp

    mul,shift are obtained from iscale, wscale, oscale

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, the input tensor in 4-D NCHW format.
* weight: Tensor type, the convolution kernel in 4-D [oc, ic, kh, kw] format, where
    oc = number of output channels
    ic = number of input channels
    kh = kernel height
    kw = kernel width
* bias: Tensor type or None. If None, no bias is applied; otherwise shape must be [1, oc, 1, 1]. Data type is int32.
* stride: List[int] or None, the stride for each spatial dimension. Defaults to [1, 1] if None; if provided, length must be 2.
* dilation: List[int] or None, the dilation for each spatial dimension. Defaults to [1, 1] if None; if provided, length must be 2.
* pad: List[int] or None, the padding for each spatial dimension in [x1_begin, x2_begin, x1_end, x2_end] order. Defaults to [0, 0, 0, 0] if None; if provided, length must be 4.
* groups: int, number of convolution groups. If ic = oc = groups, performs depthwise convolution.
* input_scale: float or List[float] or None, the input quantization scale(s). Defaults to the tensor's existing scale if None; if a list is provided its length must equal ic. (List mode not supported.)
* weight_scale: float or List[float] or None, the kernel quantization scale(s). Defaults to the tensor's existing scale if None; if a list is provided its length must equal oc.
* output_scale: float or List[float], the output quantization scale(s). Must be provided; if a list is given its length must equal oc. (List mode not supported.)
* input_zp: int or List[int] or None, the input zero-point(s). Defaults to 0 if None; if a list is provided its length must equal ic. (List mode not supported.)
* weight_zp: int or List[int] or None, the kernel zero-point(s). Defaults to 0 if None; if a list is provided its length must equal oc.
* output_zp: int or List[int] or None, the output zero-point(s). Defaults to 0 if None; if a list is provided its length must equal oc. (List mode not supported.)
* out_dtype: string or None, the output tensor's data type. Defaults to int8 if None. Valid values: "int8", "uint8".
* out_name: string or None, the name of the output tensor. If None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8 or UINT8. The bias data type must be INT32.
* BM1684X: The input data type can be INT8 or UINT8. The bias data type must be INT32.

deconv
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def deconv(input: Tensor,
                 weight: Tensor,
                 bias: Tensor = None,
                 stride: List[int] = None,
                 dilation: List[int] = None,
                 pad: List[int] = None,
                 output_padding: List[int] = None,
                 group: int = 1,
                 out_dtype: str = None,
                 out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional deconvolution operation. You can refer to the definitions of 2D deconvolution in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input Tensor in 4D NCHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 4D NCHW format.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1], where oc represents the number of output channels.
* stride: List of integers, representing the stride size along each spatial axis. If None, it is [1, 1]. If not None, it requires a length of 2.
* dilation: List of integers, representing the dilation size along each spatial axis. If None, it is [1, 1]. If not None, it requires a length of 2.
* pad: List of integers, representing the padding size along each spatial axis. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 4.
* output_padding: List of integers, representing the output padding size along each spatial axis, which follows the order of [x1_begin, x2_begin…x1_end, x2_end,…]. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 4.
* group: An integer, representing the number of group in the deconvolution layer.
* out_dtype: str or None. If None, the output tensor's data type matches the input's. Choices: "float32" or "float16".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.
* BM1684X: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.

deconv_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def deconv_int(input: Tensor,
                   weight: Tensor,
                   bias: Tensor = None,
                   stride: List[int] = None,
                   dilation: List[int] = None,
                   pad: List[int] = None,
                   output_padding: List[int] = None,
                   group: int = 1,
                   input_zp: Union[int, List[int]] = None,
                   weight_zp: Union[int, List[int]] = None,
                   out_dtype: str = None,
                   out_name: str = None):
          # pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional convolution operation. You can refer to the definitions of 2D convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    output = (input - izp) Deconv (weight - wzp) + bias[c];

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor, the input tensor in 4-D NCHW format.
* weight: Tensor, the deconvolution (transpose convolution) kernel in 4-D `[oc, ic, kh, kw]` format, where
    oc = number of output channels
    ic = number of input channels
    kh = kernel height
    kw = kernel width
* bias: Tensor or None. If None, no bias is applied; otherwise its shape must be `[1, oc, 1, 1]`. Data type is `int32`.
* stride: List[int] or None, the stride for each spatial dimension. Defaults to `[1, 1]` if None; if provided, length must be 2.
* dilation: List[int] or None, the dilation for each spatial dimension. Defaults to `[1, 1]` if None; if provided, length must be 2.
* pad: List[int] or None, the padding for each spatial dimension in `[x1_begin, x2_begin, x1_end, x2_end]` order. Defaults to `[0, 0, 0, 0]` if None; if provided, length must be 4.
* output_padding: List[int] or None, the additional size added to the output shape. Defaults to `[0, 0]` if None; if provided, length must be 1 or 2.
* groups: int, the number of deconvolution groups.
* input_zp: int or List[int] or None, the zero-point for input quantization. Defaults to 0 if None; if a list is provided its length must equal `ic`. (List mode not supported currently.)
* weight_zp: int or List[int] or None, the zero-point for kernel quantization. Defaults to 0 if None; if a list is provided its length must equal `ic` (the number of input channels).
* out_dtype: string or None, the output tensor's data type. Defaults to `int32` if None. Valid values: `"int32"`, `"uint32"`.
* out_name: string or None, the name of the output tensor. If None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8 or UINT8. The bias data type must be INT32.
* BM1684X: The input data type can be INT8 or UINT8. The bias data type must be INT32.

conv3d
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv3d(input: Tensor,
                 weight: Tensor,
                 bias: Tensor = None,
                 stride: List[int] = None,
                 dilation: List[int] = None,
                 pad: List[int] = None,
                 group: int = 1,
                 out_dtype: str = None,
                 out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Three-dimensional convolution operation. You can refer to the definitions of 3D convolution in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input Tensor in 5D NCDHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 4D NCDHW format.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1, 1] or [oc], where oc represents the number of output channels.
* stride: List of integers, representing the stride size along each spatial axis. If None, it is [1, 1, 1]. If not None, it requires a length of 3.
* dilation: List of integers, representing the dilation size along each spatial axis. If None, it is [1, 1, 1]. If not None, it requires a length of 3.
* pad: List of integers, representing the padding size along each spatial axis, which follows the order of [x1_begin, x2_begin…x1_end, x2_end,…]. If None, it is [0, 0, 0, 0, 0, 0]. If not None, it requires a length of 6.
* groups: An integer, representing the number of groups in the convolution layer.
* out_dtype: string or None, the output tensor's data type. If None, inherits the input tensor's data type. Valid values: "float32", "float16".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.
* BM1684X: The input data type can be FLOAT32 or FLOAT16. The data types of input and weight must match. The bias data type must be FLOAT32.


conv3d_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv3d_int(input: Tensor,
                     weight: Tensor,
                     bias: Tensor = None,
                     stride: List[int] = None,
                     dilation: List[int] = None,
                     pad: List[int] = None,
                     group: int = 1,
                     input_zp: Union[int, List[int]] = None,
                     weight_zp: Union[int, List[int]] = None,
                     out_dtype: str = None,
                     out_name: str = None):


Description of the function
"""""""""""""""""""""""""""""""""
Fixed-point three-dimensional convolution operation. You can refer to the definitions of fixed-point 3D convolution in various frameworks.

::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) Conv3d (weight - kzp) + bias[c];

Conv3d represents 3D convolution computation.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input Tensor in 5D NCTHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 5D [oc, ic, kt, kh, kw] format. Here, oc represents the number of output channels, ic represents the number of input channels, kt is the kernel depth, kh is the kernel height, and kw is the kernel width.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1, 1].
* stride: List of integers, representing the stride size. If None, it is [1, 1, 1]. If not None, it requires a length of 3. The order in the list is [stride_t, stride_h, stride_w].
* dilation: List of integers, representing the dilation size. If None, it is [1, 1, 1]. If not None, it requires a length of 2. The order in the list is [dilation_t, dilation_h, dilation_w].
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0, 0, 0]. If not None, it requires a length of 6. The order in the list is [before, after, top, bottom, left, right].
* groups: An integer, representing the number of groups in the convolution layer. If ic=oc=groups, the convolution is depthwise conv3d.
* input_zp: List of integers or an integer, representing the input offset. If None, it is 0. If a list is provided, it should have a length of ic.
* weight_zp: List of integers or an integer, representing the kernel offset. If None, it is 0. If a list is provided, it should have a length of ic, where ic represents the number of input channels.
* out_dtype: A string or None, representing the data type of the input Tensor. If None, it is int32. Possible values: int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the data type determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
BM1688: The data type of input and weight can be INT8/UINT8. The data type of bias is INT32.
BM1684X: The data type of input and weight can be INT8/UINT8. The data type of bias is INT32.

conv3d_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv3d_quant(input: Tensor,
                   weight: Tensor,
                   bias: Tensor = None,
                   stride: List[int] = None,
                   dilation: List[int] = None,
                   pad: List[int] = None,
                   group: int = 1,
                   input_scale: Union[float, List[float]] = None,
                   weight_scale: Union[float, List[float]] = None,
                   output_scale: Union[float, List[float]] = None,
                   input_zp: Union[int, List[int]] = None,
                   weight_zp: Union[int, List[int]] = None,
                   output_zp: Union[int, List[int]] = None,
                   out_dtype: str = None,
                   out_name: str = None):
          # pass

Description of the function
"""""""""""""""""""""""""""""""""
Two-dimensional convolution operation. You can refer to the definitions of 2D convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    conv_i32 = (input - izp) Conv (weight - wzp) + bias[c];
    output = requant_int(conv_i32, mul, shift) + ozp
    mul,shift are obtained from iscale, wscale, oscale

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor, the input tensor in 5-D NCTHW format (N, C, T, H, W).
* weight: Tensor, the 3D convolution kernel in 5-D [oc, ic, kt, kh, kw] format, where
  - oc = number of output channels
  - ic = number of input channels
  - kt = kernel temporal depth
  - kh = kernel height
  - kw = kernel width
* bias: Tensor or None. If None, no bias is applied; otherwise its shape must be [1, oc, 1, 1, 1]. Data type is int32.
* stride: List[int] or None, the stride along each spatial/temporal dimension. Defaults to [1, 1, 1] if None; if provided, length must be 3.
* dilation: List[int] or None, the dilation along each spatial/temporal dimension. Defaults to [1, 1, 1] if None; if provided, length must be 3.
* pad: List[int] or None, the padding for each dimension in [t_begin, h_begin, w_begin, t_end, h_end, w_end] order. Defaults to [0, 0, 0, 0, 0, 0] if None; if provided, length must be 6.
* groups: int, the number of convolution groups. If ic == oc == groups, this is a depthwise 3D conv.
* input_scale: float, List[float], or None, the quantization scale(s) for the input. If None, uses the scale in tensor_i; if a list is provided, its length must be ic. (List mode not supported currently.)
* weight_scale: float, List[float], or None, the quantization scale(s) for the kernel. If None, uses the scale in weight; if a list is provided, its length must be oc.
* output_scale: float or List[float], the quantization scale(s) for the output. Cannot be None; if a list is provided, its length must be oc. (List mode not supported currently.)
* input_zp: int, List[int], or None, the zero-point(s) for the input. Defaults to 0 if None; if a list is provided, its length must be ic. (List mode not supported currently.)
* weight_zp: int, List[int], or None, the zero-point(s) for the kernel. Defaults to 0 if None; if a list is provided, its length must be oc.
* output_zp: int, List[int], or None, the zero-point(s) for the output. Defaults to 0 if None; if a list is provided, its length must be oc. (List mode not supported currently.)
* out_dtype: string or None, the output tensor's data type. If None, defaults to int8. Valid values: "int8", "uint8".
* out_name: string or None, the name of the output tensor. If None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the data type determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The data type of input and weight can be INT8/UINT8. The data type of bias is INT32.
* BM1684X: The data type of input and weight can be INT8/UINT8. The data type of bias is INT32.

matmul
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def matmul(input: Tensor,
                 right: Tensor,
                 bias: Tensor = None,
                 right_transpose: bool = False,
                 left_transpose: bool = False,
                 output_transpose: bool = False,
                 keep_dims: bool = True,
                 out_dtype: str = None,
                 out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Matrix multiplication operation. You can refer to the definitions of matrix multiplication in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor, the left operand of the matmul. Must have rank ≥ 2, with shape [..., m, k] where m and k are the last two dimensions.
* right: Tensor, the right operand of the matmul. Must have rank ≥ 2, with shape [..., k, n] where k and n are the last two dimensions.
* bias: Tensor or None. If None, no bias is applied; otherwise its shape must be [n].
* left_transpose: bool, default False. If True, transpose the last two dims of input before multiplication (i.e. swap m and k).
* right_transpose: bool, default False. If True, transpose the last two dims of right before multiplication (i.e. swap k and n).
* output_transpose: bool, default False. If True, transpose the last two dims of the result before returning (i.e. swap result's last two dims).
* keep_dims: bool, default True. If True, the output retains the same rank as the broadcasted inputs; if False, the output is squeezed to a 2-D matrix of shape [M, N].
* out_dtype: string or None. If None, inherits the data type of input. Valid values: "float32", "float16".
* out_name: string or None. The name of the output tensor. If None, a name is generated automatically.

Notes on shapes and broadcasting:
input and right must have the same rank.
If rank = 2, a simple matrix-matrix multiply is performed.
If rank > 2, a batched matmul is performed:
The inner dimensions must match: input.shape[-1] == right.shape[-2].
The batch dims (input.shape[:-2] and right.shape[:-2]) must be broadcastable to a common shape.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16. The input and right data types must be consistent. The bias data type must be FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16. The input and right data types must be consistent.

matmul_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def matmul_int(input: Tensor,
                     right: Tensor,
                     bias: Tensor = None,
                     right_transpose: bool = False,
                     left_transpose: bool = False,
                     output_transpose: bool = False,
                     keep_dims: bool = True,
                     input_zp: Union[int, List[int]] = None,
                     right_zp: Union[int, List[int]] = None,
                     out_dtype: str = None,
                     out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Matrix multiplication operation. You can refer to the definitions of matrix multiplication in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor, the left operand of the matmul. Must have rank ≥ 2, with shape [..., m, k] (i.e. its last two dims are [m, k]).
* right: Tensor, the right operand of the matmul. Must have rank ≥ 2, with shape [..., k, n] (i.e. its last two dims are [k, n]).
* bias: Tensor or None. If None, no bias is applied; otherwise its shape must be [n].
* left_transpose: bool, default False. If True, transpose the last two dims of input before multiplication (swap m and k).
* right_transpose: bool, default False. If True, transpose the last two dims of right before multiplication (swap k and n).
* output_transpose: bool, default False. If True, transpose the last two dims of the result before returning.
* keep_dims: bool, default True. If True, the output retains the same rank as the broadcasted inputs; if False, the output is squeezed to a 2-D matrix of shape [M, N].
* input_zp: int or List[int], the zero-point(s) for input. Defaults to 0 if None. (List mode not supported currently.)
* right_zp: int or List[int], the zero-point(s) for right. Defaults to 0 if None. (List mode not supported currently.)
* out_dtype: string or None. If None, defaults to int32. Valid values: "int32", "uint32".
* out_name: string or None. The name of the output tensor. If None, a name is generated automatically.

Notes on shapes and broadcasting:
input and right must have the same rank.
If rank = 2, a simple matrix-matrix multiply is performed.
If rank > 2, a batched matmul is performed:
The inner dimensions must match: input.shape[-1] == right.shape[-2].
The batch dims (input.shape[:-2] and right.shape[:-2]) must be broadcastable to a common shape.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8/UINT8. The bias data type is INT32.
* BM1684X: The input data type can be INT8/UINT8. The bias data type is INT32.

matmul_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def matmul_quant(input: Tensor,
                     right: Tensor,
                     bias: Tensor = None,
                     right_transpose: bool = False,
                     keep_dims: bool = True,
                     input_scale: Union[float, List[float]] = None,
                     right_scale: Union[float, List[float]] = None,
                     output_scale: Union[float, List[float]] = None,
                     input_zp: Union[int, List[int]] = None,
                     right_zp: Union[int, List[int]] = None,
                     output_zp: Union[int, List[int]] = None,
                     out_dtype: str = None,
                     out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Matrix multiplication operation. You can refer to the definitions of matrix multiplication in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input:Tensor type, representing the left operand; rank ≥ 2, with its last two dims shaped [m, k].
* right:Tensor type, representing the right operand; rank ≥ 2, with its last two dims shaped [k, n].
* bias:Tensor type, representing the bias tensor. If None, no bias is applied; otherwise its shape must be [n].
* right_transpose:bool type, default False. Specifies whether to transpose the right matrix before computation.
* keep_dims:bool type, default True. Specifies whether to retain the original number of dims; if False, the output shape is 2-D.
* input_scale:List[float] or float, representing the quantization scale for input. If None, uses the input tensor's own scale. List[float] not supported.
* right_scale:List[float] or float, representing the quantization scale for right. If None, uses the right tensor's own scale. List[float] not supported.
* output_scale:List[float] or float, representing the quantization scale for output. Cannot be None. List[float] not supported.
* input_zp:List[int] or int, representing the zero-point for input. If None, defaults to 0. List[int] not supported.
* right_zp:List[int] or int, representing the zero-point for right. If None, defaults to 0. List[int] not supported.
* output_zp:List[int] or int, representing the zero-point for output. If None, defaults to 0. List[int] not supported.
* out_dtype:string type or None, representing the output tensor's data type; if None, defaults to int8. Valid values: int8/uint8.
* out_name:string type or None, representing the output tensor's name; if None, an internal name is autogenerated.

The ranks of the left and right Tensors must match.
If the rank of the Tensors is 2, a matrix-matrix multiplication is performed.
If the rank of the Tensors is greater than 2, a batched matrix multiplication is performed. It requires input.shape[-1] == right.shape[-2], and input.shape[:-2] and right.shape[:-2] must satisfy broadcasting rules.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8/UINT8. The bias data type is INT32.
* BM1684X: The input data type can be INT8/UINT8. The bias data type is INT32.



Base Element-wise Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def add(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise addition operation between tensors. :math:`tensor\_o = tensor\_i0 + tensor\_i1`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* scale: List[float] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the scales of tensor_i0, tensor_i1, and the output.
* zero_point: List[int] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the zero-points of tensor_i0, tensor_i1, and the output.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include float32/float16/int8/uint8/int16/uint16/int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype or is consistent with the input data type (when one of the inputs is int8, the output defaults to int8 type). When the input is float32/float16, the output data type must be consistent with the input.


Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.


sub
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sub(tensor_i0: Union[Tensor, Scalar, int, float],
                tensor_i1: Union[Tensor, Scalar, int, float],
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_dtype: str = None,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise subtraction operation between tensors. :math:`tensor\_o = tensor\_i0 - tensor\_i1`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* scale: List[float] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the scales of tensor_i0, tensor_i1, and the output.
* zero_point: List[int] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the zero-points of tensor_i0, tensor_i1, and the output.
* out_dtype: A string type or None, representing the data type of the output tensor. If None, it is consistent with the input tensors' dtype. The optional parameters are float32/float16/int8/int16/int32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is float32/float16,
the output data type must be the same as the input. When the input is int8/uint8/int16/uint16/int32/uint32, the output data type is int8/int16/int32.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.


mul
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mul(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Element-wise multiplication operation between tensors. :math:`tensor\_o = tensor\_i0 * tensor\_i1`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* scale: List[float] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the scales of tensor_i0, tensor_i1, and the output.
* zero_point: List[int] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the zero-points of tensor_i0, tensor_i1, and the output.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include float32/float16/int8/uint8/int16/uint16/int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype or is consistent with the input data type (when one of the inputs is int8, the output defaults to int8 type). When the input is float32/float16, the output data type must be consistent with the input.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. When the data type is FLOAT16/FLOAT32, the data types of tensor_i0 and tensor_i1 must be consistent.


div
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def div(tensor_i0: Union[Tensor, Scalar],
            tensor_i1: Union[Tensor, Scalar],
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Element-wise division operation between tensors. :math:`tensor\_o = tensor\_i0 / tensor\_i1`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


max
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def max(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise maximum operation between tensors. :math:`tensor\_o = max(tensor\_i0, tensor\_i1)`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* scale: List[float] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the scales of tensor_i0, tensor_i1, and the output.
* zero_point: List[int] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the zero-points of tensor_i0, tensor_i1, and the output.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include float32/float16/int8/uint8/int16/uint16/int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is float32/float16,
the output data type must be the same as the input. When the input is int8/uint8/int16/uint16/int32/uint32, the output can be any integer type.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT8/UINT8.


min
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def min(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise minimum operation between tensors. :math:`tensor\_o = min(tensor\_i0, tensor\_i1)`.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* scale: List[float] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the scales of tensor_i0, tensor_i1, and the output.
* zero_point: List[int] type or None, representing the quantization parameters; if None, indicates non-quantized computation; if a List, its length must be 3, corresponding to the zero-points of tensor_i0, tensor_i1, and the output.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include float32/float16/int8/uint8/int16/uint16/int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type.
When the input is float32/float16, the output data type must be the same as the input. When the input is int8/uint8/int16/uint16/int32/uint32, the output can be any integer type.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.

add_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def add_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Operation Formula :math:`tensor\_o = (tensor\_i0 - tensor\_i1) << shift`.
After adding tensor_i0 and tensor_i1 element-wise, a rounded arithmetic shift by shift bits is applied. A positive shift denotes a left shift; a negative shift denotes a right shift. The rounding mode is determined by round_mode.
The sum is first stored in INT64 as an intermediate result, then the rounded arithmetic shift is performed on the INT64 value.
The result supports saturation. If tensor_i0 and tensor_i1 are signed and tensor_o is unsigned, saturation is mandatory.
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor, Scalar, or int type, representing the left-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* tensor_i1: Tensor, Scalar, or int type, representing the right-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* shift: int type, specifying the number of bits to shift.
* round_mode: String type, specifying the rounding mode; default is 'half_away_from_zero'. Valid values are 'half_away_from_zero', 'half_to_even', 'towards_zero', 'down', and 'up'.
* is_saturate: bool type, indicating whether to apply saturation; default is True.
* out_dtype: String type or None, specifying the output Tensor data type; if None, defaults to the type of tensor_i0. Optional values are int8, uint8, int16, uint16, int32, and uint32.
* out_name: String type or None, specifying the name of the output Tensor; if None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor.
The data type of the Tensor is specified by out_dtype, or is consistent with the input data type.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.
* BM1684X: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.

sub_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sub_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Operation Formula :math:`tensor\_o = (tensor\_i0 - tensor\_i1) << shift`.
Element-wise subtraction between two tensors followed by a rounded arithmetic shift by shift bits. If shift > 0, performs a left shift; if shift < 0, performs a right shift. The rounding mode is determined by round_mode.
This operation supports broadcasting of input tensors.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor, Scalar, or int type, representing the left-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* tensor_i1: Tensor, Scalar, or int type, representing the right-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* shift: int type, specifying the number of bits to shift.
* round_mode: String type, specifying the rounding mode; default is 'half_away_from_zero'. Valid values are 'half_away_from_zero', 'half_to_even', 'towards_zero', 'down', and 'up'.
* is_saturate: bool type, indicating whether to apply saturation; default is True.
* out_dtype: String type or None, specifying the output Tensor's data type; if None, defaults to tensor_i0's type. Optional values are 'int8', 'int16', and 'int32'.
* out_name: String type or None, specifying the name of the output Tensor; if None, a name is generated automatically.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor.
The data type of the Tensor is specified by out_dtype, or is consistent with the input data type.


Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.
* BM1684X: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.

mul_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mul_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Operation Formula :math:`tensor\_o = (tensor\_i0 * tensor\_i1) << shift`
Subtract the tensors element-wise and then perform a rounded arithmetic shift for the result. When shift is positive, perform a left shift; when shift is negative, perform a right shift. The rounding mode is determined by round_mode.
After multiplying the data for mul_shift, save the intermediate result as INT64, and then perform a rounded arithmetic shift operation based on INT64.
The result supports saturation processing. When tensor_i0 and tensor_i1 are signed and tensor_o is unsigned, the result must be saturated.
This operation supports broadcasting of input tensors.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor, Scalar, or int type, representing the left-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* tensor_i1: Tensor, Scalar, or int type, representing the right-hand input operand. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* shift: int type, specifying the number of bits to shift.
* round_mode: String type, specifying the rounding mode; default is 'half_away_from_zero'. Valid values are 'half_away_from_zero', 'half_to_even', 'towards_zero', 'down', and 'up'.
* is_saturate: bool type, indicating whether to apply saturation; default is True.
* out_dtype: String type or None, specifying the output Tensor's data type; if None, defaults to tensor_i0's type. Optional values are 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: String type or None, specifying the name of the output Tensor; if None, a name is generated automatically.


Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor.
The data type of the Tensor is specified by out_dtype, or is consistent with the input data type.


Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.
* BM1684X: The input data type can be INT32/UINT32/INT16/UINT6/INT8/UINT8.

copy
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def copy(tensor_i, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The Copy function is applied to copy the input data into the output Tensor.
This operation belongs to **global operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


clamp
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""

    .. code-block:: python

      def clamp(tensor_i, min, max, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Clipping operation for all elements in the input tensor, restricting values to a specified minimum and maximum range.
Values greater than the maximum are truncated to the maximum, and values less than the minimum are truncated to the minimum.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor.
* min_value: Scalar type, representing the lower bound of the range.
* max_value: Scalar type, representing the upper bound of the range.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gt
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def gt(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise greater than comparison operation between tensors. :math:`tensor\_o = tensor\_i0 > tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

lt
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def lt(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise less than comparison operation between tensors. :math:`tensor\_o = tensor\_i0 < tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

ge
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""

    .. code-block:: python

      def ge(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""

Element-wise greater than or equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 >= tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

le
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def le(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""

Element-wise less than or equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 <= tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

eq
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""

    .. code-block:: python

      def eq(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""

Element-wise equality comparison operation between tensors. :math:`tensor\_o = tensor\_i0 == tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

ne
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ne(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise not equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 != tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three floats corresponding to the scales of tensor_i0, tensor_i1, and the output; the scales of tensor_i0 and tensor_i1 must be identical.
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of three integers corresponding to the zero_points of tensor_i0, tensor_i1, and the output; the zero_points of tensor_i0 and tensor_i1 must be identical.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data types of tensor_i0 and tensor_i1 must be consistent.

gts
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def gts(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise greater-than comparison operation between tensors and scalars. :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.


lts
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def lts(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise less-than comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.

ges
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ges(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise greater-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.

les
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def les(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise less-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.

eqs
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def eqs(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The element-wise equality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.

nes
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nes(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The element-wise inequality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8. The data type of scalar_i1 is FLOAT32.

Activation Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

relu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def relu(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ReLU activation function, implemented on an element-wise basis. :math:`y = max(0, x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

prelu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def prelu(input: Tensor, slope : Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
prelu activation function, implements function element by element :math:`y =\begin{cases}x\quad x>0\\x*slope \quad x<=0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input Tensor.
* slope: Tensor type, representing the slope Tensor. Only supports slope as a coeff Tensor.
* out_name: string type or None, representing the name of the output Tensor; if None, a name will be automatically generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.



leaky_relu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def leaky_relu(input: Tensor,
                    negative_slope: float = 0.01,
                    out_name: str = None,
                    round_mode : str="half_away_from_zero",):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The leaky ReLU activation function, implemented on an element-wise basis. :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor.
* negative_slope: float type, representing the negative slope for inputs < 0; default is 0.01.
* out_name: string type or None, the name of the output tensor; if None, a name is auto-generated internally.
* round_mode: string type, the rounding mode; default is “half_away_from_zero”. Valid values are “half_away_from_zero”, “half_to_even”, “towards_zero”, “down”, and “up.”

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

abs
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def abs(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The abs absolute value activation function, implemented on an element-wise basis. :math:`y = \left | x \right |`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

ln
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ln(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ln activation function, implemented on an element-wise basis. :math:`y = log(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor.
* scale: List[float] type or None, quantization parameter(s). If None, indicates non-quantized computation. If a list, length must be 2, specifying the scales for tensor_i0 and the output.
* zero_point: List[int] type or None, quantization parameter(s). If None, indicates non-quantized computation. If a list, length must be 2, specifying the zero points for tensor_i0 and the output.
* out_name: string type or None, the name of the output tensor; if None, a name will be automatically generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

ceil
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ceil(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ceil rounding up activation function, implemented on an element-wise basis. :math:`y = \left \lfloor x \right \rfloor`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor.
* scale: List[float] type or None, quantization parameter(s).
  • None indicates non-quantized computation.
  • If a list, it must have length 2, specifying [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, quantization parameter(s).
  • None indicates non-quantized computation.
  • If a list, it must have length 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: string type or None, the name of the output tensor; if None, a name is automatically generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.


floor
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def floor(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The floor rounding down activation function, implemented on an element-wise basis. :math:`y = \left \lceil x \right \rceil`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale:List[float] type or None, quantization parameters. None indicates non-quantized computation. If a list, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point:List[int] type or None, quantization parameters. None indicates non-quantized computation. If a list, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

round
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def round(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The round activation function, which rounds to the nearest integer using the round half up (four-way tie-breaking) method, implemented on an element-wise basis. :math:`y = round(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


sin
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sin(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sin sine activation function, implemented on an element-wise basis. :math:`y = sin(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale:List[float] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point:List[int] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.


cos
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cos(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The cos cosine activation function, implemented on an element-wise basis. :math:`y = cos(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale:List[float] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point:List[int] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.


exp
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def exp(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The exp exponential activation function, implemented on an element-wise basis. :math:`y = e^{x}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale:List[float] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point:List[int] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

tanh
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tanh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None,
            round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The tanh hyperbolic tangent activation function, implemented on an element-wise basis. :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale:List[float] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point:List[int] type or None, quantization parameters. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* round_mode:string type, rounding mode. Defaults to "half_away_from_zero". Allowed values: "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.

sigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sigmoid(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None,
                round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sigmoid activation function, implemented on an element-wise basis. :math:`y = 1 / (1 + e^{-x})`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor.
* scale: List[float] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: string type or None, name of the output tensor. If None, a name is auto-generated internally.
* round_mode: string type, rounding mode. Defaults to `"half_away_from_zero"`. Allowed values: `"half_away_from_zero"`, `"half_to_even"`, `"towards_zero"`, `"down"`, `"up"`.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.

log_sigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def log_sigmoid(input: Tensor,
                    scale: List[float]=None,
                    zero_point: List[int]=None,
                    out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The log_sigmoid activation function, implemented on an element-wise basis. :math:`y = log(1 / (1 + e^{-x}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor.
* scale: List[float] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: string type or None, name of the output tensor. If None, a name is auto-generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.


elu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def elu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ELU (Exponential Linear Unit) activation function, implemented on an element-wise basis. :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.

square
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def square(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The square function, implemented on an element-wise basis. :math:`y = \square{x}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

sqrt
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sqrt(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sqrt square root activation function, implemented on an element-wise basis. :math:`y = \sqrt{x}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.

rsqrt
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rsqrt(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The rsqrt square root  takes the deactivation function, implemented on an element-wise basis. :math:`y = 1 / (sqrt{x})`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.


silu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def silu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The silu activation function, implemented on an element-wise basis. :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

swish
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def swish(input: Tensor,
              beta: float,
              scale: List[float]=None,
              zero_point: List[int]=None,
              round_mode: str = "half_away_from_zero",
              out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The swish activation function, implemented on an element-wise basis. :math:`y = x * (1 / (1 + e^{-x * beta}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor.
* beta: Scalar or float type, representing the β value.
* scale: List[float] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [input_scale, output_scale].
* zero_point: List[int] type or None, quantization parameter. None indicates non-quantized computation. If a List, length must be 2, specifying [input_zero_point, output_zero_point].
* round_mode: string type, rounding mode. Defaults to `"half_away_from_zero"`. Allowed values: `"half_away_from_zero"`, `"half_to_even"`, `"towards_zero"`, `"down"`, `"up"`.
* out_name: string type or None, name of the output tensor. If None, a name is auto-generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.


erf
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def erf(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The erf activation function, for the corresponding elements x and y at the same positions in the input and output Tensors,
is implemented on an element-wise basis. :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8.FLOAT16 data is automatically converted to FLOAT32.

tan
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tan(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The tan tangent activation function, implemented on an element-wise basis. :math:`y = tan(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.


softmax
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def softmax(input: Tensor,
                axis: int,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The softmax activation function, which normalizes an input vector into a probability distribution consisting of probabilities proportional
to the exponentials of the input numbers. :math:`tensor\_o = exp(tensor\_i)/sum(exp(tensor\_i),axis)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* axis: An int type, representing the axis along which the operation is performed.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

softmax_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def softmax_int(input: Tensor,
                    axis: int,
                    scale: List[float],
                    zero_point: List[int] = None,
                    out_name: str = None,
                    round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Softmax fixed-point operation. Please refer to the softmax definition in each framework.

    ::

      for i in range(256)
        table[i] = exp(scale[0] * i)

      for n,h,w in N,H,W
        max_val = max(input[n,c,h,w] for c in C)
        sum_exp = sum(table[max_val - input[n,c,h,w]] for c in C)
        for c in C
          prob = table[max_val - input[n,c,h,w]] / sum_exp
          output[n,c,h,w] = saturate(int(round(prob * scale[1])) + zero_point[1]),    其中saturate饱和到output数据类型

Among them, "table" represents table lookup.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor.
* axis: int type, axis along which the operation is performed.
* scale: List[float] type, quantization scales for input and output. Must be of length 2, specifying [input_scale, output_scale].
* zero_point: List[int] type or None, quantization zero points for input and output. Must match the length of `scale`. If None, defaults to [0, 0].
* out_name: string type or None, name of the output tensor. If None, a name is auto-generated internally.
* round_mode: string type, rounding mode. Defaults to `"half_away_from_zero"`. Allowed values: `"half_away_from_zero"`, `"half_to_even"`, `"towards_zero"`, `"down"`, `"up"`.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be INT8/UINT8.
* BM1684X: The input data type can be INT8/UINT8.


mish
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mish(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The Mish activation function, implemented on an element-wise basis. :math:`y = x * tanh(ln(1 + e^{x}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.



hswish
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hswish(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The h-swish activation function, implemented on an element-wise basis. :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.



arccos
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arccos(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The arccosine (inverse cosine) activation function, implemented on an element-wise basis. :math:`y = arccos(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.


arctanh
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arctanh(input: Tensor, out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The arctanh (inverse hyperbolic tangent) activation function, implemented on an element-wise basis. :math:`y = arctanh(x)=\frac{1}{2}ln(\frac{1+x}{1-x})`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.


sinh
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sinh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sinh (hyperbolic sine) activation function, implemented on an element-wise basis. :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32.FLOAT16 data is automatically converted to FLOAT32.



cosh
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cosh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The cosh (hyperbolic cosine) activation function, implemented on an element-wise basis. :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32. FLOAT16 data is automatically converted to FLOAT32.


sign
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sign(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sign activation function, implemented on an element-wise basis. :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.


gelu
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def gelu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None,
            round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The GELU (Gaussian Error Linear Unit) activation function, implemented on an element-wise basis. :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* round_mode: string type, rounding mode. Defaults to `"half_away_from_zero"`.
  Allowed values: `"half_away_from_zero"`, `"half_to_even"`, `"towards_zero"`, `"down"`, `"up"`.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.


hsigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hsigmoid(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The hsigmoid (hard sigmoid) activation function, implemented on an element-wise basis. :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* scale: List[float] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two floats [tensor_i0_scale, output_scale].
* zero_point: List[int] type or None, specifying quantization parameters. A value of None indicates non-quantized computation. If provided, it must be a list of two integers [tensor_i0_zero_point, output_zero_point].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.
* BM1684X: The input data type can be FLOAT32/INT8/UINT8. FLOAT16 data is automatically converted to FLOAT32.

Data Arrange Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

permute
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def permute(input:tensor,
                  order:Union[List[int], Tuple[int]],
                  out_name:str=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Permute the dimensions of the input Tensor according to the permutation parameter.

For example: Given an input shape of (6, 7, 8, 9) and a permutation parameter `order` of (1, 3, 2, 0), the output shape will be (7, 9, 8, 6).
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, reprsenting input Tensor.
* order: List[int] or Tuple[int] type, reprsenting permutation order. The length of `order` should be the same as the dimensions of input tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

tile
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tile(tensor_i: Tensor,
               reps: Union[List[int], Tuple[int]],
               out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Repeat the data by copying it along the specified dimension(s).
This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* `reps`: A `List[int]` or `Tuple[int]` indicating the number of copies for each dimension. The length of `reps` must match the number of dimensions of the tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

broadcast
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def broadcast(input: Tensor,
                    reps: Union[List[int], Tuple[int]],
                    out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Repeat the data by copying it along the specified dimension(s).
This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor for the operation.
* `reps`: A `List[int]` or `Tuple[int]` indicating the number of copies for each dimension. The length of `reps` must match the number of dimensions of the tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.


concat
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def concat(inputs: List[Tensor],
               scales: Optional[Union[List[float],List[int]]] = None,
               zero_points: Optional[List[int]] = None,
               axis: int = 0,
               out_name: str = None,
               dtype="float32",
               round_mode: str="half_away_from_zero"):
        #pass

Description of the function
"""""""""""""""""""""""""""""""""
Concatenate multiple tensors along the specified axis.

This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* inputs: A `List[Tensor]` type, containing multiple tensors. All tensors must have the same data type and the same number of shape dimensions.
* scales: An optional Union[List[float], List[int]] type, containing multiple input scales and one output scale, where the last element is the scale for the output.
* zero_points: An optional List[int] type, containing multiple input zero points and one output zero point, with the last one being the zero point for the output.
* axis: An `int` type, indicating the axis along which the concatenation operation will be performed.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* dtype: A string type, defaulting to "float32".
* round_mode: String type, representing rounding type. default to "half_away_from_zero".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

split
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def split(input:tensor,
                axis:int=0,
                num:int=1,
                size:Union[List[int], Tuple[int]]=None,
                out_name:str=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Split the input tensor into multiple tensors along the specified axis. If `size` is not empty, the dimensions of the split tensors are determined by `size`.
 Otherwise, the tensor is split into `num` equal parts along the specified axis, assuming the tensor's size along that axis is divisible by `num`.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: A `Tensor` type, indicating the tensor that is to be split.
* axis: An `int` type, indicating the axis along which the tensor will be split.
* num: An `int` type, indicating the number of parts to split the tensor into.
* size: A `List[int]` or `Tuple[int]` type. When not splitting evenly, this specifies the size of each part. For even splitting, it can be set to empty.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a `List[Tensor]`, where each `Tensor` has the same data type as the input `Tensor`.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

pad
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def pad(input:tensor,
              method='constant',
              value:Union[Scalar, Variable, None]=None,
              padding:Union[List[int], Tuple[int], None]=None,
              out_name:str=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Padding the input tensor.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: A `Tensor` type, indicating the tensor that is to be padded.
* method: string type, representing the padding method. Optional values are "constant", "reflect","symmetric" or "edge".
* value: A `Scalar`, `Variable` type, or `None`, representing the value to be filled. The data type is consistent with that of the tensor.
* padding: A `List[int]`, `Tuple[int]`, or `None`. If `padding` is `None`, a zero-filled list of length `2 * len(tensor.shape)` is used. For example, the padding of a hw 2D Tensor is [h_top, w_left, h_bottom, w_right].
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

repeat
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

       def repeat(tensor_i:Tensor,
                 reps:Union[List[int], Tuple[int]],
                 out_name:str=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Duplicate data along a specified dimension. Functionally equivalent to `tile`.
This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* reps: A `List[int]` or `Tuple[int]` type, representing the number of replications for each dimension. The length of `reps` must be consistent with the number of dimensions of the tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.

extract
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def extract(input: Tensor,
                    start: Union[List[int], Tuple[int]] = None,
                    end: Union[List[int], Tuple[int]] = None,
                    stride: Union[List[int], Tuple[int]] = None,
                    out_name: str = None)

Description
"""""""""""""""""""""""""""""""""
Extract slice of input tensor.
This operation is considered a **restricted local operation**.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing input tensor.
* start: A list or tuple of int, or None, representing the start of slice. If set to None, `start` is filled all with 0.
* end: A list or tuple of int, or None, representing the end of slice. If set to None, `end` is given as shape of input.
* stride: A list or tuple of int, or None, representing the stride of slice. If set to None, `stride` is filled all with 1.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor, whose data type is same of that of `table`.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688:  Data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: Data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.


roll
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def roll(input:Tensor,
                shifts: Union[int, List[int], Tuple[int]],
                dims: Union[int, List[int], Tuple[int]]   = None,
                out_name:str=None):
          #pass

Description
"""""""""""""""""""""""""""""""""
Roll the tensor input along the given dimension(s). Elements that are shifted beyond the last position are re-introduced at the first position. If dims is None, the tensor will be flattened before rolling and then restored to the original shape.
This operation is considered a **restricted local operation**.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type. the input tensor.
* shifts: int, a list or tuple of int. the number of places by which the elements of the tensor are shifted. If shifts is a tuple.
* dims: int, a list or tuple of int or None. Axis along which to roll.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688:  Data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.
* BM1684X: Data type can be FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16.



Sort Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arg
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def arg(input: Tensor,
                method: str = "max",
                axis: int = 0,
                keep_dims: bool = True,
                out_name: str = None):
        #pass


Description of the function
"""""""""""""""""""""""""""""""""
Translate: For the input tensor, find the maximum or minimum values along the specified axis, output the corresponding indices, and set the dimension of that axis to 1.
This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the Tensor to be operated on.
* method: A string type, indicating the method of operation, options include 'max' and 'min'.
* axis: An integer, indicating the specified axis. Default to 0.
* keep_dims: A boolean, indicating whether to keep the specified axis after the operation. The default value is True, which means to keep it (in this case, the length of that axis is 1).
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns two Tensors, the first Tensor represents indices, of type int32; and the second Tensor represents values, the type of which will be the same as the type of the input.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

topk
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def topk(input: Tensor,
                 axis: int,
                 k: int,
                 out_name: str = None):

Description
"""""""""""""""""""""""""""""""""
Find top k numbers after sorted

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor.
* axis: Int type, representing axis used in sorting.
* k: Int type, representing the number of top values along axis.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns two Tensors: the first one represents the values, whose data type is the same as that of the input tensor while the second one represents the indices in input tensor after sorted along axis.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sort
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sort(input: Tensor,
                 axis: int = -1,
                 descending : bool = True,
                 out_name = None)

Description
"""""""""""""""""""""""""""""""""
Sort input tensor along axis then return the sorted tensor and correspending indices.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing input.
* axis: Int type, representing the axis used in sorting. (Recently, only support axis == -1)
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns two Tensors: data type of the first is the same of that of input, and data type of the second is INT32.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


argsort
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def argsort(input: Tensor,
                    axis: int = -1,
                    descending : bool = True,
                    out_name : str = None)

Description
"""""""""""""""""""""""""""""""""
Sort input tensor along axis then return the correspending indices of sorted tensor.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing input.
* axis: Int type, representing the axis used in sorting. (Recently, only support axis == -1)
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns one Tensor whose data type is INT32.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


sort_by_key (TODO)
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sort_by_key(input: Tensor,
                        key: Tensor,
                        axis: int = -1,
                        descending : bool = True,
                        out_name = None)

Description
"""""""""""""""""""""""""""""""""
Sort input tensor by key along axis then return the sorted tensor and correspending keys.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing input.
* key: Tensor type, representing key.
* axis: Int type, representing the axis used in sorting.
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns two Tensors: data type of the first is the same of that of input, and data type of the second is is the same of that of key.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


Shape About Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

squeeze
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def squeeze(tensor_i: Tensor, axis: Union[Tuple[int], List[int]], out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The operation reduces dimensions by removing axes with a size of 1 from the shape of the input. If no axes (axis) are specified, it removes all axes that have a size of 1.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* axis: A List[int] or Tuple[int] type, indicating the specified axes.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

reshape
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def reshape(tensor: Tensor, new_shape: Union[Tuple[int], List[int], Tensor], out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Translate: Perform a reshape operation on the input tensor.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A `Tensor` type, representing the tensor for the input operation.
* new_shape: A `List[int]`, `Tuple[int]`, or `Tensor` type, representing the shape after transformation.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

shape_fetch
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def shape_fetch(tensor_i: Tensor,
                begin_axis: int = None,
                end_axis: int = None,
                step: int = 1,
                out_name: str = None):
          #pass


Description of the function
"""""""""""""""""""""""""""""""""
To extract the shape information of an input tensor between specified axes (axis).
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* begin_axis: An int type, indicating the axis to start from.
* end_axis: An int type, indicating the axis to end at.
* step: An int type, indicating the step size.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a `Tensor` with the data type `INT32`.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

unsqueeze
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def unsqueeze(input: Tensor, axes: List[int] = [1,2], out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The operation adds dimensions by adding axes with a size of 1 from the shape of the input.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor for the operation.
* axis: A List[int] or Tuple[int] type, indicating the specified axes.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.


Quant Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

requant_fp_to_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def requant_fp_to_int(tensor_i,
                              scale,
                              offset,
                              requant_mode,
                              out_dtype,
                              out_name = None,
                              round_mode='half_away_from_zero'):

Description of the function
"""""""""""""""""""""""""""""""""
Quantizes the input tensor.

When `requant_mode` equals 0, the corresponding calculation for this operation is:

    ::

        output = saturate(int(round(input * scale)) + offset),
        Where `saturate` refers to saturation to the data type of the output.

    * For the BM1684X: The input data type can be `FLOAT32`, and the output data type can be `INT16`, `UINT16`, `INT8`, or `UINT8`.

When requant_mode equals 1, the corresponding calculation formula for this operation is:

    ::

        output = saturate(int(round(float(input) * scale + offset))),
        Where `saturate` refers to saturation to the data type of the output.

    * For the BM1684X: The input data type can be `INT32`, `INT16`, or `UINT16`, and the output data type can be `INT16`, `UINT16`, `INT8`, or `UINT8`.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor with 3 to 5 dimensions.
* scale: Either a List[float] or float type, representing the quantization coefficient.
* offset: When requant_mode == 0, either a List[int] or int type; when requant_mode == 1, either a List[float] or float type. Represents the output offset.
* requant_mode: An int type, representing the quantization mode.
* round_mode: A string type, representing the rounding mode. The default is "half_away_from_zero". The possible values for round_mode are "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* out_dtype: A string type, representing the data type of the output tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

requant_fp
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def requant_fp(tensor_i: Tensor,
               scale: Union[float, List[float]],
               offset: Union[float, List[float]],
               out_dtype: str,
               out_name: str=None,
               round_mode: str='half_away_from_zero',
               first_round_mode: str='half_away_from_zero'):

Description of the function
"""""""""""""""""""""""""""""""""
Quantizes the input tensor.

The calculation formula for this operation is:

    ::

        output = saturate(int(round(float(input) * scale + offset))),
        where saturate saturates to the output data type.


This operation is a **local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor, with 3-5 dimensions.
* scale: List[float] or float, representing the quantization scale.
* offset: List[int] or int, representing the output offset.
* out_dtype: String type, representing the data type of the input tensor. The data type can be "int16"/"uint16"/"int8"/"uint8".
* out_name: String type or None, representing the name of the output tensor. When set to None, the name will be automatically generated internally.
* round_mode: String type, representing the rounding mode. Default is "half_away_from_zero". The round_mode can take values of "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, representing the rounding mode used for quantizing tensor_i previously. Default is "half_away_from_zero". The first_round_mode can take values of "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return Value
"""""""""""""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: Support input datatype: INT32/INT16/UINT16.
* BM1684X: Support input datatype: INT32/INT16/UINT16.

requant_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def requant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                requant_mode: int,
                out_dtype: str="int8",
                out_name=None,
                round_mode='half_away_from_zero', rq_axis:int = 1, fuse_rq_to_matmul: bool = False):

          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Quantize the input tensor.

computation mode
"""""""""""""""""""""""""""""""""
When requant_mode == 0, the corresponding computation is:
output = shift > 0 ? (input << shift) : input
output = saturate((output * multiplier) >> 31),     where >> is round_half_up, saturate to INT32
output = shift < 0 ? (output >> -shift) : output,   where >> rounding mode is determined by round_mode
output = saturate(output + offset),                 where saturate to the output data type
BM1684X: Input data type can be INT32, output data type can be INT32/INT16/INT8
BM1688: Input data type can be INT32, output data type can be INT32/INT16/INT8

When requant_mode == 1, the corresponding computation is:
output = saturate((input * multiplier) >> 31),     where >> is round_half_up, saturate to INT32
output = saturate(output >> -shift + offset),      where >> rounding mode is determined by round_mode, saturate to the output data type
BM1684X: Input data type can be INT32, output data type can be INT32/INT16/INT8
BM1688: Input data type can be INT32, output data type can be INT32/INT16/INT8

When requant_mode == 2 (recommended), the corresponding computation is:
output = input * multiplier
output = shift > 0 ? (output << shift) : (output >> -shift),    where >> rounding mode is determined by round_mode
output = saturate(output + offset),                             where saturate to the output data type
BM1684X: Input data type can be INT32/INT16/UINT16, output data type can be INT16/UINT16/INT8/UINT8
BM1688: Input data type can be INT32/INT16/UINT16, output data type can be INT16/UINT16/INT8/UINT8

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor, 3-5 dimensions.
* mul: List[int] or int, representing the quantization multiplier coefficients.
* shift:List[int] or int, representing the quantization shift coefficients. Right shift is negative, left shift is positive.
* offset: List[int] or int, representing the output offset.
* requant_mode: int, representing the quantization mode.
* round_mode: string, representing the rounding mode. Default is "half_up".
* out_dtype: string or None, representing the output tensor type. None means the output data type is "int8".
* out_name: string or None, representing the output tensor name. If None, the name will be generated automatically.
* rq_axis: int, representing the axis on which to apply requant.
* fuse_rq_to_matmul: bool, indicating whether to fuse requant into matmul. Default is False.

Return value
"""""""""""""""""""""""""""""""""
Returns a tensor. The data type of this tensor is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X
* BM1688

dequant_int_to_fp
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
def dequant_int_to_fp(tensor_i: Tensor,
                  scale: Union[float, List[float]],
                  offset: Union[int, List[int], float, List[float]],
                  out_dtype: str="float32",
                  out_name: str=None,
                  round_mode: str='half_away_from_zero'):

Description of the function
"""""""""""""""""""""""""""""""""
Dequantizes the input tensor.

The calculation formula for this operation is:
    ::
        output = (input - offset) * scale

This operation is a **local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor with 3-5 dimensions.
* scale: List[float] or float, representing the quantization scale.
* offset: List[int] or int, representing the output offset.
* out_dtype: String type, representing the output tensor type. Default output data type is "float32". For input data types int8/uint8, the values can be "float16", "float32". For input types int16/uint16, the output type can only be "float32".
* out_name: String type or None, representing the name of the output tensor. If set to None, the name will be automatically generated internally.
* round_mode: String type, representing the rounding mode. Default is "half_away_from_zero". The round_mode can take values of "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is specified by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: Input data types can be INT16/UINT16/INT8/UINT8.


dequant_int
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
def dequant_int(tensor_i: Tensor,
        mul: Union[int, List[int]],
        shift: Union[int, List[int]],
        offset: Union[int, List[int]],
        lshift: int,
        requant_mode: int,
        out_dtype: str="int8",
        out_name=None,
        round_mode='half_up'):

Description of the function
"""""""""""""""""""""""""""""""""
Dequantizes the input tensor.

When requant_mode==0, the calculation formula for this operation is:

    ::
        output = (input - offset) * multiplier
        output = saturate(output >> -shift)

    *BM1684X*: Input data types can be INT16/UINT16/INT8/UINT8, output data types can be INT32/INT16/UINT16.

When requant_mode==1, the calculation formula for this operation is:

    ::
        output = ((input - offset) * multiplier) << lshift
        output = saturate(output >> 31)
        output = saturate(output >> -shift)


    *BM1684X*: Input data types can be INT16/UINT16/INT8/UINT8, output data types can be INT32/INT16/INT8.

This operation is a **local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor with 3-5 dimensions.
* mul: List[int] or int, representing the quantization multiplier.
* shift: List[int] or int, representing the quantization shift. Negative for right shift, positive for left shift.
* offset: List[int] or int, representing the output offset.
* lshift: int, representing the left shift coefficient.
* requant_mode: int, representing the quantization mode. Values can be 0 or 1, where 0 is "Normal" and 1 is "TFLite".
* round_mode: String type, representing the rounding mode. Default is "half_up", with options "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* out_dtype: String type, representing the input tensor type. Default is "int8".
* out_name: String type or None, representing the name of the output tensor. If set to None, the name will be automatically generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X


cast
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cast(tensor_i: Tensor,
         out_dtype: str = 'float32',
         out_name: str = None,
         round_mode: str = 'half_away_from_zero'):

Description of the function
"""""""""""""""""""""""""""""""""
Converts the input tensor `tensor_i` to the specified data type `out_dtype`, and rounds the data according to the specified rounding mode `round_mode`.
Note that this operator cannot be used alone and must be used in conjunction with other operators.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input Tensor.
* out_dtype: str = 'float32', the data type of the output tensor, default is `float32`.
* out_name: str = None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* round_mode: str = 'half_away_from_zero', the rounding mode, default is `half_away_from_zero`. Possible values are “half_away_from_zero”, “half_to_even”, “towards_zero”, “down”, “up”. Note that this function does not support the rounding modes “half_up” and “half_down”.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is determined by the input `out_dtype`.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.

Up/Down Scaling Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

maxpool2d
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def maxpool2d(input: Tensor,
                    kernel: Union[List[int],Tuple[int],None] = None,
                    stride: Union[List[int],Tuple[int],None] = None,
                    pad:    Union[List[int],Tuple[int],None] = None,
                    ceil_mode: bool = False,
                    scale: List[float] = None,
                    zero_point: List[int] = None,
                    out_name: str = None,
                    round_mode: str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Performs Max Pooling on the input Tensor.The Max Pooling 2d operation can refer to the maxpool2d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, indicating the input operation Tensor.
* kernel: List[int] or Tuple[int] type or None. If None is entered, global_pooling is used. If not None, the length of this parameter is required to be 2.
* stride: List[int] or Tuple[int] type or None, indicating the step size. If None is entered, the default value [1,1] is used. If not None, the length of this parameter is required to be 2.
* pad: List[int] or Tuple[int] type or None, indicating the padding size. If None is entered, the default value [0,0,0,0] is used. If not None, the length of this parameter is required to be 4.
* ceil: bool type, indicating whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the scales of input and output respectively.
* zero_point: List[int] type or None, offset parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the zero_points of input and output respectively.
* out_name: string type or None, indicating the name of the output Tensor. If it is None, the name will be automatically generated internally.
* round_mode: string type, indicates the rounding mode for the second time when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.


maxpool2d_with_mask
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def maxpool2d_with_mask(input: Tensor,
                              kernel: Union[List[int],Tuple[int],None] = None,
                              stride: Union[List[int],Tuple[int],None] = None,
                              pad:    Union[List[int],Tuple[int],None] = None,
                              ceil_mode: bool = False,
                              out_name: str = None,
                              mask_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Perform Max pooling on the input Tensor and output its mask index. Please refer to the pooling operations under various frameworks.
This operation belongs to **local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, indicating the input operation Tensor.
* kernel: List[int] or Tuple[int] type or None. If None is entered, global_pooling is used. If not None, the length of this parameter is required to be 2.
* pad: List[int] or Tuple[int] type or None. Indicates the padding size. If None is entered, the default value [0,0,0,0] is used. If not None, the length of this parameter is required to be 4.
* stride: List[int] or Tuple[int] type or None. Indicates the stride size. If None is entered, the default value [1,1] is used. If not None, the length of this parameter is required to be 2.
* ceil_mode: bool type, indicating whether to round up when calculating the output shape.
* out_name: string type or None. Indicates the name of the output Tensor. If None, the name is automatically generated internally.
* mask_name: string type or None. Indicates the name of the output Mask. If None, the name is automatically generated internally.

Return value
"""""""""""""""""""""""""""""""""
Returns two Tensors, one of which has the same data type as the input Tensor and the other returns a coordinate Tensor, which records the coordinates selected when using comparison operation pooling.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32
* BM1684X: The input data type can be FLOAT32


maxpool3d
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def maxpool3d(input: Tensor,
                kernel: Union[List[int],int,Tuple[int, ...]] = None,
                stride: Union[List[int],int,Tuple[int, ...]] = None,
                pad:    Union[List[int],int,Tuple[int, ...]] = None,
                ceil_mode: bool = False,
                scale: List[float] = None,
                zero_point: List[int] = None,
                out_name: str = None,
                round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Performs Max Pooling on the input Tensor.The Max Pooling 3d operation can refer to the maxpool3d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input tensor for the operation.
* kernel: List[int] or Tuple[int] or int or None, if None, global pooling is used. If not None and a single integer is provided, it indicates the same kernel size in three dimensions. If a List or Tuple is provided, its length must be 3.
* pad: List[int] or Tuple[int] or int or None, represents the padding size. If None, the default value [0,0,0,0,0,0] is used. If not None and a single integer is provided, it indicates the same padding size in three dimensions. If a List or Tuple is provided, its length must be 6.
* stride: List[int] or Tuple[int] or int or None, represents the stride size. If None, the default value [1,1,1] is used. If not None and a single integer is provided, it indicates the same stride size in three dimensions. If a List or Tuple is provided, its length must be 3.
* ceil_mode: bool type, indicates whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the scale for input and output respectively.
* zero_point: List[int] type or None, offset parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the zero point for input and output respectively.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.
* round_mode: string type, indicates the rounding mode for the second time when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8/UINT8.

avgpool2d
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def avgpool2d(input: Tensor,
                    kernel: Union[List[int],Tuple[int],None] = None,
                    stride: Union[List[int],Tuple[int],None] = None,
                    pad:    Union[List[int],Tuple[int],None] = None,
                    ceil_mode: bool = False,
                    scale: List[float] = None,
                    zero_point: List[int] = None,
                    out_name: str = None,
                    count_include_pad : bool = False,
                    round_mode : str="half_away_from_zero",
                    first_round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Performs Avg Pooling on the input Tensor.The Avg Pooling 2d operation can refer to the avgpool2d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, indicating the input operation Tensor.
* kernel: List[int] or Tuple[int] type or None. If None is entered, global_pooling is used. If not None, the length of this parameter is required to be 2.
* stride: List[int] or Tuple[int] type or None, indicating the step size. If None is entered, the default value [1,1] is used. If not None, the length of this parameter is required to be 2.
* pad: List[int] or Tuple[int] type or None, indicating the padding size. If None is entered, the default value [0,0,0,0] is used. If not None, the length of this parameter is required to be 4.
* ceil_mode: bool type, indicating whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the scales of input and output respectively.
* zero_point: List[int] type or None, offset parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the zero_points of input and output respectively.
* out_name: string type or None, indicating the name of the output Tensor. If it is None, the name will be automatically generated internally.
* count_include_pad: Bool type, indicating whether the pad value is included when calculating the average value. The default value is False.
* round_mode: String type, when the input and output Tensors are quantized, it indicates the second rounding mode. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, when the input and output Tensors are quantized, it indicates the first rounding mode. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.

avgpool3d
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def avgpool3d(input: Tensor,
            kernel: Union[List[int],int,Tuple[int, ...]] = None,
            stride: Union[List[int],int,Tuple[int, ...]] = None,
            pad:    Union[List[int],int,Tuple[int, ...]] = None,
            ceil_mode: bool = False,
            scale: List[float] = None,
            zero_point: List[int] = None,
            out_name: str = None,
            count_include_pad : bool = False,
            round_mode : str="half_away_from_zero",
            first_round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Performs Avg Pooling on the input Tensor.The Avg Pooling 3d operation can refer to the avgpool3d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: Tensor type, representing the input tensor for the operation.
* kernel: List[int] or Tuple[int] or int or None, if None, global pooling is used. If not None and a single integer is provided, it indicates the same kernel size in three dimensions. If a List or Tuple is provided, its length must be 3.
* pad: List[int] or Tuple[int] or int or None, represents the padding size. If None, the default value [0,0,0,0,0,0] is used. If not None and a single integer is provided, it indicates the same padding size in three dimensions. If a List or Tuple is provided, its length must be 6.
* stride: List[int] or Tuple[int] or int or None, represents the stride size. If None, the default value [1,1,1] is used. If not None and a single integer is provided, it indicates the same stride size in three dimensions. If a List or Tuple is provided, its length must be 3.
* ceil_mode: bool type, indicates whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the scale for input and output respectively.
* zero_point: List[int] type or None, offset parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the zero point for input and output respectively.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.
* count_include_pad: bool type, specifies whether to include padded elements in the average calculation. Defaults to False.
* round_mode: string type, indicates the rounding mode for the second time when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, indicating the rounding mode for the first round when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.


upsample
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def upsample(tensor_i: Tensor,
                   scale: int = 2,
                   out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The output is scaled repeatedly on the input tensor data in h and w dimensions.
This operation is considered a **local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* scale: int type, representing the expansion multiple.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8.

reduce
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def reduce(tensor_i: Tensor,
                 method: str = 'ReduceSum',
                 axis: Union[List[int],Tuple[int],int] = None,
                 keep_dims: bool = False,
                 out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Perform reduce operations on the input tensor according to axis_list.
This operation is considered a **restricted local operation**. This operation is considered a **local operation** only when the input data type is FLOAT32.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* method: string type, representing the reduce method.The method The can be "ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1","ReduceSum".
* axis: A List[int] or Tuple[int] type, indicating the specified axes.
* keep_dims: A boolean, indicating whether to keep the specified axis after the operation.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


Normalization Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

batch_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def batch_norm(input: Tensor,
                     mean: Tensor,
                     variance: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     out_name: str = None):
          #pass


Description of the function
"""""""""""""""""""""""""""""""""
The batch_norm op first completes batch normalization of the input values, and then scales and shifts them.
The batch normalization operation can refer to the batch_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""

* input: * input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* mean: A Tensor type, representing the mean value of the input, shape is [c].
* variance: A Tensor type, representing the variance value of the input, shape is [c].
* gamma: A Tensor type or None, representing the scaling after batch normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after batch normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

layer_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def layer_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int,
                     out_name: str = None):
          #pass


Description of the function
"""""""""""""""""""""""""""""""""
The layer_norm op first completes layer normalization of the input values, and then scales and shifts them.
The layer normalization operation can refer to the layer_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* gamma: A Tensor type or None, representing the scaling after layer normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after layer normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* axis: int type, the first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.


Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

group_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def group_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     num_groups: int,
                     out_name: str = None):
          #pass


Description of the function
"""""""""""""""""""""""""""""""""
The group_norm op first completes group normalization of the input values, and then scales and shifts them.
The group normalization operation can refer to the group_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* gamma: A Tensor type or None, representing the scaling after group normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after group normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* num_groups:int type, The number of groups of channels. It should be a divisor of the number of channels `C`.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


rms_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rms_norm(input: Tensor,
                     gamma: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int = -1,
                     out_name: str = None):
          #pass



Description of the function
"""""""""""""""""""""""""""""""""
The rms_norm op first completes RMS normalization of the input values, and then scales them.
The RMS normalization operation can refer to the RMSNorm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited.
* gamma: A Tensor type or None, representing the scaling after RMS normalization. If the value is not None, shape is required to be equal with the last dimension of the input. If None is used, shape is equivalent to all 1 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* axis: int type, the first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

normalize
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""
    .. code-block:: python

      def normalize(input: Tensor,
                        p: float = 2.0,
                        axes: Union[List[int], int] = 1,
                        eps : float = 1e-12,
                        out_name: str = None):

Description
"""""""""""""""""""""""""""""""""
Perfrom :math:`L_p` normalization over specified dimension of input tensor.
For a tensor input of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`,
each :math:`n_{dim}`-element vector :math:`v` along dimension :attr:`axes`  is transformed as:

.. math::
  v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

With the default arguments, it uses the Euclidean norm over vectors along dimension (1) for normalization.

This operation belongs to **local operations**.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input Tensor.The dimension of input is not limited. Support data type included: float32, float16.
* p: float type, representing the exponent vaue in the norm operation. Default to 2.0 .
* axes: Union[list[int], int] type, representing the dimension need to normalized. Default to 1. If axes is list, all the values in the list must be continuous. Caution: axes = [0, -1] is not continuous.
* eps: float type, the epsilon value to use to avoid division by zero. Default to 1e-12.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

Vision Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nms
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def nms(boxes: Tensor,
                scores: Tensor,
                format: str = 'PYTORCH',
                max_box_num_per_class: int = 1,
                out_name: str = None)

Description
"""""""""""""""""""""""""""""""""
Perform non-maximum-suppression upon input tensor.

Parameters
"""""""""""""""""""""""""""""""""
* boxes: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of box, the third dimension is 4 coordinates of boxes.
* scores: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of classes, the third dimension is number of boxes.
* format: String type, where 'TENSORFLOW' representing Tensorflow format [y1, x1, y2, x2] and 'PYTORCH'表示representing Pytorch format [x_center, y_center, width, height]. The default value is 'PYTORCH'.
* max_box_num_per_class: Int type, representing max number of boxes per class. It must be greater than 0. The default value is 1.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns one Tensor, which is the selected indices from the boxes tensor of 2 dimensions:[num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


interpolate
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def interpolate(input: Tensor,
                        scale_h: float,
                        scale_w: float,
                        method: str = 'nearest',
                        coord_mode: str = "pytorch_half_pixel",
                        out_name: str = None)

Description
"""""""""""""""""""""""""""""""""
Perform interpolation upon input tensor.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input Tensor. Must be at least a 2-dimensional tensor.
* scale_h: Float type, representing the resize scale along h-axis. Must be greater than 0.
* scale_w: Float type, representing the resize scale along w-axis. Must be greater than 0.
* method: String type, representing the interpolation method. Optional values are "nearest" or "linear". Default is "nearest".
* coord_mode: string type, representing the method used in inverse map of coordinates. Optional values are "align_corners", "pytorch_half_pixel", "half_pixel" or "asymmetric". Default is "pytorch_half_pixel".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Note that, parameter `coord_mode` defined here is the same as the parameter `coordinate_transformation_mode` defined in onnx operator `Resize`. Supposed that resize scale along h/w-axis is `scale`, input coordinate is `x_in`, input size is `l_in`, output coordinate is `x_out`, output size is `l_out`, then the defintion of inverse map of coordinates is as follows:
* `"half_pixel"`:

    ::

        x_in = (x_out + 0.5) / scale - 0.5

* `"pytorch_half_pixel"`:

    ::

        x_in = len > 1 ? (x_out + 0.5) / scale - 0.5 : 0

* `"align_corners"`:

    ::

        x_in = x_out * (l_in - 1) / (l_out - 1)

* `"asymmetric"`:

    ::

        x_in = x_out / scale


Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor representing the interpolated result. The data type is the same as the input type, and the shape is adjusted based on the scaling factors.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: Supports input data types FLOAT32/FLOAT16/INT8.
* BM1684X: Supports input data types FLOAT32/FLOAT16/INT8.



yuv2rgb
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def yuv2rgb(
            inputs: Tensor,
            src_format: int,
            dst_format: int,
            ImageOutFormatAttr: str,
            formula_mode: str,
            round_mode: str,
            out_name: str = None,
        ):


Description of the function
"""""""""""""""""""""""""""""""""
Transfer input tensor from yuv to rgb. Require tensor shape=[n,h*3/2,w], n represents `batch`, h represents `pixels height`, w represents `pixels width`.


Explanation of parameters
"""""""""""""""""""""""""""""""""
* inputs: Tensor type, representing the input yuv tensor。Its dims must be 3, 1st dim represents `batch`, 2nd dim represents `pixels height`, 3rd dim represents `pixels width`.
* src_format: Int type, representing the input format. `FORMAT_MAPPING_YUV420P_YU12`=0, `FORMAT_MAPPING_YUV420P_YV12`=1, `FORMAT_MAPPING_NV12`=2, `FORMAT_MAPPING_NV21`=3.
* dst_format: Int type, representing the output format. `FORMAT_MAPPING_RGB`=4, `FORMAT_MAPPING_BGR`=5.
* ImageOutFormatAttr: string type, representing the output dtype, currently only support `UINT8`.
* formula_mode: string type, representing the formula to transfer from yuv to rgb, currently support `_601_limited`, `_601_full`.
* round_mode: string type, currently support `HalfAwayFromZero`, `HalfToEven`.
* out_name: string type, representing the name of output tensor, default= `None`.

Return value
"""""""""""""""""""""""""""""""""
One rgb tensor will be output, with shape=[n,3,h,w], where n represents `batch`, h represents `pixels height`, w represents `pixels width`.


Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type must be UINT8/INT8. Output data type is UINT8.
* BM1688: The input data type must be UINT8/INT8. Output data type is UINT8.

roiExtractor
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def roiExtractor(rois: Tensor,
                         target_lvls: Tensor,
                         feats: List[Tensor],
                         PH: int,
                         PW: int,
                         sampling_ratio: int,
                         list_spatial_scale: Union[int, List[int], Tuple[int]],
                         mode:str=None,
                         out_name:str=None)


Description
"""""""""""""""""""""""""""""""""
Given 4 feature maps, extract the corresponding ROI from `rois` based on the `target_lvls` indices and perform ROI Align with the corresponding feature maps to obtain the final output. This operation is considered a **restricted local operation**.


Parameters
"""""""""""""""""""""""""""""""""
* rois: Tensor type, representing all the ROIs.
* target_lvls: Tensor type, representing which level of feature map each ROI corresponds to.
* feats: List[Tensor], representing all feature maps.
* PH: Int type, representing the height of the output.
* PW: Int type, representing the width of the output.
* sampling_ratio: Int type, representing the sample ratio for each level of the feature maps.
* list_spatial_scale: List[int] or int, representing the spatial scale corresponding to each feature map level.
        Please note that spatial scale follows mmdetection style, where one int value is initially given, and but its float reciprocal is adapted for roialign.
* mode: string type, representing the implementation forms, now supporting two modes: DynNormal, or DynFuse.
        Please note that in DynFuse mode, coordinates of rois can satisfy either mmdetection style, which is 5-length of [batch_id, x0, y0 x1, y1],
                                          or customized style, which is 7-length of [a, b, x0, y0, x1, y1, c], please customize the position of batch_id.
                         in DynNormal mode, a customized [a, b, x0, y0 x1, y1, c] coordinates style is adapted in case any customers desire to apply their models.
* out_name: string type, representing the name of output tensor, default= `None`.

Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as the input `rois`.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: Supports input data types FLOAT32/FLOAT16.
* BM1684X: Supports input data types FLOAT32/FLOAT16.



Select Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nonzero
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nonzero(tensor_i:Tensor,
                  dtype: str = 'int32',
                  out_name: str = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Extract the corresponding location information when input Tensor data is true.
This operation is considered a **global operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* dtype: String type. The data type of the output tensor, with a default value of "int32."
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with data type INT32.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


lut
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def lut(input: Tensor,
                table: Tensor,
                out_name: str = None):
        #pass

Description
"""""""""""""""""""""""""""""""""
Use look-up table to transform values of input tensor.

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input.
* table: Tensor type, representing the look-up table.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""""""""""""""""""""""""
Returns one Tensor, whose data type is the same as that of the `table` tensor.

Processor support
"""""""""""""""""""""""""""""""""
* BM1688:  The data type of `input` can be INT8/UINT8. The data type of `table` an be INT8/UINT8.
* BM1684X: The data type of `input` can be INT8/UINT8. The data type of `table` an be INT8/UINT8.

select
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def select(lhs: Tensor,
                   rhs: Tensor,
                   tbrn: Tensor,
                   fbrn: Tensor,
                   type: str,
                   out_name = None):
        #pass

Description
"""""""""""""""""""""""""""""""""
Select by the comparison result of `lhs` and `rhs`. If condition is True, select `tbrn`, otherwise select `fbrn`.

Parameters
"""""""""""""""""""""""""""""""""
* lhs: Tensor type, representing the left-hand-side.
* rhs: Tensor type, representing the right-hand-side.
* tbrn: Tensor type, representing the true branch.
* fbrn: Tensor type, representing the false branch.
* type: String type, representing the comparison operator. Optional values are "Greater"/"Less"/"GreaterOrEqual"/"LessOrEqual"/"Equal"/"NotEqual".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Constraint: The shape and data type of `lhs` and `rhs` should be the same. The shape and data type of `tbrn` and `fbrn` should be the same.

Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is the same that of `tbrn`.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688:  Data type of `lhs`/ `rhs`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16(TODO).
* BM1684X:  Data type of `lhs`/ `rhs`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16(TODO).

cond_select
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def cond_select(cond: Tensor,
                        tbrn: Union[Tensor, Scalar],
                        fbrn: Union[Tensor, Scalar],
                        out_name:str = None):
        #pass

Description
"""""""""""""""""""""""""""""""""
Select by condition representing by `cond`. If condition is True, select `tbrn`, otherwise select `fbrn`.

Parameters
"""""""""""""""""""""""""""""""""
* cond: Tensor type, representing condition.
* tbrn: Tensor type or Scalar type, representing true branch.
* fbrn: Tensor type or Scalar type, representing false branch.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Constraint: If `tbrn` and `fbrn` are all Tensors, then the shape and data type of `tbrn` and `fbrn` should be the same.

Returns
"""""""""""""""""""""""""""""""""
Returns a Tensor whose data type is the same that of `tbrn`.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688:  Data type of `cond`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X:  Data type of `cond`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.

bmodel_inference_combine
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

        def bmodel_inference_combine(
            bmodel_file: str,
            final_mlir_fn: str,
            input_data_fn: Union[str, dict],
            tensor_loc_file: str,
            reference_data_fn: str,
            dump_file: bool = True,
            save_path: str = "",
            out_fixed: bool = False,
            dump_cmd_info: bool = True,
            skip_check: bool = True,  # disable data_check to increase processing speed
            run_by_op: bool = False, # enable to run_by_op, may cause timeout error when some OPs contain too many atomic cmds
            desire_op: list = [], # set ["A","B","C"] to only dump tensor A/B/C, dump all tensor as defalt
            is_soc: bool = False,  # soc mode ONLY support {reference_data_fn=xxx.npz, dump_file=True}
            using_memory_opt: bool = False, # required when is_soc=True
            enable_soc_log: bool = False, # required when is_soc=True
            soc_tmp_path: str = "/tmp",  # required when is_soc=True
            hostname: str = None,  # required when is_soc=True
            port: int = None,  # required when is_soc=True
            username: str = None,  # required when is_soc=True
            password: str = None,  # required when is_soc=True
        ):

Description
"""""""""""""""""""""""""""""""""
Dump tensors layer by layer according to the bmodel, which help to verify the correctness of bmodel.

Parameters
"""""""""""""""""""""""""""""""""
* bmodel_file: String type, representing the abs path of bmodel.
* final_mlir_fn: String type, representing the abs path of final.mlir.
* input_data_fn: String type or Dict type, representing the input data, supporting Dict/.dat/.npz.
* tensor_loc_file: String type, representing the abs path of tensor_location.json.
* reference_data_fn: String type, representing the abs path of .mlir/.npz with `module.state = "TPU_LOWERED"`. Used to restore the shape during bmodel infer.
* dump_file: Bool type, representing whether save results as file.
* save_path: String type, representing the abs path of saving results on host.
* out_fixed: Bool type, representing whether to get results in fixed number.
* dump_cmd_info: Bool type, enable to save atomic cmd info at `save_path`.
* skip_check: Bool tyoe, set to True to disable data check to decrease time cost for CMODEL/PCIE mode.
* run_by_op: Bool type, enable to run_by_op, decrease time cost but may cause timeout error when some OPs contain too many atomic cmds.
* desire_op: List type, specify this option to dump specific tensors, dump all tensor as defalut.
* is_soc: Bool type, representing whether to use in soc mode.
* using_memory_opt: Bool type, enable to use memory opt, decrease memory usage at the expense of increasing time cost. Suggest to enable when running large model.
* enable_soc_log: Bool type, enable to print and save log at `save_path`.
* soc_tmp_path: String type, representing the abs path of tmp files and tools on device in soc mode.
* hostname: String type, representing the ip address of device in soc mode.
* port: Int type, representing the port of device in soc mode.
* username: String type, representing the username of device in soc mode.
* password: String type, representing the password of device in soc mode.

Attention:

* When the funciton is called in cmodel/pcie mode, functions `use_cmodel/use_chip` from `/tpu-mlir/envsetup.sh` is required.
* When the funciton is called in soc mode, use `use_chip` and `reference_data_fn` must be .npz.

Returns
"""""""""""""""""""""""""""""""""
* cmodel/pcie mode: if `dump_file=True`, then bmodel_infer_xxx.npz will be generated in `save_path`, otherwise return python dict.
* soc mode: soc_infer_xxx.npz will be generated in `save_path`.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1688:  only cmodel mode.
* BM1684X: cmodel/pcie/soc mode.

scatter
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def scatter(input: Tensor,
            index: Tensor,
            updates: Tensor,
            axis: int = 0,
            out_name: str = None):
        #pass

Description
"""""""""""""""""""""""""""""""""
Based on the specified indices, write the input data to specific positions in the target Tensor. This operation allows the elements of the input Tensor to be scattered to the specified positions in the output Tensor. Refer to the ScatterElements operation in various frameworks for more details.
This operation belongs to **local operation**。

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, represents the input operation Tensor, i.e., the target Tensor that needs to be updated.
* index: Tensor type, represents the index Tensor that specifies the update positions.
* updates: Tensor type, represents the values to be written into the target Tensor.
* axis: int type, represents the axis along which to update.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.


Returns
"""""""""""""""""""""""""""""""""
Returns a new Tensor with updates applied at the specified positions, while other positions retain the original values from the input Tensor.



Processor Support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.
* BM1688: The input data type can be FLOAT32/UINT8/INT8.


scatterND
:::::::::::::::::::::::::::::::::::::::::::::::::

Definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def scatterND(input: Tensor,
            indices: Tensor,
            updates: Tensor,
            out_name: str = None):
        #pass

Description
"""""""""""""""""""""""""""""""""
Based on the specified indices, write the input data to specific positions in the target Tensor. This operation allows the elements of the input Tensor to be scattered to the specified positions in the output Tensor. Refer to the scatterND operation in ONNX 11 for more details.
This operation belongs to **local operation**。

Parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, represents the input operation Tensor, i.e., the target Tensor that needs to be updated.
* indices: Tensor type, represents the index Tensor that specifies the update positions. The datatype must be uint32.
* updates: Tensor type, represents the values to be written into the target Tensor. Rank(updates) = Rank(input) + Rank(indices) - shape(indices)[-1] -1.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.

Returns
"""""""""""""""""""""""""""""""""
Returns a new Tensor with updates applied at the specified positions, while other positions retain the original values from the input Tensor. The shape and datatype are the same with the input tensor.

Processor Support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.
* BM1688: The input data type can be FLOAT32/UINT8/INT8.


Preprocess Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mean_std_scale
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mean_std_scale(input: Tensor,
                         std: List[float],
                         mean: List[float],
                         scale: Optional[Union[List[float],List[int]]] = None,
                         zero_points: Optional[List[int]] = None,
                         out_name: str = None,
                         odtype="float16",
                         round_mode: str = "half_away_from_zero"):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Preproces input Tensor data.
This operation is considered a **global operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, representing the input data.
* std: List[float], representing the standard deviation of the dataset. The dimensions of mean and std must match the channel dimension of the input, i.e., the second dimension of the input.
* mean: List[float], representing the mean of the dataset. The dimensions of mean and std must match the channel dimension of the input, i.e., the second dimension of the input.
* scale: Optional[Union[List[float],List[int]]] type or None, reprpesenting the scale factor.
* zero_points: Optional[List[int]] type or None,representing the zero point.
* out_name: string type or None, representing the name of Tensor, tpulang will auto generate name if out_name is None.
* odtype: String, representing the data type of the output Tensor. Default is "float16". Currently supports float16 and int8.
* round_mode: String, representing the rounding method. Default is "half_away_from_zero", with options "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the type of odtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/UINT8/INT8, the output data type can be INT8/FLOAT16.


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


rope
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rope( input: Tensor,
                weight0: Tensor,
                weight1: Tensor,
                is_permute_optimize: bool = False,    # unused
                mul1_round_mode: str = 'half_up',
                mul2_round_mode: str= 'half_up',
                add_round_mode: str = 'half_up',
                mul1_shift: int = None,
                mul2_shift: int = None,
                add_shift: int = None,
                mul1_saturation: bool = True,
                mul2_saturation: bool = True,
                add_saturation: bool = True,
                out_name: str = None):
            #pass

Description of the function
"""""""""""""""""""""""""""""""""
Perform a rotation encoding (RoPE) operation on the input Tensor.
This operation belongs to **global operation**

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, indicating the input operation Tensor. It must be four-dimensional.
* weight0: Tensor, indicating the input operation Tensor.
* weight1: Tensor, indicating the input operation Tensor.
* is_permute_optimize: bool type, indicating whether to perform permute sinking and check the shape of permute sinking. # unused
* mul1_round_mode: Type String, representing the rounding method of mul1 in RoPE. The default value is "half_away_from_zero", and the range is "half_away_from_zero", "half_to_even "," towards_zero ", "down "," up ", "half_up "," half_down ".
* mul2_round_mode: Type String, representing the rounding method of mul2 in RoPE. The default value is "half_away_from_zero", and the range is "half_away_from_zero", "half_to_even "," towards_zero ", "down "," up ", "half_up "," half_down ".
* add_round_mode: Type String, representing the rounding method of add in RoPE. The default value is "half_away_from_zero", and the range is "half_away_from_zero", "half_to_even "," towards_zero ", "down "," up ", "half_up "," half_down ".
* mul1_shift: int type, representing the number of bits of the shift of mul1 in RoPE.
* mul2_shift: int type, indicating the number of bits of the shift of mul2 in RoPE.
* add_shift: int type, indicating the number of bits shifted by add in RoPE.
* mul1_saturation: bool type, indicating whether the calculation result of mul1 in RoPE requires saturation processing. The default is True saturation processing, and no modification is needed unless necessary.
* mul2_saturation: bool type, indicating whether the calculation result of mul2 in RoPE requires saturation processing. The default is True saturation processing, and no modification is needed unless necessary.
* add_saturation: bool type, indicating whether the add calculation result in RoPE requires saturation processing. The default is True saturation processing, and no modification is needed unless necessary.
* out_name: output name, type string, default to None.

Return value
"""""""""""""""""""""""""""""""""
Return a Tensor with the data type of odtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data types can be FLOAT32,FLOAT16 and INT types.


multi_scale_deformable_attention
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
    .. code-block:: python

      def multi_scale_deformable_attention(
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        sampling_offsets_weight: Tensor,
        sampling_offsets_bias_ori: Tensor,
        attention_weights_weight: Tensor,
        attention_weights_bias_ori: Tensor,
        value_proj_weight: Tensor,
        value_proj_bias_ori: Tensor,
        output_proj_weight: Tensor,
        output_proj_bias_ori: Tensor,
        spatial_shapes: List[List[int]],
        embed_dims: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        out_name: str = None):

        #pass

Description of the function
"""""""""""""""""""""""""""""""""
Perform multi-scale deformable attention on the input, and the specific function can refer to https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/multi_scale_deform_attn.py:MultiScaleDeformableAttention:forward, the implementation of this operation is different from the official one.
This operation is considered a **global operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* query: Tensor type, query of Transformer with shape (1, num_query, embed_dims).
* value: Tensor type, the value tensor with shape (1, num_key, embed_dims).
* key_padding_mask: Tensor type, the mask of the query tensor with shape (1, num_key).
* reference_points: Tensor type, normalized reference points with shape (1, num_query, num_levels, 2), all elements are in the range [0, 1], the upper left corner is (0,0), and the lower right corner is (1,1), including the padding area.
* sampling_offsets_weight: Tensor type, the weight of the fully connected layer for calculating the sampling offset with shape (embed_dims, num_heads\*num_levels\*num_points\*2).
* sampling_offsets_bias_ori: Tensor type, the bias of the fully connected layer for calculating the sampling offset with shape (num_heads\*num_levels\*num_points\*2).
* attention_weights_weight: Tensor type, the weight of the fully connected layer for calculating the attention weight with shape (embed_dims, num_heads\*num_levels\*num_points).
* attention_weights_bias_ori: Tensor type, the bias of the fully connected layer for calculating the attention weight with shape (num_heads\*num_levels\*num_points).
* value_proj_weight: Tensor type, the weight of the fully connected layer for calculating the value projection with shape (embed_dims, embed_dims).
* value_proj_bias_ori: Tensor type, the bias of the fully connected layer for calculating the value projection with shape (embed_dims).
* output_proj_weight: Tensor type, the weight of the fully connected layer for calculating the output projection with shape (embed_dims, embed_dims).
* output_proj_bias_ori: Tensor type, the bias of the fully connected layer for calculating the output projection with shape (embed_dims).
* spatial_shapes: List[List[int]] type, the spatial shape of different level features with shape (num_levels, 2), the last dimension represents (h, w).
* embed_dims: int type, hidden_size of query, key, and value.
* num_heads: int type, the number of attention heads, default is 8.
* num_levels: int type, the number of levels of multi-scale attention, default is 4.
* num_points: int type, the number of sampling points at each level, default is 4.
* out_name: string type or None, the name of the output Tensor, and the name will be automatically generated internally if it is None.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as query.dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/FLOAT16.
* BM1688: The input data type can be FLOAT32/FLOAT16.


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a16matmul
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
    .. code-block:: python

      def a16matmul(input: Tensor,
                    weight: Tensor,
                    scale: Tensor,
                    zp: Tensor,
                    bias: Tensor = None,
                    right_transpose=True,
                    out_dtype: str = 'float16',
                    out_name: str = None,
                    group_size: int = 128,
                    bits: int = 4,
                    g_idx: Tensor = None,
                    ):

        #pass

Description of the function
"""""""""""""""""""""""""""""""""
Perform W4A16/W8A16 MatMul on the input.
This operation is considered a **global operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* input: Tensor type, represents the input tensor.
* weight: Tensor type, represents the weight after 4-bit/8-bit quantization, stored as int32.
* scale: Tensor type, represents the quantization scaling factor for the weights, stored as float32.
* zp: Tensor type, represents the quantization zero point for the weights, stored as int32.
* bias: Tensor type, represents the bias, stored as float32.
* right_transpose: Boolean type, indicates whether the weight matrix is transposed; currently only supports True.
* out_dtype: String type, represents the data type of the output tensor.
* out_name: String type or None, represents the name of the output Tensor; if None, a name will be automatically generated internally.
* group_size: Integer type, indicates the group size for quantization.
* bits: Integer type, represents the quantization bit-width; only supports 4 bits/8 bits.
* g_idx: Tensor type, represents the quantization reordering coefficient; currently not supported.

Return value
"""""""""""""""""""""""""""""""""
Returns a Tensor with the same data type as out_dtype。

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/FLOAT16.
* BM1688: The input data type can be FLOAT32/FLOAT16.


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qwen2_block
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
    .. code-block:: python

      def qwen2_block(hidden_states: Tensor,
                      position_ids: Tensor,
                      attention_mask: Tensor,
                      q_proj_weights: Tensor,
                      q_proj_scales: Tensor,
                      q_proj_zps: Tensor,
                      q_proj_bias: Tensor,
                      k_proj_weights: Tensor,
                      k_proj_scales: Tensor,
                      k_proj_zps: Tensor,
                      k_proj_bias: Tensor,
                      v_proj_weights: Tensor,
                      v_proj_scales: Tensor,
                      v_proj_zps: Tensor,
                      v_proj_bias: Tensor,
                      o_proj_weights: Tensor,
                      o_proj_scales: Tensor,
                      o_proj_zps: Tensor,
                      o_proj_bias: Tensor,
                      down_proj_weights: Tensor,
                      down_proj_scales: Tensor,
                      down_proj_zps: Tensor,
                      gate_proj_weights: Tensor,
                      gate_proj_scales: Tensor,
                      gate_proj_zps: Tensor,
                      up_proj_weights: Tensor,
                      up_proj_scales: Tensor,
                      up_proj_zps: Tensor,
                      input_layernorm_weight: Tensor,
                      post_attention_layernorm_weight: Tensor,
                      cos: List[Tensor],
                      sin: List[Tensor],
                      out_dtype: str = 'float16',
                      group_size: int = 128,
                      weight_bits: int = 4,
                      hidden_size: int = 3584,
                      rms_norm_eps: float = 1e-06,
                      num_attention_heads: int = 28,
                      num_key_value_heads: int = 4,
                      mrope_section: List[int] = [16, 24, 24],
                      quant_method: str = "gptq",
                      out_name: str = None
                      ):

        #pass

Description of the function
"""""""""""""""""""""""""""""""""
A block layer of qwen2 during the prefill stage.
This operation is considered a **global operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* hidden_states: Tensor type, representing activation values, with shape (1, seq_length, hidden_size).
* position_ids: Tensor type, representing positional indices, with shape (3, 1, seq_length).
* attention_mask: Tensor type, representing the attention mask, with shape (1, 1, seq_length, seq_length).
* q_proj_weights: Tensor type, representing the quantized query weights, stored as int32.
* q_proj_scales: Tensor type, representing the quantization scaling factors for the query, stored as float32.
* q_proj_zps: Tensor type, representing the quantization zero-points for the query, stored as int32.
* q_proj_bias: Tensor type, representing the query bias, stored as float32.
* k_proj_weights: Tensor type, representing the quantized key weights, stored as int32.
* k_proj_scales: Tensor type, representing the quantization scaling factors for the key, stored as float32.
* k_proj_zps: Tensor type, representing the quantization zero-points for the key, stored as int32.
* k_proj_bias: Tensor type, representing the key bias, stored as float32.
* v_proj_weights: Tensor type, representing the quantized value weights, stored as int32.
* v_proj_scales: Tensor type, representing the quantization scaling factors for the value, stored as float32.
* v_proj_zps: Tensor type, representing the quantization zero-points for the value, stored as int32.
* v_proj_bias: Tensor type, representing the value bias, stored as float32.
* o_proj_weights: Tensor type, representing the quantized output projection weights, stored as int32.
* o_proj_scales: Tensor type, representing the quantization scaling factors for the output projection, stored as float32.
* o_proj_zps: Tensor type, representing the quantization zero-points for the output projection, stored as int32.
* o_proj_bias: Tensor type, representing the output projection bias, stored as float32.
* down_proj_weights: Tensor type, representing the quantized down projection layer weights, stored as int32.
* down_proj_scales: Tensor type, representing the quantization scaling factors for the down projection layer, stored as float32.
* down_proj_zps: Tensor type, representing the quantization zero-points for the down projection layer, stored as int32.
* gate_proj_weights: Tensor type, representing the quantized gate projection layer weights, stored as int32.
* gate_proj_scales: Tensor type, representing the quantization scaling factors for the gate projection layer, stored as float32.
* gate_proj_zps: Tensor type, representing the quantization zero-points for the gate projection layer, stored as int32.
* up_proj_weights: Tensor type, representing the quantized up projection layer weights, stored as int32.
* up_proj_scales: Tensor type, representing the quantization scaling factors for the up projection layer, stored as float32.
* up_proj_zps: Tensor type, representing the quantization zero-points for the up projection layer, stored as int32.
* input_layernorm_weight: Tensor type, representing the weights for layer normalization on the input, stored as int32.
* post_attention_layernorm_weight: Tensor type, representing the weights for layer normalization on the attention layer output, stored as int32.
* cos: List[Tensor] type, representing the cosine positional encodings.
* sin: List[Tensor] type, representing the sine positional encodings.
* out_dtype: string type, representing the data type of the output tensor.
* group_size: int type, representing the group size used for quantization.
* weight_bits: int type, representing the quantization bit width, currently only supports 4 bits/8 bits.
* hidden_size: int type, representing the hidden size for the query/key/value.
* rms_norm_eps: float type, representing the epsilon parameter in layer normalization.
* num_attention_heads: int type, representing the number of attention heads.
* num_key_value_heads: int type, representing the number of key/value heads.
* mrope_section: List[int] type, representing the sizes of the three dimensions for the positional encoding.
* quant_method: str type, representing the quantization method, currently only GPTQ quantization is supported.
* out_name: string type or None, representing the name of the output tensor; if None, the name will be automatically generated.

Return value
"""""""""""""""""""""""""""""""""
Returns 3 Tensors: the activation output, the key cache, and the value cache, all with the data type specified by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/FLOAT16.
* BM1688: The input data type can be FLOAT32/FLOAT16.


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qwen2_block_cache
:::::::::::::::::::::::::::::::::::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""
    .. code-block:: python

      def qwen2_block_cache(hidden_states: Tensor,
                            position_ids: Tensor,
                            attention_mask: Tensor,
                            k_cache: Tensor,
                            v_cache: Tensor,
                            q_proj_weights: Tensor,
                            q_proj_scales: Tensor,
                            q_proj_zps: Tensor,
                            q_proj_bias: Tensor,
                            k_proj_weights: Tensor,
                            k_proj_scales: Tensor,
                            k_proj_zps: Tensor,
                            k_proj_bias: Tensor,
                            v_proj_weights: Tensor,
                            v_proj_scales: Tensor,
                            v_proj_zps: Tensor,
                            v_proj_bias: Tensor,
                            o_proj_weights: Tensor,
                            o_proj_scales: Tensor,
                            o_proj_zps: Tensor,
                            o_proj_bias: Tensor,
                            down_proj_weights: Tensor,
                            down_proj_scales: Tensor,
                            down_proj_zps: Tensor,
                            gate_proj_weights: Tensor,
                            gate_proj_scales: Tensor,
                            gate_proj_zps: Tensor,
                            up_proj_weights: Tensor,
                            up_proj_scales: Tensor,
                            up_proj_zps: Tensor,
                            input_layernorm_weight: Tensor,
                            post_attention_layernorm_weight: Tensor,
                            cos: List[Tensor],
                            sin: List[Tensor],
                            out_dtype: str = 'float16',
                            group_size: int = 128,
                            weight_bits: int = 4,
                            hidden_size: int = 3584,
                            rms_norm_eps: float = 1e-06,
                            num_attention_heads: int = 28,
                            num_key_value_heads: int = 4,
                            mrope_section: List[int] = [16, 24, 24],
                            quant_method: str = "gptq",
                            out_name: str = None
                            ):

        #pass

Description of the function
"""""""""""""""""""""""""""""""""
A block layer of qwen2 during the decode stage.
This operation is considered a **global operation** 。

Explanation of parameters
"""""""""""""""""""""""""""""""""
* hidden_states: Tensor type, representing activation values, with shape (1, 1, hidden_size).
* position_ids: Tensor type, representing positional indices, with shape (3, 1, 1).
* attention_mask: Tensor type, representing the attention mask, with shape (1, 1, 1, seq_length + 1).
* k_cache: Tensor type, representing the key cache. Its shape is (1, seq_length, num_key_value_heads, head_dim).
* v_cache: Tensor type, representing the value cache. Its shape is (1, seq_length, num_key_value_heads, head_dim).
* q_proj_weights: Tensor type, representing the quantized query weights, stored as int32.
* q_proj_scales: Tensor type, representing the quantization scaling factors for the query, stored as float32.
* q_proj_zps: Tensor type, representing the quantization zero-points for the query, stored as int32.
* q_proj_bias: Tensor type, representing the query bias, stored as float32.
* k_proj_weights: Tensor type, representing the quantized key weights, stored as int32.
* k_proj_scales: Tensor type, representing the quantization scaling factors for the key, stored as float32.
* k_proj_zps: Tensor type, representing the quantization zero-points for the key, stored as int32.
* k_proj_bias: Tensor type, representing the key bias, stored as float32.
* v_proj_weights: Tensor type, representing the quantized value weights, stored as int32.
* v_proj_scales: Tensor type, representing the quantization scaling factors for the value, stored as float32.
* v_proj_zps: Tensor type, representing the quantization zero-points for the value, stored as int32.
* v_proj_bias: Tensor type, representing the value bias, stored as float32.
* o_proj_weights: Tensor type, representing the quantized output projection weights, stored as int32.
* o_proj_scales: Tensor type, representing the quantization scaling factors for the output projection, stored as float32.
* o_proj_zps: Tensor type, representing the quantization zero-points for the output projection, stored as int32.
* o_proj_bias: Tensor type, representing the output projection bias, stored as float32.
* down_proj_weights: Tensor type, representing the quantized down projection layer weights, stored as int32.
* down_proj_scales: Tensor type, representing the quantization scaling factors for the down projection layer, stored as float32.
* down_proj_zps: Tensor type, representing the quantization zero-points for the down projection layer, stored as int32.
* gate_proj_weights: Tensor type, representing the quantized gate projection layer weights, stored as int32.
* gate_proj_scales: Tensor type, representing the quantization scaling factors for the gate projection layer, stored as float32.
* gate_proj_zps: Tensor type, representing the quantization zero-points for the gate projection layer, stored as int32.
* up_proj_weights: Tensor type, representing the quantized up projection layer weights, stored as int32.
* up_proj_scales: Tensor type, representing the quantization scaling factors for the up projection layer, stored as float32.
* up_proj_zps: Tensor type, representing the quantization zero-points for the up projection layer, stored as int32.
* input_layernorm_weight: Tensor type, representing the weights for layer normalization on the input, stored as int32.
* post_attention_layernorm_weight: Tensor type, representing the weights for layer normalization on the attention layer output, stored as int32.
* cos: List[Tensor] type, representing the cosine positional encodings.
* sin: List[Tensor] type, representing the sine positional encodings.
* out_dtype: string type, representing the data type of the output tensor.
* group_size: int type, representing the group size used for quantization.
* weight_bits: int type, representing the quantization bit width, currently only supports 4 bits/8 bits.
* hidden_size: int type, representing the hidden size for the query/key/value.
* rms_norm_eps: float type, representing the epsilon parameter in layer normalization.
* num_attention_heads: int type, representing the number of attention heads.
* num_key_value_heads: int type, representing the number of key/value heads.
* mrope_section: List[int] type, representing the sizes of the three dimensions for the positional encoding.
* quant_method: str type, representing the quantization method, currently only GPTQ quantization is supported.
* out_name: string type or None, representing the name of the output tensor; if None, the name will be automatically generated.

Return value
"""""""""""""""""""""""""""""""""
Returns 3 Tensors: the activation output, the key cache, and the value cache, all with the data type specified by out_dtype.

Processor support
"""""""""""""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/FLOAT16.
* BM1688: The input data type can be FLOAT32/FLOAT16.
