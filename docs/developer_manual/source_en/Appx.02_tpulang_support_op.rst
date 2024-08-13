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
                     dtype: str = "float32")
               #pass


As shown above, a Tensor in TpuLang has five parameters:

* shape: The shape of the Tensor, a List[int]. For Tensors that serve as the output of an Operator, the shape can be left unspecified with a default value of [].
* Name: The name of the Tensor, a string or None. It is recommended to use the default value None to avoid potential issues arising from identical names.
* ttype: The type of the Tensor, which can be "neuron," "coeff," or None. The initial value is "neuron."
* data: The input data for the Tensor. If the default value None is used, the Tensor will be initialized with all zeros based on the specified shape. Otherwise, it should be an ndarray.
* dtype: The data type of the Tensor, with a default value of "float32." Other possible values include "float32," "float16," "int32," "uint32," "int16," "uint16," "int8," and "uint8."


Example of declaring a Tensor:

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      #weight
      weight = tpul.Tensor(dtype='float32', shape=[3,4], data=np.random.uniform(0,1,shape).astype('float32'), ttype="coeff")

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
:::::::::::::::::::::::::::::

    .. code-block:: python

        def compile(name: str,
            inputs: List[Tensor],
            outputs: List[Tensor],
            cmp=True,
            refs=None,
            mode='f32',
            dynamic=False,
            asymmetric=False,
            no_save=False,
            opt=2,
            mlir_inference=True,
            bmodel_inference=True,log_level="normal"):
            #pass


Description of the function
:::::::::::::::::::::::::::::

The function for comipling TpuLang model to bmodel.

Explanation of parameters
:::::::::::::::::::::::::::::

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

.. _deinit:


Deinitialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After constructing the network, it is necessary to perform deinitialization to conclude the process.
Only after deinitialization, the TPU executable target generated by the previously initiated compilation will be saved to the specified output directory.

    .. code-block:: python

       def deinit():
          #pass

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
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv(input,
            weight,
            bias=None,
            kernel=None,
            dilation=None,
            pad=None,
            stride=None,
            groups=1,
            out_name=None):
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
* kernel: This parameter is currently deprecated and not used.
* dilation: List of integers, representing the dilation size. If None, it is [1, 1]. If not None, it requires a length of 1 or 2.
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 1 or 2 or 4.
* stride: List of integers, representing the stride size. If None, it is [1, 1]. If not None, it requires a length of 1 or 2.
* groups: An integer, representing the number of groups in the convolution layer.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


conv_v2
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv_v2(tensor_i,
                  weight,
                  bias = None,
                  stride = None,
                  dilation = None,
                  pad = None,
                  group = 1,
                  input_zp = None,
                  weight_zp = None,
                  out_dtype = None,
                  out_name = None):
          # pass

Description of the function
"""""""""""""""""""""""""""""""""

Fixed-point two-dimensional convolution operation. You can refer to the definitions of fixed-point 2D convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) Conv (weight - kzp) + bias[c];

This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input Tensor in 4D NCHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 4D [oc, ic, kh, kw] format. Here, oc represents the number of output channels, ic represents the number of input channels, kh is the kernel height, and kw is the kernel width.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1].
* dilation: List of integers, representing the dilation size. If None, it is [1, 1]. If not None, it requires a length of 2. The order in the list is [height, width].
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 4. The order in the list is [top, bottom, left, right].
* stride: List of integers, representing the stride size. If None, it is [1, 1]. If not None, it requires a length of 2. The order in the list is [height, width].
* groups: An integer, representing the number of groups in the convolution layer. If ic=oc=groups, the convolution is depthwise.
* input_zp: List of integers or an integer, representing the input offset. If None, it is 0. If a list is provided, it should have a length of ic.
* weight_zp: List of integers or an integer, representing the kernel offset. If None, it is 0. If a list is provided, it should have a length of ic, where ic represents the number of input channels.
* out_dtype: A string or None, representing the data type of the input Tensor. If None, it is int32. Possible values: int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""""""
Returns a Tensor with the data type determined by out_dtype.

Processor support
""""""""""""""""""""""
BM1688: The input data type can be INT8/UINT8.
BM1684X: The input data type can be INT8/UINT8.


deconv
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def deconv(input,
            weight,
            bias=None,
            kernel=None,
            dilation=None,
            pad=None,
            output_padding = None,
            stride=None,
            output_padding=None,
            out_name=None):
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
* kernel: This parameter is currently deprecated and not used.
* dilation: List of integers, representing the dilation size. If None, it is [1, 1]. If not None, it requires a length of 1 or 2.
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 1 or 2 or 4.
* output_padding: List of integers, representing the output padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 1 or 2 or 4.
* stride: List of integers, representing the stride size. If None, it is [1, 1]. If not None, it requires a length of 1 or 2.
* output_padding: List of integers, representing the padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 1 or 2 or 4.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


deconv_v2
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def deconv_v2(tensor_i,
                    weight,
                    bias = None,
                    stride = None,
                    dilation = None,
                    pad = None,
                    output_padding = None,
                    group = 1,
                    input_zp = None,
                    weight_zp = None,
                    out_dtype = None,
                    out_name = None):


Description of the function
"""""""""""""""""""""""""""""""""

Fixed-point two-dimensional transposed convolution operation. You can refer to the definitions of fixed-point 2D transposed convolution in various frameworks.
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) DeConv (weight - kzp) + bias[c];

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input Tensor in 4D NCHW format.
* weight: Tensor type, representing the convolutional kernel Tensor in 4D [ic, oc, kh, kw] format. Here, oc represents the number of output channels, ic represents the number of input channels, kh is the kernel height, and kw is the kernel width.
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [1, oc, 1, 1].
* dilation: List of integers, representing the dilation size. If None, it is [1, 1]. If not None, it requires a length of 2. The order in the list is [height, width].
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 4. The order in the list is [top, bottom, left, right].
* output_padding: List of integers, representing the output padding size. If None, it is [0, 0, 0, 0]. If not None, it requires a length of 1 or 2 or 4.
* stride: List of integers, representing the stride size. If None, it is [1, 1]. If not None, it requires a length of 2. The order in the list is [height, width].
* groups: An integer, representing the number of groups in the convolution layer. If ic=oc=groups, the convolution is depthwise deconvolution.
* input_zp: List of integers or an integer, representing the input offset. If None, it is 0. If a list is provided, it should have a length of ic.
* weight_zp: List of integers or an integer, representing the kernel offset. If None, it is 0. If a list is provided, it should have a length of ic, where ic represents the number of input channels.
* out_dtype: A string or None, representing the data type of the input Tensor. If None, it is int32. Possible values: int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""
Returns a Tensor with the data type determined by out_dtype.

Processor support
""""""""""""""""""""""
BM1688: The input data type can be INT8/UINT8.
BM1684X: The input data type can be INT8/UINT8.


conv3d
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv3d(input,
            weight,
            bias=None,
            kernel=None,
            dilation=None,
            pad=None,
            stride=None,
            groups=1,
            out_name=None):
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
* kernel: This parameter is currently deprecated and not used.
* dilation: List of integers, representing the dilation size. If None, it is [1, 1, 1]. If not None, it requires a length of 1 or 3.
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0, 0, 0]. If not None, it requires a length of 1 or 3 or 6.
* stride: List of integers, representing the stride size. If None, it is [1, 1, 1]. If not None, it requires a length of 1 or 3.
* groups: An integer, representing the number of groups in the convolution layer.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


conv3d_v2
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def conv3d_v2(tensor_i,
                    weight,
                    bias = None,
                    stride = None,
                    dilation = None,
                    pad = None,
                    group = 1,
                    input_zp = None,
                    weight_zp = None,
                    out_dtype = None,
                    out_name = None):


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
* dilation: List of integers, representing the dilation size. If None, it is [1, 1, 1]. If not None, it requires a length of 2. The order in the list is [dilation_t, dilation_h, dilation_w].
* pad: List of integers, representing the padding size. If None, it is [0, 0, 0, 0, 0, 0]. If not None, it requires a length of 6. The order in the list is [before, after, top, bottom, left, right].
* stride: List of integers, representing the stride size. If None, it is [1, 1, 1]. If not None, it requires a length of 3. The order in the list is [stride_t, stride_h, stride_w].
* groups: An integer, representing the number of groups in the convolution layer. If ic=oc=groups, the convolution is depthwise conv3d.
* input_zp: List of integers or an integer, representing the input offset. If None, it is 0. If a list is provided, it should have a length of ic.
* weight_zp: List of integers or an integer, representing the kernel offset. If None, it is 0. If a list is provided, it should have a length of ic, where ic represents the number of input channels.
* out_dtype: A string or None, representing the data type of the input Tensor. If None, it is int32. Possible values: int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""
Returns a Tensor with the data type determined by out_dtype.

Processor support
""""""""""""""""""""""
BM1688: The input data type can be INT8/UINT8.
BM1684X: The input data type can be FINT8/UINT8.

matrix_mul
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def matrix_mul(lhs, rhs, bias=None, left_zp=None, right_zp=None, \
                     out_dtype=None, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Matrix multiplication operation. You can refer to the definitions of matrix multiplication in various frameworks.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* lhs: Tensor type, representing the input left operand, with dimensions greater than or equal to 2, and the last two dimensions have shape=[m, k].
* rhs: Tensor type, representing the input right operand, with dimensions greater than or equal to 2, and the last two dimensions have shape=[k, n].
* bias: Tensor type, representing the bias Tensor. If None, it indicates no bias. Otherwise, it requires a shape of [n].
* left_zp: List of integers or an integer, representing the offset of lhs. If None, it is 0. If a list is provided, it should have a length of k. This parameter is only useful when the dtype of lhs is 'int8/uint8'. Currently, only 0 is supported.
* right_zp: List of integers or an integer, representing the offset of rhs. If None, it is 0. If a list is provided, it should have a length of k. This parameter is only useful when the dtype of rhs is 'int8/uint8'.
* out_dtype: A string or None, representing the data type of the input Tensor. If None, it is consistent with the dtype of lhs. When the dtype of lhs is 'int8/uint8', the possible values for out_dtype are int32/uint32.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

It is required that the dimensions of the left and right tensors are consistent. When the dimensions of the tensor are 2, it represents matrix-matrix multiplication. When the dimensions of the tensor are greater than 2, it represents batch matrix multiplication.
It is required that lhr.shape[-1] == rhs.shape[-2], and lhr.shape[:-2] and rhs.shape[:-2] need to satisfy the broadcasting rules.

Return value
"""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

Base Element-wise Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add
:::::::::::::::::

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
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype or is consistent with the input data type (when one of the inputs is 'int8', the output defaults to 'int8' type). When the input is 'float32'/'float16', the output data type must be consistent with the input.


Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sub
:::::::::::::::::

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
* out_dtype: A string type or None, representing the data type of the output tensor. If None, it is consistent with the input tensors' dtype. The optional parameters are 'float32'/'float16'/'int8'/'int16'/'int32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is 'float32'/'float16',
the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output data type is 'int8'/'int16'/'int32'.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


mul
:::::::::::::::::

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
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor whose data type is specified by out_dtype or is consistent with the input data type (when one of the inputs is 'int8', the output defaults to 'int8' type). When the input is 'float32'/'float16', the output data type must be consistent with the input.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


div
:::::::::::::::::

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
"""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


max
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

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
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is 'float32'/'float16',
the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.


min
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

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
""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type.
When the input is 'float32'/'float16', the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8.


copy
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


clamp
:::::::::::::::::

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
""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor.
* min_value: Scalar type, representing the lower bound of the range.
* max_value: Scalar type, representing the upper bound of the range.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gt
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def gt(tensor_i0, tensor_i1, out_name = None):
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
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

lt
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def lt(tensor_i0, tensor_i1, out_name = None):
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
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

ge
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""

    .. code-block:: python

      def ge(tensor_i0, tensor_i1, out_name = None):
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
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

le
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def le(tensor_i0, tensor_i1, out_name = None):
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
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

eq
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""

    .. code-block:: python

      def eq(tensor_i0, tensor_i1, out_name = None):
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
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

ne
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def ne(tensor_i0, tensor_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise not equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 != tensor\_i1 ? 1 : 0`.
This operation supports broadcasting.
tensor_i0 or tensor_i1 can be assigned as COEFF_TENSOR.
This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input Tensor.
* tensor_i1: Tensor type, representing the right operand input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

gts
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def gts(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise greater-than comparison operation between tensors and scalars. :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

lts
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def lts(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise less-than comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

ges
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ges(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise greater-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

les
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def les(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise less-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

eqs
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def eqs(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The element-wise equality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

nes
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nes(tensor_i0, scalar_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The element-wise inequality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type, representing the left operand input.
* scalar_i1: Tensor type, representing the right operand input.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

Activation Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

relu
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def relu(tensor, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


leaky_relu
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def leaky_relu(tensor, negative_slope=0.01, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The leaky ReLU activation function, implemented on an element-wise basis. :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* negative_slope: A FLOAT type, representing the negative slope of the input.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

abs
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def abs(tensor, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

ln
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ln(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ln activation function, implemented on an element-wise basis. :math:`y = log(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

square
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def square(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The square function, implemented on an element-wise basis. :math:`y = x*x`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

ceil
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ceil(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ceil rounding up activation function, implemented on an element-wise basis. :math:`y = \left \lfloor x \right \rfloor`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

floor
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def floor(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The floor rounding down activation function, implemented on an element-wise basis. :math:`y = \left \lceil x \right \rceil`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

round
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def round(tensor, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sin
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sin(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sin sine activation function, implemented on an element-wise basis. :math:`y = sin(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


cos
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cos(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The cos cosine activation function, implemented on an element-wise basis. :math:`y = cos(x)`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

exp
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def exp(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The exp exponential activation function, implemented on an element-wise basis. :math:`y = e^{x}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

tanh
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tanh(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The tanh hyperbolic tangent activation function, implemented on an element-wise basis. :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

sigmoid
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sigmoid(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sigmoid activation function, implemented on an element-wise basis. :math:`y = 1 / (1 + e^{-x})`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

elu
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def elu(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The ELU (Exponential Linear Unit) activation function, implemented on an element-wise basis. :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

sqrt
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sqrt(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sqrt square root activation function, implemented on an element-wise basis. :math:`y = \sqrt{x}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

rsqrt
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rsqrt(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The rsqrt square root  takes the deactivation function, implemented on an element-wise basis. :math:`y = 1 / (sqrt{x})`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

silu
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def silu(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The silu activation function, implemented on an element-wise basis. :math:`y = x * (1 / (1 + e^{-x}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

erf
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def erf(tensor, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The erf activation function, for the corresponding elements x and y at the same positions in the input and output Tensors,
is implemented on an element-wise basis. :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

tan
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tan(tensor, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


softmax
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def softmax(tensor_i, axis, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


mish
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mish(tensor_i,  out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The Mish activation function, implemented on an element-wise basis.:math:`y = x * tanh(ln(1 + e^{x}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.



hswish
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hswish(tensor_i, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The h-swish activation function, implemented on an element-wise basis. :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.



arccos
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arccos(tensor_i, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


arctanh
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arctanh(tensor_i, out_name=None):
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
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sinh
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sinh(tensor_i, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sinh (hyperbolic sine) activation function, implemented on an element-wise basis. :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.



cosh
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cosh(tensor_i,  out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The cosh (hyperbolic cosine) activation function, implemented on an element-wise basis. :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sign
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sign(tensor_i, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The sign activation function, implemented on an element-wise basis. :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


gelu
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def gelu(tensor_i, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The GELU (Gaussian Error Linear Unit) activation function, implemented on an element-wise basis. :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


hsigmoid
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hsigmoid(tensor_i,  out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
The hsigmoid (hard sigmoid) activation function, implemented on an element-wise basis. :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A Tensor type, representing the input Tensor.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same shape and data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

Data Arrange Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

permute
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

tile
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

broadcast
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


concat
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

split
:::::::::::::::::

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
""""""""""""""""""""""
Returns a `List[Tensor]`, where each `Tensor` has the same data type as the input `Tensor`.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

pad
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

repeat
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

extract
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def extract(input: Tensor,
                    start: Union[List[int], Tuple[int]] = None,
                    end: Union[List[int], Tuple[int]] = None,
                    stride: Union[List[int], Tuple[int]] = None,
                    out_name: str = None)

Description
"""""""""""
Extract slice of input tensor.
This operation is considered a **restricted local operation**.

Parameters
"""""""""""
* input: Tensor type, representing input tensor.
* start: A list or tuple of int, or None, representing the start of slice. If set to None, `start`` is filled all with 0.
* end: A list or tuple of int, or None, representing the end of slice. If set to None, `end`` is given as shape of input.
* stride: A list or tuple of int, or None, representing the stride of slice. If set to None, `stride` is filled all with 1.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns a Tensor, whose data type is same of that of `table`.

Processor Support
""""""""""""""""""""""
* BM1688:  Data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: Data type can be FLOAT32/FLOAT16/INT8/UINT8.


roll
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def roll(input:Tensor,
                shifts: Union[int, List[int], Tuple[int]],
                dims: Union[int, List[int], Tuple[int]]   = None,
                out_name:str=None):
          #pass

Description
"""""""""""
Roll the tensor input along the given dimension(s). Elements that are shifted beyond the last position are re-introduced at the first position. If dims is None, the tensor will be flattened before rolling and then restored to the original shape.
This operation is considered a **restricted local operation**.

Parameters
"""""""""""
* input: Tensor type. the input tensor.
* shifts: int, a list or tuple of int. the number of places by which the elements of the tensor are shifted. If shifts is a tuple.
* dims: int, a list or tuple of int or None. Axis along which to roll.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor Support
""""""""""""""""""""""
* BM1688:  Data type can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X: Data type can be FLOAT32/FLOAT16/INT8/UINT8.



Sort Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arg
:::::::::::::::::

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
""""""""""""""""""""""
Returns two Tensors, the first Tensor represents indices, of type int32; and the second Tensor represents values, the type of which will be the same as the type of the input.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

topk
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def topk(input: Tensor,
                 axis: int,
                 k: int,
                 out_name: str = None):

Description
"""""""""""
Find top k numbers after sorted

Parameters
"""""""""""
* input: Tensor type, representing the input tensor.
* axis: Int type, representing axis used in sorting.
* k: Int type, representing the number of top values along axis.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns two Tensors: the first one represents the values, whose data type is the same as that of the input tensor while the second one represents the indices in input tensor after sorted along axis.

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sort
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def sort(input: Tensor,
                 axis: int = -1,
                 descending : bool = True,
                 out_name = None)

Description
"""""""""""
Sort input tensor along axis then return the sorted tensor and correspending indices.

Parameters
"""""""""""
* input: Tensor type, representing input.
* axis: Int type, representing the axis used in sorting. (Recently, only support axis == -1)
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns two Tensors: data type of the first is the same of that of input, and data type of the second is INT32.

Processor Support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


argsort
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def argsort(input: Tensor,
                    axis: int = -1,
                    descending : bool = True,
                    out_name : str = None)

Description
"""""""""""
Sort input tensor along axis then return the correspending indices of sorted tensor.

Parameters
"""""""""""
* input: Tensor type, representing input.
* axis: Int type, representing the axis used in sorting. (Recently, only support axis == -1)
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns one Tensor whose data type is INT32.

Processor Support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


sort_by_key (TODO)
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def sort_by_key(input: Tensor,
                        key: Tensor,
                        axis: int = -1,
                        descending : bool = True,
                        out_name = None)

Description
"""""""""""
Sort input tensor by key along axis then return the sorted tensor and correspending keys.

Parameters
"""""""""""
* input: Tensor type, representing input.
* key: Tensor type, representing key.
* axis: Int type, representing the axis used in sorting.
* descending: Bool type, representing whether it is sorted descending or not.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns two Tensors: data type of the first is the same of that of input, and data type of the second is is the same of that of key.

Processor Support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


Shape About Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

squeeze
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

reshape
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

shape_fetch
:::::::::::::::::

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
""""""""""""""""""""""
Returns a `Tensor` with the data type `INT32`.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

unsqueeze
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


Quant Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

requant_fp_to_int
:::::::::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

requant_fp
:::::::::::::::::::

The interface definition
"""""""""""

    .. code-block:: python

        def requant_fp(tensor_i: Tensor,
               scale: Union[float, List[float]],
               offset: Union[float, List[float]],
               out_dtype: str,
               out_name: str=None,
               round_mode: str='half_away_from_zero',
               first_round_mode: str='half_away_from_zero'):

Description of the function
"""""""""""
Quantizes the input tensor.

The calculation formula for this operation is:

    ::

        output = saturate(int(round(float(input) * scale + offset))),
        where saturate saturates to the output data type.


This operation is a **local operation**.

Explanation of parameters
"""""""""""
* tensor_i: Tensor type, representing the input tensor, with 3-5 dimensions.
* scale: List[float] or float, representing the quantization scale.
* offset: List[int] or int, representing the output offset.
* out_dtype: String type, representing the data type of the input tensor. The data type can be "int16"/"uint16"/"int8"/"uint8".
* out_name: String type or None, representing the name of the output tensor. When set to None, the name will be automatically generated internally.
* round_mode: String type, representing the rounding mode. Default is "half_away_from_zero". The round_mode can take values of "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, representing the rounding mode used for quantizing tensor_i previously. Default is "half_away_from_zero". The first_round_mode can take values of "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return Value
"""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
"""""""""""
* BM1688：Support input datatype: INT32/INT16/UINT16.
* BM1684X：Support input datatype: INT32/INT16/UINT16.

requant_int
:::::::::::::::::

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
""""""""""""""""""""""
Returns a tensor. The data type of this tensor is determined by out_dtype.

Processor support
""""""""""""""""""""""
* BM1684X
* BM1688

dequant_int_to_fp
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is specified by out_dtype.

Processor support
""""""""""""""""""""""
* BM1684X: Input data types can be INT16/UINT16/INT8/UINT8.


dequant_int
:::::::::::::::::

The interface definition
"""""""""""
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
""""""""""""""""""""""
Returns a Tensor. The data type of this Tensor is determined by out_dtype.

Processor support
""""""""""""""""""""""
* BM1684X


cast
:::::::::::::::::

The interface definition
"""""""""""

    .. code-block:: python

      def cast(tensor_i: Tensor,
         out_dtype: str = 'float32',
         out_name: str = None,
         round_mode: str = 'half_away_from_zero'):

Description of the function
"""""""""""
Converts the input tensor `tensor_i` to the specified data type `out_dtype`, and rounds the data according to the specified rounding mode `round_mode`.
Note that this operator cannot be used alone and must be used in conjunction with other operators.

Explanation of parameters
"""""""""""
* tensor_i: Tensor type, representing the input Tensor.
* out_dtype: str = 'float32', the data type of the output tensor, default is `float32`.
* out_name: str = None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.
* round_mode: str = 'half_away_from_zero', the rounding mode, default is `half_away_from_zero`. Possible values are “half_away_from_zero”, “half_to_even”, “towards_zero”, “down”, “up”. Note that this function does not support the rounding modes “half_up” and “half_down”.

Return value
"""""""""""
Returns a Tensor whose data type is determined by the input `out_dtype`.

Processor Support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/UINT8/INT8.

Up/Down Scaling Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

maxpool2d
:::::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
Performs Max Pooling on the input Tensor.The Max Pooling 2d operation can refer to the maxpool2d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""
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
"""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.


maxpool2d_with_mask
:::::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
Perform Max pooling on the input Tensor and output its mask index. Please refer to the pooling operations under various frameworks.
This operation belongs to **local operation**.

Explanation of parameters
"""""""""""
* input: Tensor type, indicating the input operation Tensor.
* kernel: List[int] or Tuple[int] type or None. If None is entered, global_pooling is used. If not None, the length of this parameter is required to be 2.
* pad: List[int] or Tuple[int] type or None. Indicates the padding size. If None is entered, the default value [0,0,0,0] is used. If not None, the length of this parameter is required to be 4.
* stride: List[int] or Tuple[int] type or None. Indicates the stride size. If None is entered, the default value [1,1] is used. If not None, the length of this parameter is required to be 2.
* ceil: bool type, indicating whether to round up when calculating the output shape.
* out_name: string type or None. Indicates the name of the output Tensor. If None, the name is automatically generated internally.
* mask_name: string type or None. Indicates the name of the output Mask. If None, the name is automatically generated internally.

Return value
"""""""""""
Returns two Tensors, one of which has the same data type as the input Tensor and the other returns a coordinate Tensor, which records the coordinates selected when using comparison operation pooling.

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32
* BM1684X: The input data type can be FLOAT32

upsample
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16/INT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16/INT8.

reduce
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

maxpool3d
:::::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
Performs Max Pooling on the input Tensor.The Max Pooling 3d operation can refer to the maxpool3d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""
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
"""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.

avgpool2d
:::::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
Performs Avg Pooling on the input Tensor.The Avg Pooling 2d operation can refer to the avgpool2d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""
* input: Tensor type, indicating the input operation Tensor.
* kernel: List[int] or Tuple[int] type or None. If None is entered, global_pooling is used. If not None, the length of this parameter is required to be 2.
* stride: List[int] or Tuple[int] type or None, indicating the step size. If None is entered, the default value [1,1] is used. If not None, the length of this parameter is required to be 2.
* pad: List[int] or Tuple[int] type or None, indicating the padding size. If None is entered, the default value [0,0,0,0] is used. If not None, the length of this parameter is required to be 4.
* ceil: bool type, indicating whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the scales of input and output respectively.
* zero_point: List[int] type or None, offset parameter. None is used to represent non-quantized calculation. If it is a List, the length is 2, which are the zero_points of input and output respectively.
* out_name: string type or None, indicating the name of the output Tensor. If it is None, the name will be automatically generated internally.
* count_include_pad: Bool type, indicating whether the pad value is included when calculating the average value. The default value is False.
* round_mode: String type, when the input and output Tensors are quantized, it indicates the second rounding mode. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, when the input and output Tensors are quantized, it indicates the first rounding mode. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.

avgpool3d
:::::::::::::::::::

The interface definition
"""""""""""

    .. code-block:: python

      def avgpool3d(input: Tensor,
                kernel: Union[List[int],int,Tuple[int, ...]] = None,
                stride: Union[List[int],int,Tuple[int, ...]] = None,
                pad:    Union[List[int],int,Tuple[int, ...]] = None,
                ceil_mode: bool = False,
                scale: List[float] = None,
                zero_point: List[int] = None,
                out_name: str = None,
                round_mode : str="half_away_from_zero",
                first_round_mode : str="half_away_from_zero"):
          #pass

Description of the function
"""""""""""
Performs Avg Pooling on the input Tensor.The Avg Pooling 3d operation can refer to the avgpool3d operator of each framework This operation is a  **local operation** 。

Explanation of parameters
"""""""""""
* tensor: Tensor type, representing the input tensor for the operation.
* kernel: List[int] or Tuple[int] or int or None, if None, global pooling is used. If not None and a single integer is provided, it indicates the same kernel size in three dimensions. If a List or Tuple is provided, its length must be 3.
* pad: List[int] or Tuple[int] or int or None, represents the padding size. If None, the default value [0,0,0,0,0,0] is used. If not None and a single integer is provided, it indicates the same padding size in three dimensions. If a List or Tuple is provided, its length must be 6.
* stride: List[int] or Tuple[int] or int or None, represents the stride size. If None, the default value [1,1,1] is used. If not None and a single integer is provided, it indicates the same stride size in three dimensions. If a List or Tuple is provided, its length must be 3.
* ceil_mode: bool type, indicates whether to round up when calculating the output shape.
* scale: List[float] type or None, quantization parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the scale for input and output respectively.
* zero_point: List[int] type or None, offset parameters. If None, non-quantized computation is performed. If a List is provided, its length must be 2, representing the zero point for input and output respectively.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.
* round_mode: string type, indicates the rounding mode for the second time when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".
* first_round_mode: String type, indicating the rounding mode for the first round when the input and output Tensors are quantized. The default value is 'half_away_from_zero'.The value range of round_mode is "half_away_from_zero", "half_to_even", "towards_zero", "down", "up".

Return value
"""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/UINT8/INT8.
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.






Normalization Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

batch_norm
:::::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
The batch_norm op first completes batch normalization of the input values, and then scales and shifts them.
The batch normalization operation can refer to the batch_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""

* input: * input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* mean: A Tensor type, representing the mean value of the input, shape is [c].
* variance: A Tensor type, representing the variance value of the input, shape is [c].
* gamma: A Tensor type or None, representing the scaling after batch normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after batch normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

layer_norm
:::::::::::::::::::

The interface definition
"""""""""""

    .. code-block:: python

      def layer_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int,
                     out_name: str = None):
          #pass


Description of the function
"""""""""""
The layer_norm op first completes layer normalization of the input values, and then scales and shifts them.
The layer normalization operation can refer to the layer_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* gamma: A Tensor type or None, representing the scaling after layer normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after layer normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* axis: int type, the first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.


Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

group_norm
:::::::::::::::::::

The interface definition
"""""""""""""""""""""""""

    .. code-block:: python

      def group_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     num_groups: int,
                     out_name: str = None):
          #pass


Description of the function
"""""""""""""""""""""""""""""
The group_norm op first completes group normalization of the input values, and then scales and shifts them.
The group normalization operation can refer to the group_norm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited, if x is only 1 dimension, c is 1, otherwise c is equal to the shape[1] of x.
* gamma: A Tensor type or None, representing the scaling after group normalization. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 1 Tensor.
* beta: A Tensor type or None, representing he translation after group normalization and scaling. If the value is not None, shape is required to be [c]. If None is used, shape[1] is equivalent to all 0 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* num_groups:int type, The number of groups of channels. It should be a divisor of the number of channels `C`.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.


rms_norm
:::::::::::::::::::

The interface definition
"""""""""""""""""""""""""

    .. code-block:: python

      def rms_norm(input: Tensor,
                     gamma: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int = -1,
                     out_name: str = None):
          #pass



Description of the function
"""""""""""""""""""""""""""""
The rms_norm op first completes RMS normalization of the input values, and then scales them.
The RMS normalization operation can refer to the RMSNorm operator of each framework.

This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""

* input: A Tensor type, representing the input Tensor.The dimension of input is not limited.
* gamma: A Tensor type or None, representing the scaling after RMS normalization. If the value is not None, shape is required to be equal with the last dimension of the input. If None is used, shape is equivalent to all 1 Tensor.
* epsilon: FLOAT type, The epsilon value to use to avoid division by zero.
* axis: int type, the first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

normalize
:::::::::::::::::

Definition
"""""""""""
    .. code-block:: python

      def normalize(input: Tensor,
                        p: float = 2.0,
                        axes: Union[List[int], int] = 1,
                        eps : float = 1e-12,
                        out_name: str = None):

Description
"""""""""""
Perfrom :math:`L_p` normalization over specified dimension of input tensor.
For a tensor input of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`,
each :math:`n_{dim}`-element vector :math:`v` along dimension :attr:`axes`  is transformed as:
.. math::
v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

With the default arguments, it uses the Euclidean norm over vectors along dimension (1) for normalization.

This operation belongs to **local operations**.

Parameters
"""""""""""
* input: Tensor type, representing the input Tensor.The dimension of input is not limited. Support data type included: float32, float16.
* p: float type, representing the exponent vaue in the norm operation. Default to 2.0 .
* axes: Union[list[int], int] type, representing the dimension need to normalized. Default to 1. If axes is list, all the values in the list must be continuous. Caution: axes = [0, -1] is not continuous.
* eps: float type, the epsilon value to use to avoid division by zero. Default to 1e-12.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
"""""""""""""
Returns the Tensor type with the same data type as the input Tensor., representing the normalized output.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16.
* BM1684X: The input data type can be FLOAT32/FLOAT16.

Vision Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nms
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def nms(boxes: Tensor,
                scores: Tensor,
                format: str = 'PYTORCH',
                max_box_num_per_class: int = 1,
                out_name: str = None)

Description
"""""""""""
Perform non-maximum-suppression upon input tensor.

Parameters
"""""""""""
* boxes: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of box, the third dimension is 4 coordinates of boxes.
* scores: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of classes, the third dimension is number of boxes.
* format: String type, where 'TENSORFLOW' representing Tensorflow format [y1, x1, y2, x2] and 'PYTORCH'表示representing Pytorch format [x_center, y_center, width, height]. The default value is 'PYTORCH'.
* max_box_num_per_class: Int type, representing max number of boxes per class. It must be greater than 0. The default value is 1.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns one Tensor, which is the selected indices from the boxes tensor of 2 dimensions:[num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16(TODO)/INT8/UINT8.
* BM1684X: The input data type can be FLOAT32/FLOAT16(TODO)/INT8/UINT8.


interpolate
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def interpolate(input: Tensor,
                        scale_h: float,
                        scale_w: float,
                        method: str = 'nearest',
                        coord_mode: str = "pytorch_half_pixel",
                        out_name: str = None)

Description
"""""""""""
Perform interpolation upon input tensor.

Parameters
"""""""""""
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
"""""""""""
Returns a Tensor representing the interpolated result. The data type is the same as the input type, and the shape is adjusted based on the scaling factors.

Processor support
"""""""""""
* BM1688: Supports input data types FLOAT32/FLOAT16/INT8.
* BM1684X: Supports input data types FLOAT32/FLOAT16/INT8.



yuv2rgb
:::::::::::::::::

The interface definition
"""""""""""

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
"""""""""""
Transfer input tensor from yuv to rgb. Require tensor shape=[n,h*3/2,w], n represents `batch`, h represents `pixels height`, w represents `pixels width`.

Explanation of parameters
"""""""""""
* inputs: Tensor type, representing the input yuv tensor。Its dims must be 3, 1st dim represents `batch`, 2nd dim represents `pixels height`, 3rd dim represents `pixels width`.
* src_format: Int type, representing the input format. `FORMAT_MAPPING_YUV420P_YU12`=0, `FORMAT_MAPPING_YUV420P_YV12`=1, `FORMAT_MAPPING_NV12`=2, `FORMAT_MAPPING_NV21`=3.
* dst_format: Int type, representing the output format. `FORMAT_MAPPING_RGB`=4, `FORMAT_MAPPING_BGR`=5.
* ImageOutFormatAttr: string type, representing the output dtype, currently only support `UINT8`.
* formula_mode: string type, representing the formula to transfer from yuv to rgb, currently support `_601_limited`, `_601_full`.
* round_mode: string type, currently support `HalfAwayFromZero`, `HalfToEven`.
* out_name: string type, representing the name of output tensor, default= `None`.

Return value
"""""""""""
One rgb tensor will be output, with shape=[n,3,h,w], where n represents `batch`, h represents `pixels height`, w represents `pixels width`.

Processor support
"""""""""""
* BM1684X: The input data type must be UINT8/INT8. Output data type is INT8.
* BM1688: The input data type must be UINT8/INT8. Output data type is INT8.


Select Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nonzero
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nonzero(tensor_i:Tensor,
                  dtype = 'int32',
                  out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Extract the corresponding location information when input Tensor data is true.
This operation is considered a **global operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i: Tensor type, representing the input tensor for the operation.
* dtype: The data type of the output tensor, with a default value of "int32."
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor with the same data type as the input Tensor.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


lut
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def lut(input: Tensor,
                table: Tensor,
                out_name: str = None):
        #pass

Description
"""""""""""
Use look-up table to transform values of input tensor.

Parameters
"""""""""""
* input: Tensor type, representing the input.
* table: Tensor type, representing the look-up table.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Returns
"""""""""""
Returns one Tensor, whose data type is the same as that of the `table` tensor.

Processor support
"""""""""""
* BM1688:  The data type of `input` can be INT8/UINT8. The data type of `table` an be INT8/UINT8.
* BM1684X: The data type of `input` can be INT8/UINT8. The data type of `table` an be INT8/UINT8.

select
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def select(lhs: Tensor,
                   rhs: Tensor,
                   tbrn: Tensor,
                   fbrn: Tensor,
                   type: str,
                   out_name = None):
        #pass

Description
"""""""""""
Select by the comparison result of `lhs` and `rhs`. If condition is True, select `tbrn`, otherwise select `fbrn`.

Parameters
"""""""""""
* lhs: Tensor type, representing the left-hand-side.
* rhs: Tensor type, representing the right-hand-side.
* tbrn: Tensor type, representing the true branch.
* fbrn: Tensor type, representing the false branch.
* type: String type, representing the comparison operator. Optional values are "Greater"/"Less"/"GreaterOrEqual"/"LessOrEqual"/"Equal"/"NotEqual".
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Constraint: The shape and data type of `lhs` and `rhs` should be the same. The shape and data type of `tbrn` and `fbrn` should be the same.

Returns
"""""""""""
Returns a Tensor whose data type is the same that of `tbrn`.

Processor Support
"""""""""""
* BM1688:  Data type of `lhs`/ `rhs`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X:  Data type of `lhs`/ `rhs`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.

cond_select
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def cond_select(cond: Tensor,
                        tbrn: Union[Tensor, Scalar],
                        fbrn: Union[Tensor, Scalar],
                        out_name:str = None):
        #pass

Description
"""""""""""
Select by condition representing by `cond`. If condition is True, select `tbrn`, otherwise select `fbrn`.

Parameters
"""""""""""
* cond: Tensor type, representing condition.
* tbrn: Tensor type or Scalar type, representing true branch.
* fbrn: Tensor type or Scalar type, representing false branch.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Constraint: If `tbrn` and `fbrn` are all Tensors, then the shape and data type of `tbrn` and `fbrn` should be the same.

Returns
"""""""""""
Returns a Tensor whose data type is the same that of `tbrn`.

Processor Support
"""""""""""
* BM1688:  Data type of `cond`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.
* BM1684X:  Data type of `cond`/ `tbrn`/ `fbrn` can be FLOAT32/FLOAT16/INT8/UINT8.

bmodel_inference_combine
:::::::::::::::::

Definition
"""""""""""

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
            is_soc: bool = False,  # soc mode ONLY support {reference_data_fn=xxx.npz, dump_file=True}
            tmp_path: str = "/tmp",  # should config when is_soc=True
            tools_path: str = "/soc_infer",  # should config when is_soc=True
            hostname: str = None,  # should config when is_soc=True
            port: int = None,  # should config when is_soc=True
            username: str = None,  # should config when is_soc=True
            password: str = None,  # should config when is_soc=True
        ):

Description
"""""""""""
Dump tensors layer by layer according to the bmodel, which help to verify the correctness of bmodel.

Parameters
"""""""""""
* bmodel_file: String type, representing the abs path of bmodel.
* final_mlir_fn: String type, representing the abs path of final.mlir.
* input_data_fn: String type or Dict type, representing the input data, supporting Dict/.dat/.npz.
* tensor_loc_file: String type, representing the abs path of tensor_location.json.
* reference_data_fn: String type, representing the abs path of .mlir/.npz with `module.state = "TPU_LOWERED"`. Used to restore the shape during bmodel infer.
* dump_file: Bool type, representing whether save results as file.
* save_path: String type, representing the abs path of saving results on host.
* out_fixed: Bool type, representing whether to get results in fixed number.
* is_soc: Bool type, representing whether to use in soc mode.
* tmp_path: String type, representing the abs path of tmp files on device in soc mode.
* tools_path: String type, representing the dir of soc_infer tools on device in soc mode.
* hostname: String type, representing the ip address of device in soc mode.
* port: Int type, representing the port of device in soc mode.
* username: String type, representing the username of device in soc mode.
* password: String type, representing the password of device in soc mode.

Attention:

* When the funciton is called in cmodel/pcie mode, functions `use_cmodel/use_chip` from `/tpu-mlir/envsetup.sh` is required.
* When the funciton is called in soc mode, use `use_chip` and `reference_data_fn` must be .npz.

Returns
"""""""""""
* cmodel/pcie mode: if `dump_file=True`, then bmodel_infer_xxx.npz will be generated in `save_path`, otherwise return python dict.
* soc mode: soc_infer_xxx.npz will be generated in `save_path`.

Processor Support
"""""""""""
* BM1688:  only cmodel mode.
* BM1684X: cmodel/pcie/soc mode.

scatter
:::::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

      def scatter(input: Tensor,
            index: Tensor,
            updates: Tensor,
            axis: int = 0,
            out_name: str = None):
        #pass

Description
"""""""""""
Based on the specified indices, write the input data to specific positions in the target Tensor. This operation allows the elements of the input Tensor to be scattered to the specified positions in the output Tensor. Refer to the ScatterElements operation in various frameworks for more details.
This operation belongs to **local operation**。

Parameters
"""""""""""
* input: Tensor type, represents the input operation Tensor, i.e., the target Tensor that needs to be updated.
* index: Tensor type, represents the index Tensor that specifies the update positions.
* updates: Tensor type, represents the values to be written into the target Tensor.
* axis: int type, represents the axis along which to update.
* out_name: string type or None, represents the name of the output Tensor. If None, a name will be automatically generated internally.


Returns
"""""""""""
Returns a new Tensor with updates applied at the specified positions, while other positions retain the original values from the input Tensor.



Processor Support
"""""""""""
* BM1684X: The input data type can be FLOAT32/UINT8/INT8.
* BM1688: The input data type can be FLOAT32/UINT8/INT8.



Preprocess Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mean_std_scale
:::::::::::::::::

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
""""""""""""""""""""""
Returns a Tensor with the type of odtype.

Processor support
""""""""""""""""""""""
* BM1684X: The input data type can be FLOAT32/UINT8/INT8, the output data type can be INT8/FLOAT16.
