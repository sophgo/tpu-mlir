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

* The device parameter is of type string and can take values from the range "cpu" | "bm1684" | "bm1684x".

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
            dynamic=False):
            #pass


Description of the function
:::::::::::::::::::::::::::::

The function for comipling TpuLang model to bmodel.

Explanation of parameters
:::::::::::::::::::::::::::::

* name: A string. Model name.
* inputs: List of Tensors, representing all input Tensors for compiling the network.
* outputs: List of Tensors, representing all output Tensors for compiling the network.
* cmp: A boolean. True indicates result verification is needed, False indicates compilation only.
* refs: List of Tensors, representing all Tensors requiring verification in the compiled network.
* mode: A string. Indicates the type of model, supporting "f32" and "int8".
* dynamic: A boolean. Whether to do dynamic compilation.

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

      def add(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise addition operation between tensors. :math:`tensor\_o = tensor\_i0 + tensor\_i1`。
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type.
When the input is 'float'/'float16', the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


sub
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sub(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Element-wise subtraction operation between tensors. :math:`tensor\_o = tensor\_i0 - tensor\_i1`。
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string type or None, representing the data type of the output tensor. If None, it is consistent with the input tensors' dtype. The optional parameters are 'float'/'float16'/'int8'/'int16'/'int32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is 'float'/'float16',
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

      def mul(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Element-wise multiplication operation between tensors. :math:`tensor\_o = tensor\_i0 * tensor\_i1`。
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type.
When the input is 'float'/'float16', the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


div
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def div(tensor_i0, tensor_i1, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""

Element-wise division operation between tensors. :math:`tensor\_o = tensor\_i0 / tensor\_i1`。
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

      def max(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise maximum operation between tensors. :math:`tensor\_o = max(tensor\_i0, tensor\_i1)`。
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type. When the input is 'float'/'float16',
the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
"""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.


min
:::::::::::::::::

The interface definition
""""""""""""""""""""""""""

    .. code-block:: python

      def min(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""
Element-wise minimum operation between tensors. :math:`tensor\_o = min(tensor\_i0, tensor\_i1)`。
This operation supports broadcasting.
This operation belongs to **local operations**.

Explanation of parameters
""""""""""""""""""""""""""
* tensor_i0: Tensor type or Scalar, int, float. It represents the left operand Tensor or Scalar for the input.
* tensor_i1: Tensor type or Scalar, int, float. It represents the right operand Tensor or Scalar for the input. At least one of tensor_i0 and tensor_i1 must be a Tensor.
* out_dtype: A string or None, representing the data type of the output Tensor. If set to None, it will be consistent with the input data type. Optional values include 'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""
Returns a Tensor, and the data type of this Tensor is specified by out_dtype or is consistent with the input data type.
When the input is 'float'/'float16', the output data type must be the same as the input. When the input is 'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32', the output can be any integer type.

Processor support
"""""""""""""""""""""
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
Element-wise greater than comparison operation between tensors. :math:`tensor\_o = tensor\_i0 > tensor\_i1 ? 1 : 0`。
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
Element-wise less than comparison operation between tensors. :math:`tensor\_o = tensor\_i0 < tensor\_i1 ? 1 : 0`。
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

Element-wise greater than or equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 >= tensor\_i1 ? 1 : 0`。
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

Element-wise less than or equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 <= tensor\_i1 ? 1 : 0`。
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

Element-wise equality comparison operation between tensors. :math:`tensor\_o = tensor\_i0 == tensor\_i1 ? 1 : 0`。
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
Element-wise not equal to comparison operation between tensors. :math:`tensor\_o = tensor\_i0 != tensor\_i1 ? 1 : 0`。
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
Element-wise greater-than comparison operation between tensors and scalars. :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`。
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
Element-wise less-than comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`。
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
Element-wise greater-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`。
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
Element-wise less-than-or-equal-to comparison between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`。
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
The element-wise equality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`。
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
The element-wise inequality comparison operation between a tensor and a scalar. :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`。
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

Element-wise Compare Operator
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
The ReLU activation function, implemented on an element-wise basis. :math:`y = max(0, x)`。
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
The leaky ReLU activation function, implemented on an element-wise basis. :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`。
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
The abs absolute value activation function, implemented on an element-wise basis. :math:`y = \left | x \right |`。
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
The ceil rounding up activation function, implemented on an element-wise basis. :math:`y = \left \lfloor x \right \rfloor`。
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
The floor rounding down activation function, implemented on an element-wise basis. :math:`y = \left \lceil x \right \rceil`。
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
The round activation function, which rounds to the nearest integer using the round half up (four-way tie-breaking) method, implemented on an element-wise basis. :math:`y = round(x)`。
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
The sin sine activation function, implemented on an element-wise basis. :math:`y = sin(x)`。
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
The cos cosine activation function, implemented on an element-wise basis. :math:`y = cos(x)`。
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
The exp exponential activation function, implemented on an element-wise basis. :math:`y = e^{x}`。
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
The tanh hyperbolic tangent activation function, implemented on an element-wise basis. :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`。
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
The sigmoid activation function, implemented on an element-wise basis. :math:`y = 1 / (1 + e^{-x})`。
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
The ELU (Exponential Linear Unit) activation function, implemented on an element-wise basis. :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`。
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
The sqrt square root activation function, implemented on an element-wise basis. :math:`y = \sqrt{x}`。
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
is implemented on an element-wise basis. :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`。
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
The tan tangent activation function, implemented on an element-wise basis. :math:`y = tan(x)`。
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
to the exponentials of the input numbers. :math:`tensor\_o = exp(tensor\_i)/sum(exp(tensor\_i),axis)`。
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
The Mish activation function, implemented on an element-wise basis.:math:`y = x * tanh(ln(1 + e^{x}))`。
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
The h-swish activation function, implemented on an element-wise basis. :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`。
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
The arccosine (inverse cosine) activation function, implemented on an element-wise basis. :math:`y = arccos(x)`。
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
The arctanh (inverse hyperbolic tangent) activation function, implemented on an element-wise basis. :math:`y = arctanh(x)=\frac{1}{2}ln(\frac{1+x}{1-x})`。
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
The sinh (hyperbolic sine) activation function, implemented on an element-wise basis. :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`。
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
The cosh (hyperbolic cosine) activation function, implemented on an element-wise basis. :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`。
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
The sign activation function, implemented on an element-wise basis. :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`。
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
The GELU (Gaussian Error Linear Unit) activation function, implemented on an element-wise basis. :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`。
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
The hsigmoid (hard sigmoid) activation function, implemented on an element-wise basis. :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`。
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

      def permute(tensor, order=(), out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Permute the dimensions of the input Tensor according to the permutation parameter.

For example: Given an input shape of (6, 7, 8, 9) and a permutation parameter `order` of (1, 3, 2, 0), the output shape will be (7, 9, 8, 6).
This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor:Tensor type,表示输入操作Tensor。
* order:List[int]或Tuple[int]型,表示置换参数。要求order长度和tensor维度一致。
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

      def tile(tensor_i, reps, out_name=None):
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


concat
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def concat(tensors, axis=0, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Concatenate multiple tensors along the specified axis.

This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensors: A `List[Tensor]` type, containing multiple tensors. All tensors must have the same data type and the same number of shape dimensions.Except for the dimension to be concatenated, the values of the other dimensions should be equal.
* axis: An `int` type, indicating the axis along which the concatenation operation will be performed.
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

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

      def split(tensor, axis=0, num=1, size=(), out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Split the input tensor into multiple tensors along the specified axis. If `size` is not empty, the dimensions of the split tensors are determined by `size`.
 Otherwise, the tensor is split into `num` equal parts along the specified axis, assuming the tensor's size along that axis is divisible by `num`.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A `Tensor` type, indicating the tensor that is to be split.
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

      def pad(tensor, padding=None, value=None, method='constant', out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Padding the input tensor.

This operation belongs to **local operations**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* tensor: A `Tensor` type, indicating the tensor that is to be padded.
* padding: A `List[int]`, `Tuple[int]`, or `None`. If `padding` is `None`, a zero-filled list of length `2 * len(tensor.shape)` is used. For example, the padding of a hw 2D Tensor is [h_top, w_left, h_bottom, w_right]
* value: A `Scalar`, `Variable` type, or `None`, representing the value to be filled. The data type is consistent with that of the tensor.
* method:string type,表示填充方法,可选方法"constant","reflect","symmetric","edge"。
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

      def repeat(tensor_i, reps, out_name=None):
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



Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arg
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arg(tensor, method='max', axis=0, keep_dims=True, out_name=None):
          #pass

Description of the function
"""""""""""""""""""""""""""""""""
Translate: For the input tensor, find the maximum or minimum values along the specified axis, output the corresponding indices, and set the dimension of that axis to 1.
This operation is considered a **restricted local operation**.

Explanation of parameters
"""""""""""""""""""""""""""""""""
* Tensor: Tensor type, representing the Tensor to be operated on.
* method: A string type, indicating the method of operation, options include 'max' and 'min'.
* axis: An integer, indicating the specified axis.
* keep_dims: A boolean, indicating whether to keep the specified axis after the operation. The default value is True, which means to keep it (in this case, the length of that axis is 1).
* out_name: A string or None, representing the name of the output Tensor. If set to None, the system will automatically generate a name internally.

Return value
""""""""""""""""""""""
Returns a Tensor.

The data type of the tensor can be FLOAT32/INT8/UINT8. The data type of tensor_o can be INT32/FLOAT32.
When the data type of the tensor is INT8/UINT8, tensor_o can only be INT32.

Processor support
""""""""""""""""""""""
* BM1688: The input data type can be FLOAT32.
* BM1684X: The input data type can be FLOAT32.

Shape About Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

squeeze
:::::::::::::::::

The interface definition
"""""""""""""""""""""""""""""""""

    .. code-block:: python

      def squeeze(tensor_i, axis, out_name=None):
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

      def reshape(tensor, new_shape, out_name=None):
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

      def shape_fetch(tensor_i, begin_axis=0, end_axis=1, step=1, out_name=None):
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


nms
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def nms(boxes: Tensor,
                scores: Tensor,
                format: str = 'PYTORCH',
                max_box_num_per_class: int = 0,
                out_name: str = None)

Description
"""""""""""
Perform non-maximum-suppression upon input tensor.

Parameters
"""""""""""
* boxes: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of box, the third dimension is 4 coordinates of boxes.
* scores: Tensor type, representing a tensor of 3 dimensions, where the first dimension is number of batch, the second dimension is number of classes, the third dimension is number of boxes.
* format: String type, where 'TENSORFLOW' representing Tensorflow format [y1, x1, y2, x2] and 'PYTORCH'表示representing Pytorch format [x_center, y_center, width, height].
* max_box_num_per_class: Int type, representing max number of boxes per class. The default value is 0.
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
* input: Tensor type, representing the input Tensor.
* scale_h: Float type, representing the resize scale along h-axis.
* scale_w: Float type, representing the resize scale along w-axis.
* method: String type, representing the interpolation method. Optional values are "nearest" or "linear".
* coord_mode: string type, representing the method used in inverse map of coordinates. Optional values are "align_corners", "pytorch_half_pixel", "half_pixel" or "asymmetric".
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
Returns one Tensor, whose data type is the same as that of the input tensor.

Processor support
"""""""""""
* BM1688: The input data type can be FLOAT32/FLOAT16(TODO).
* BM1684X: The input data type can be FLOAT32/FLOAT16(TODO).


lut
:::::::::::::::::

Definition
"""""""""""

    .. code-block:: python

        def lut(input: Tensor,
                table: Tensor,
                out_name: str = None)

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
