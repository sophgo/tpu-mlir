附录02：TpuLang的基本元素
==============================


本章将介绍TpuLang程序的基本元素：Tensor、Scalar、Control Functions和Operator。

.. _tensor:

张量(Tensor)
---------------

TpuLang中Tensor的name, data，
data type, tensor type均最多只能声明或者设置1次。

一般情况下推荐创建Tensor不指定Name，以免因为Name相同导致问题。
只有在必须指定Name时，才需要在创建Tensor时指定Name。

对于作为Operator输出的Tensor，可以不指定shape，因为Operator会自行推导。
即使指定了shape，若Tensor是Operator的输出，则同样由Operator自行推导并修改。

TpuLang中Tensor的定义如下：

   .. code-block:: python

      class Tensor:

         def __init__(self,
                     shape: list = [],
                     name: str = None,
                     ttype="neuron",
                     data=None,
                     dtype: str = "float32")
               #pass

如上所示，TpuLang中Tensor有5个参数。

* shape：Tensor的形状，List[int]，对于Operator输出的Tensor，可以不指定shape，默认值为[]。
* Name：Tensor的名称，string或None，该值推荐使用默认值None以免因为Name相同导致问题；
* ttype：Tensor的类型，可以是"neuron"或"coeff"或None，初始值为"neuron"；
* data：Tensor的输入数据，为默认值None时则会根据shape产生全0数据，否则应该是一个ndarray；
* dtype：Tensor的数据类型，默认值为"float32"，否则取值范围为"float32", "float16", "int32", "uint32", "int16", "uint16", "int8", "uint8"；


声明Tensor的示例：

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      #weight
      weight = tpul.Tensor(dtype='float32', shape=[3,4], data=np.random.uniform(0,1,shape).astype('float32'), ttype="coeff")

.. _scalar:

标量(Scalar)
---------------

定义一个标量Scalar。Scalar是一个常量，在声明时指定，且不能修改。

   .. code-block:: python

      class Scalar:

            def __init__(self, value, dtype=None):
                #pass

Scalar构造函数有两个参数，

* value：Variable型，即int/float型，无默认值，必须指定；
* dtype：Scalar的数据类型，为默认值None等同于"float32"，否则取值范围为"float32", "float16", "int32", "uint32", "int16", "uint16", "int8", "uint8"；

使用实例：

   .. code-block:: python

      pad_val = tpul.Scalar(1.0)
      pad = tpul.pad(input, value=pad_val)

Control Functions
--------------------

控制函数（control functions）主要包括控制TpuLang使用时的初始化、启动编译生成目标文件等。

控制函数常用于TpuLang程序的Tensor和Operator之前和之后。
比如在写Tensor和Operator之前，可能需要做初始化。
在完成Tensor和Operator编写之后，可能需要启动编译和反初始化。

.. _init:

初始化函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

初始化Function，在一个程序中构建网络之前使用。

初始化函数接口如下所示，选择处理器型号。

    .. code-block:: python

      def init(device):
          #pass

* device：string类型。取值范围"cpu"\|"bm1684x"。

.. _compile:
compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

接口定义
:::::::::::::::::::::::

    .. code-block:: python

        def compile(name: str,
            inputs: List[Tensor],
            outputs: List[Tensor],
            cmp=True,
            has_custom=False,
            refs=None):
            #pass


功能描述
:::::::::::::::::::::::

该接口在需要进行结果比对时，会从refs中依次取出Tensor，每个Tensor中，top层的计算结果和refs的计算结果进行比对。
如果编译过程中没有错误且所有Tensor的结果比对通过，则会产生npz和mlir。

参数说明
:::::::::::::::::::::::

* name：string类型。模型名称。
* inputs：List[Tensor]，表示编译网络的所有输入Tensor；
* outputs：List[Tensor]，表示编译网络的所有输出Tensor；
* refs：List[Tensor]，表示编译网络的所有需要比对验证的Tensor；
* cmp：bool类型。True表示需要结果比对，False表示仅编译；
* has_custom：bool型，即模型中是否包含自定义算子。值为True，则不对模型进行推理。

.. _deinit:

反初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在网络构建之后，需要进行反初始化结束。只有在反初始化后，之前所启动的
Compile生成TPU可执行目标执行才会存盘到所指定的输出目录中。

    .. code-block:: python

       def deinit():
          #pass

.. _operator:

Operator
---------------

为了TpuLang编程时可以考虑到获取好的性能，下面会将Operator分成本地操作（Local Operator）、受限本地操作（Limited Local Operator）
和全局操作（Global Operator）。

* 本地操作：在启动编译时，可以与其它的本地操作进行合并优化，使得操作之间的数据只存在于TPU的本地存储中。
* 受限本地操作：在一定条件下才能作为本地操作与其它本地操作进行合并优化。
* 全局操作：不能与其它操作进行合并优化，操作的输入输出数据都需要放到TPU的全局存储中。

以下操作中，很多属于按元素计算(Element-wise)的操作，要求输入输出Tensor的shape具备相同数量的维度。

当操作的输入Tensor是2个时，分为支持shape广播和不支持shape广播两种。
支持shape广播表示tensor_i0（输入0）和tensor_i1（输入1）的同一维度的shape值可以不同，此时其中一个tensor的shape值必须是1，数据将被广播扩展到另一个tensor对应的shape值。
不支持shape广播则要求tensor_i0（输入0）和tensor_i1（输入1）的shape值一致。

NN/Matrix Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conv
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维卷积运算。可参考各框架下的二维卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* kernel：目前废弃该参数，不使用；
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为1或2。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为1或2。
* groups：int型，表示卷积层的组数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


conv_v2
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) Conv (weight - kzp) + bias[c];

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为4。List中顺序为[上， 下， 左， 右]
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。
* BM1684X：输入数据类型可以是INT8/UINT8。


deconv
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维反卷积运算。可参考各框架下的二维反卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* kernel：目前废弃该参数，不使用；
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为1或2。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* output_padding：List[int]，表示输出的填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为1或2。
* output_padding：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


deconv_v2
:::::::::::::::::

接口定义
"""""""""""

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


功能描述
"""""""""""
二维转置卷积定点运算。可参考各框架下的二维转置卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) DeConv (weight - kzp) + bias[c];

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[ic, oc, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为4。List中顺序为[上， 下， 左， 右]
* output_padding：List[int]，表示输出的填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise deconv
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。
* BM1684X：输入数据类型可以是INT8/UINT8。


conv3d
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
三维卷积运算。可参考各框架下的三维卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，5维NCDHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCDHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1, 1]或[oc]，oc表示输出Channel数。
* kernel：目前废弃该参数，不使用；
* dilation：List[int]，表示空洞大小，取None则表示[1,1,1]，不为None时要求长度为1或3。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0,0,0]，不为None时要求长度为1或3或6。
* stride：List[int]，表示步长大小，取None则表示[1,1,1]，不为None时要求长度为1或3。
* groups：int型，表示卷积层的组数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


conv3d_v2
:::::::::::::::::

接口定义
"""""""""""

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


功能描述
"""""""""""
三维卷积定点运算。可参考各框架下的三维卷积定义。

::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) Conv3d (weight - kzp) + bias[c];

其中Conv3d表示3D卷积计算。

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，5维NCTHW格式。
* weight：Tensor类型，表示卷积核Tensor，5维[oc, ic, kt, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kt是kernel_t，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1, 1]。
* dilation：List[int]，表示空洞大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[dilation_t, dilation_h, dilation_w]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0,0,0]，不为None时要求长度为4。List中顺序为[前， 后， 上， 下， 左， 右]
* stride：List[int]，表示步长大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[stride_t, stride_h, stride_w]
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv3d
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。
* BM1684X：输入数据类型可以是FINT8/UINT8。

matrix_mul
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def matrix_mul(lhs, rhs, bias=None, left_zp=None, right_zp=None, \
                     out_dtype=None, out_name=None):
          #pass

功能描述
"""""""""""
矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* lhs：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* rhs：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* left_zp：List[int]型或int型，表示lhs的偏移。取None则表示0，取List时要求长度为k。
  该参数仅在lhs的dtype为‘int8/uint8’时有用。暂时只支持0.
* right_zp：List[int]型或int型，表示rhs的偏移。取None则表示0，取List时要求长度为k。
  该参数仅在rhs的dtype为‘int8/uint8’时有用。
* out_dtype：string类型或None，表示输入Tensor的类型，取None则与lhs的dtype一致。
  当lhs的dtype为‘int8/uint8’时，out_dtype取值范围：int32/uint32。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求lhr.shape[-1] == rhs.shape[-2]，lhr.shape[:-2]和rhs.shape[:-2]需要满足广播规则。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Base Element-wise Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def add(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素加法运算。 :math:`tensor\_o = tensor\_i0 + tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。当输入为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'时，输出可以是任意整型类型。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


sub
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sub(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素减法运算。 :math:`tensor\_o = tensor\_i0 - tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'int16'/'int32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。当输入为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'时，输出数据类型为'int8'/'int16'/'int32'。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


mul
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def mul(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素乘法运算。 :math:`tensor\_o = tensor\_i0 * tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。当输入为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'时，输出可以是任意整型类型。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


div
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def div(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素除法运算。 :math:`tensor\_o = tensor\_i0 / tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


max
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def max(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素取最大值。 :math:`tensor\_o = max(tensor\_i0, tensor\_i1)`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。当输入为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'时，输出可以是任意整型类型。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


min
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def min(tensor_i0, tensor_i1, out_dtype = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素取最小值。 :math:`tensor\_o = min(tensor\_i0, tensor\_i1)`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。当输入为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'时，输出可以是任意整型类型。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


clamp
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def clamp(tensor_i, min, max, out_name = None):
          #pass

功能描述
"""""""""""
将输入Tensor中所有元素的值都限定在设置的最大最小值范围内，大于最大值则截断为最大值，小于最大值则截断为最小值。
要求所有输入Tensor及Scalar的dtype一致。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor。
* min：Scalar类型，表示阶段的下限。
* max：Scalar类型，表示阶段的上限。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gt(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素大于比较运算。 :math:`tensor\_o = tensor\_i0 > tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

lt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def lt(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素小于比较运算。 :math:`tensor\_o = tensor\_i0 < tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

ge
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ge(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素大于等于比较运算。 :math:`tensor\_o = tensor\_i0 >= tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

le
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def le(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素小于等于比较运算。 :math:`tensor\_o = tensor\_i0 <= tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

eq
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def eq(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素等于比较运算。 :math:`tensor\_o = tensor\_i0 == tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

ne
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ne(tensor_i0, tensor_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和张量的按元素不等于比较运算。 :math:`tensor\_o = tensor\_i0 != tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

gts
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gts(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素大于比较运算。 :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

lts
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def lts(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素小于比较运算。 :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

ges
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ges(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素大于等于比较运算。 :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

les
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def les(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素小于等于比较运算。 :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

eqs
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def eqs(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素等于比较运算。 :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

nes
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def nes(tensor_i0, scalar_i1, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素不等于比较运算。 :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

relu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def relu(tensor, out_name=None):
          #pass

功能描述
"""""""""""
relu激活函数，逐元素实现功能 :math:`y = max(0, x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


leaky_relu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def leaky_relu(tensor, negative_slope=0.01, out_name=None):
          #pass

功能描述
"""""""""""
leaky_relu激活函数，逐元素实现功能 :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* negative_slope：FLOAT类型，表示输入的负斜率。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

abs
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def abs(tensor, out_name=None):
          #pass

功能描述
"""""""""""
abs绝对值激活函数，逐元素实现功能 :math:`y = \left | x \right |`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

ceil
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ceil(tensor, out_name=None):
          #pass

功能描述
"""""""""""
ceil向上取整激活函数，逐元素实现功能 :math:`y = \left \lfloor x \right \rfloor`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

floor
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def floor(tensor, out_name=None):
          #pass

功能描述
"""""""""""
floor向下取整激活函数，逐元素实现功能 :math:`y = \left \lceil x \right \rceil`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

round
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def round(tensor, out_name=None):
          #pass

功能描述
"""""""""""
round四舍五入整激活函数，逐元素实现功能 :math:`y = round(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


sin
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sin(tensor, out_name=None):
          #pass

功能描述
"""""""""""
sin正弦激活函数，逐元素实现功能 :math:`y = sin(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


cos
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def cos(tensor, out_name=None):
          #pass

功能描述
"""""""""""
cos余弦激活函数，逐元素实现功能 :math:`y = cos(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

exp
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def exp(tensor, out_name=None):
          #pass

功能描述
"""""""""""
exp指数激活函数，逐元素实现功能 :math:`y = e^{x}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

tanh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def tanh(tensor, out_name=None):
          #pass

功能描述
"""""""""""
tanh双曲正切激活函数，逐元素实现功能 :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

sigmoid
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sigmoid(tensor, out_name=None):
          #pass

功能描述
"""""""""""
sigmoid激活函数，逐元素实现功能 :math:`y = 1 / (1 + e^{-x})`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

elu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def elu(tensor, out_name=None):
          #pass

功能描述
"""""""""""
elu激活函数，逐元素实现功能 :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

sqrt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sqrt(tensor, out_name=None):
          #pass

功能描述
"""""""""""
sqrt平方根激活函数，逐元素实现功能 :math:`y = \sqrt{x}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

erf
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def erf(tensor, out_name=None):
          #pass

功能描述
"""""""""""
erf激活函数，对于输入输出Tensor对应位置的元素x和y，逐元素实现功能 :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

tan
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def tan(tensor, out_name=None):
          #pass

功能描述
"""""""""""
tan正切激活函数，逐元素实现功能 :math:`y = tan(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


softmax
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def softmax(tensor_i, axis, out_name=None):
          #pass

功能描述
"""""""""""
softmax激活函数，实现功能 :math:`tensor\_o = exp(tensor\_i)/sum(exp(tensor\_i),axis)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* axis：int型，表示进行运算的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


mish
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def mish(tensor_i,  out_name=None):
          #pass

功能描述
"""""""""""
mish激活函数，逐元素实现功能 :math:`y = x * tanh(ln(1 + e^{x}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。



hswish
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def hswish(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
hswish激活函数，逐元素实现功能 :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。



arccos
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def arccos(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
arccos反余弦激活函数，逐元素实现功能 :math:`y = arccos(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


arctanh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def arctanh(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
arctanh反双曲正切激活函数，逐元素实现功能 :math:`y = arctanh(x)=\frac{1}{2}ln(\frac{1+x}{1-x})`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


sinh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sinh(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
sinh双曲正弦激活函数，逐元素实现功能 :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。



cosh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def cosh(tensor_i,  out_name=None):
          #pass

功能描述
"""""""""""
cosh双曲余弦激活函数，逐元素实现功能 :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


sign
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sign(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
sign激活函数，逐元素实现功能 :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


gelu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gelu(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
gelu激活函数，逐元素实现功能 :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


hsigmoid
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def hsigmoid(tensor_i,  out_name=None):
          #pass

功能描述
"""""""""""
hsigmoid激活函数，逐元素实现功能 :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Data Arrange Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

permute
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def permute(tensor, order=(), out_name=None):
          #pass

功能描述
"""""""""""
根据置换参数对输入Tensor进行重排。
例如：输入shape为（6，7，8，9），置换参数order为（1，3，2，0），则输出的shape为（7，9，8，6）。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* order：List[int]或Tuple[int]型，表示置换参数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

tile
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def tile(tensor_i, reps, out_name=None):
          #pass

功能描述
"""""""""""
在指定的维度重复复制数据。
该操作属于 **受限本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* reps：List[int]或Tuple[int]型，表示每个维度的复制份数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


concat
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def concat(tensors, axis=0, out_name=None):
          #pass

功能描述
"""""""""""
对多个张量在指定的轴上进行拼接。

该操作属于 **受限本地操作** 。

参数说明
"""""""""""
* tensors：List[Tensor]类型，存放多个Tensor，所有的Tensor要求数据格式一致并具有相同的shape维度数，且除了待拼接的那一维，shape其他维度的值应该相等。
* axis：int型，表示进行拼接运算的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

split
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def split(tensor, axis=0, num=1, size=(), out_name=None):
          #pass

功能描述
"""""""""""
对输入Tensor在指定的轴上拆成多个Tensor。如果size不为空，则由分裂后的大小由size决定，反之则会根据tensor尺寸和num计算平均分裂后的大小。

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示将要进行切分的Tensor。
* axis：int型，表示进行切分运算的轴。
* num：int型，表示切分的份数；
* size：List[int]或Tuple[int]型，非平均分裂时，指定每一份大小，平均分裂时，设置为空即可。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个List[Tensor]，其中每个Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

pad
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def pad(tensor, padding=None, value=None, method='constant', out_name=None):
          #pass

功能描述
"""""""""""
对输入Tensor进行填充。

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示将要进行填充的Tensor。
* padding：List[int]或Tuple[int]型或None。padding为None时使用一个长度为2*len(tensor.shape)的全0list。
* value：Saclar或Variable型或None，表示待填充的数值。数据类型和tensor一致；
* method：string类型，表示填充方法，可选方法"constant"，"reflect"，"symmetric"，"edge"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

repeat
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def repeat(tensor_i, reps, out_name=None):
          #pass

功能描述
"""""""""""
在指定的维度重复复制数据。功能同tile。
该操作属于 **受限本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* reps：List[int]或Tuple[int]型，表示每个维度的复制份数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。



Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arg
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def arg(tensor, method='max', axis=0, keep_dims=True, out_name=None):
          #pass

功能描述
"""""""""""
对输入tensor的指定的axis求最大或最小值，输出对应的index，并将该axis的dim设置为1。
该操作属于 **受限本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入的操作Tensor。
* method：string类型，表示操作的方法，可选'max'，'min'。
* axis：int型，表示指定的轴。
* keep_dims：bool型，表示是否保留运算后的指定轴，默认值为True表示保留（此时该轴长度为1）。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。

tensor数据类型可以是FLOAT32/INT8/UINT8。tensor_o数据类型可以是INT32/FLOAT32。
tensor数据类型为INT8/UINT8时，tensor_o只能为INT32。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Shape About Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

squeeze
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def squeeze(tensor_i, axis, out_name=None):
          #pass

功能描述
"""""""""""
降维操作，去掉输入shape指定的某些1维的轴，如果没有指定轴(axis)则去除所有是1维的轴。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* axis：List[int]或Tuple[int]型，表示指定的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

reshape
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def reshape(tensor, new_shape, out_name=None):
          #pass

功能描述
"""""""""""
对输入tensor做reshape的操作。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* new_shape：List[int]或Tuple[int]或Tensor类型，表示转化后的形状。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

shape_fetch
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def shape_fetch(tensor_i, begin_axis=0, end_axis=1, step=1, out_name=None):
          #pass

功能描述
"""""""""""
对输入tensor取指定轴(axis)之间的shape信息。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* begin_axis：int型，表示指定开始的轴。
* end_axis：int型，表示指定结束的轴。
* step：int型，表示步长。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型为INT32。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

Quant Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

requant_fp_to_int
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def requant_fp_to_int(tensor_i,
                              scale,
                              offset,
                              requant_mode,
                              out_dtype,
                              out_name = None,
                              round_mode='half_away_from_zero'):

功能描述
"""""""""""
对输入tensor进行量化处理。

当requant_mode==0时，该操作对应的计算式为：

    ::

        output = saturate(int(round(input * scale)) + offset)，
        其中saturate为饱和到output的数据类型

    * BM1684X：input数据类型可以是FLOAT32, output数据类型可以是INT16/UINT16/INT8/UINT8

当requant_mode==1时，该操作对应的计算式为：

    ::

        output = saturate(int(round(float(input) * scale + offset)))，
        其中saturate为饱和到output的数据类型

    * BM1684X：input数据类型可以是INT32/INT16/UINT16, output数据类型可以是INT16/UINT16/INT8/UINT8

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：requant_mode==0时，List[int]型或int型；requant_mode==1时，List[float]型或float型。表示输出偏移。
* requant_mode：int型，表示量化模式。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* out_dtype：string类型，表示输入Tensor的类型.
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。
