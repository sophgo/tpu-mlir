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
                     dtype: str = "float32",
                     scale: Union[float, List[float]] = None,
                     zero_point: Union[int, List[int]] = None)
               #pass

如上所示，TpuLang中Tensor有5个参数。

* shape：Tensor的形状，List[int]，对于Operator输出的Tensor，可以不指定shape，默认值为[]。
* Name：Tensor的名称，string或None，该值推荐使用默认值None以免因为Name相同导致问题；
* ttype：Tensor的类型，可以是"neuron"或"coeff"，初始值为"neuron"；
* data：Tensor的数据，ndarray，为默认值None，当ttype为coeff时，不可以为None；为ndarray时，data的shape，dtype必须与输入shape，dtype一致。
* dtype：Tensor的数据类型，默认值为"float32"，否则取值范围为"float32", "float16", "int32", "uint32", "int16", "uint16", "int8", "uint8"；
* scale：Tensor的量化参数，float或List[float]，默认值为None；
* zero_point：Tensor的偏移参数，int或List[int]，默认值为None；

声明Tensor的示例：

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      #weight
      weight = tpul.Tensor(dtype='float32', shape=[3,4], data=np.random.uniform(0,1,shape).astype('float32'), ttype="coeff")

张量前处理(Tensor.preprocess)
---------------

TpuLang中Tensor如果是输入，且需要对输入进行前处理，可以调用该函数

TpuLang中Tensor.preprocess的定义如下：

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

如上所示，TpuLang中Tensor的preprocess有4个参数。

* mean：Tensor的每个channel的平均值，默认值为[0, 0, 0]；
* scale：Tensor的每个channel的scale值，默认值为[1, 1, 1]；
* pixel_format：Tensor的pixel的方式，默认值为'bgr'，取值范围为：'rgb'，'bgr'，'gray'，'rgba'，'gbrg'，'grbg'，'bggr'，'rggb'；
* channel_format：Tensor的格式是，channel维在前还是在最后。默认值为'nchw'，取值范围为"nchw"，"nhwc"。
* resize_dims：Tensor的resize后的[h，w]，默认值为None，表示取Tensor的h和w；
* keep_aspect_ratio：resize参数，是否保持相同的scale。bool量，默认值为False；
* keep_ratio_mode：resize参数，如果使能keep_aspect_ratio的两种模式，默认值'letterbox'，取值范围为'letterbox'，'short_side_scale'；
* pad_value：resize参数，当resize时pad的值。int类型，默认值为0；
* pad_type：resize参数，当resize时pad的方式。str类型，默认值为'center'，取值范围为'normal'，'center'；
* white_level：raw参数。str类型，默认值为4095；
* black_level：raw参数。str类型，默认值为112；

声明Tensor.preprocess的示例：

   .. code-block:: python

      #activation
      input = tpul.Tensor(name='x', shape=[2,3], dtype='int8')
      input.preprocess(mean=[123.675,116.28,103.53], scale=[0.017,0.017,0.017])
      # pass


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

* device：string类型。取值范围"cpu"\|"bm1684x"\|"bm1688"\|"cv183x"。

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
            refs=None,
            mode='f32',         # unused
            dynamic=False,
            save_in_mem=False)


功能描述
:::::::::::::::::::::::

用于将TpuLang模型编译为bmodel。

参数说明
:::::::::::::::::::::::

* name：string类型。模型名称。
* inputs：List[Tensor]，表示编译网络的所有输入Tensor；
* outputs：List[Tensor]，表示编译网络的所有输出Tensor；
* cmp：bool类型，True表示需要结果比对，False表示仅编译；
* refs：List[Tensor]，表示编译网络的所有需要比对验证的Tensor；
* mode：string类型，废弃。
* dynamic：bool类型，是否进行动态编译。
* save_in_mem：bool类型，是否将中间文件暂存到共享内存并随进程释放，启用该项时Compile会返回生成的bmodel文件的bytes-like object，用户需要自行接收和处理，如使用f.write(bmodel_bin)保存。

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

功能描述
"""""""""""
二维卷积运算。可参考各框架下的二维卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为1或2。
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为1或2。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* groups：int型，表示卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。


conv_int
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    output = (input - izp) Conv (weight - wzp) + bias[c];

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。bias的数据类型为int32
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为4。List中顺序为[上， 下， 左， 右]
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
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


conv_quant
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    conv_i32 = (input - izp) Conv (weight - wzp) + bias[c];
    output = requant_int(conv_i32, mul, shift) + ozp
    其中mul，shift由iscale，wscale，oscale得到

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。bias的数据类型为int32
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为2。List中顺序为[长，宽]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为4。List中顺序为[上， 下， 左， 右]
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv
* input_scale：List[float]型或float型，表示输入量化参数。取None则使用input Tensor中的量化参数，取List时要求长度为ic。
* weight_scale：List[float]型或float型，表示卷积核量化参数。取None则使用weight Tensor中的量化参数，取List时要求长度为oc。
* output_scale：List[float]型或float型，表示卷积核量化参数。不可以取None，取List时要求长度为oc。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* output_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。

deconv
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维反卷积运算。可参考各框架下的二维反卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* stride：List[int]，表示步长大小，取None则表示[1,1]，不为None时要求长度为1或2。
* dilation：List[int]，表示空洞大小，取None则表示[1,1]，不为None时要求长度为1或2。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* output_padding：List[int]，表示输出的填充大小，取None则表示[0,0,0,0]，不为None时要求长度为1或2或4。
* group：int类型，表示表示卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。


conv3d
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
三维卷积运算。可参考各框架下的三维卷积定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor，5维NCDHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCDHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1, 1]或[oc]，oc表示输出Channel数。
* stride：List[int]，表示步长大小，取None则表示[1,1,1]，不为None时要求长度为1或3。
* dilation：List[int]，表示空洞大小，取None则表示[1,1,1]，不为None时要求长度为1或3。
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0,0,0]，不为None时要求长度为1或3或6。
* groups：int型，表示卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。


conv3d_int
:::::::::::::::::

接口定义
"""""""""""

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
* stride：List[int]，表示步长大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[stride_t, stride_h, stride_w]
* dilation：List[int]，表示空洞大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[dilation_t, dilation_h, dilation_w]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0,0,0]，不为None时要求长度为4。List中顺序为[前， 后， 上， 下， 左， 右]
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
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


conv3d_quant
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
二维卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    conv_i32 = (input - izp) Conv (weight - wzp) + bias[c];
    output = requant_int(conv_i32, mul, shift) + ozp
    其中mul，shift由iscale，wscale，oscale得到

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，5维NCTHW格式。
* weight：Tensor类型，表示卷积核Tensor，5维[oc, ic, kt, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kt是kernel_t，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1， 1]。bias的数据类型为int32
* stride：List[int]，表示步长大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[stride_t, stride_h, stride_w]
* dilation：List[int]，表示空洞大小，取None则表示[1,1,1]，不为None时要求长度为2。List中顺序为[dilation_t, dilation_h, dilation_w]
* pad：List[int]，表示填充大小，取None则表示[0,0,0,0,0,0]，不为None时要求长度为4。List中顺序为[前， 后， 上， 下， 左， 右]
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv3d
* input_scale：List[float]型或float型，表示输入量化参数。取None则使用input Tensor中的量化参数，取List时要求长度为ic。
* weight_scale：List[float]型或float型，表示卷积核量化参数。取None则使用weight Tensor中的量化参数，取List时要求长度为oc。
* output_scale：List[float]型或float型，表示卷积核量化参数。不可以取None，取List时要求长度为oc。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* output_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


matmul
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* right：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* left_transpose：bool型，默认为False。表示计算时是否对左矩阵进行转置。
* right_transpose：bool型，默认为False。表示计算时是否对右矩阵进行转置。
* output_transpose：bool型，默认为False。表示计算时是否对输出矩阵进行转置。
* keep_dims：bool型，默认为True。表示结果是否保持原来的dim，False则shape为2维。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求lhr.shape[-1] == rhs.shape[-2]，lhr.shape[:-2]和rhs.shape[:-2]需要满足广播规则。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与right的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与right,bias的数据类型必须一致。


matmul_int
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* right：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* left_transpose：bool型，默认为False。表示计算时是否对左矩阵进行转置。
* right_transpose：bool型，默认为False。表示计算时是否对右矩阵进行转置。
* output_transpose：bool型，默认为False。表示计算时是否对输出矩阵进行转置。
* keep_dims：bool型，默认为True。表示结果是否保持原来的dim，False则shape为2维。
* input_zp：int型，表示input的偏移。取None则表示0。
* right_zp：int型，表示right的偏移。取None则表示0。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求lhr.shape[-1] == rhs.shape[-2]，lhr.shape[:-2]和rhs.shape[:-2]需要满足广播规则。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。

matmul_quant
:::::::::::::::::

接口定义
"""""""""""

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

功能描述
"""""""""""
量化的矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* right：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* right_transpose：bool型，默认为False。表示计算时是否对右矩阵进行转置。
* keep_dims：bool型，默认为True。表示结果是否保持原来的dim，False则shape为2维。
* input_scale：float型，表示input的量化参数。取None则使用input Tensor中的量化参数。
* right_scale：float型，表示right的量化参数。取None则使用right Tensor中的量化参数。
* output_scale：float型，表示output的量化参数。不可以取None。
* input_zp：int型，表示input的偏移。取None则表示0。
* right_zp：int型，表示right的偏移。取None则表示0。
* output_zp：int型，表示output的偏移。取None则表示0。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求lhr.shape[-1] == rhs.shape[-2]，lhr.shape[:-2]和rhs.shape[:-2]需要满足广播规则。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


Base Element-wise Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def add(tensor_i0, tensor_i1, scale = None, zero_point = None, out_dtype = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


sub
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sub(tensor_i0, tensor_i1, scale = None, zero_point = None, out_dtype = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float'/'float16'时，输出数据类型必须与输入一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


mul
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def mul(tensor_i0, tensor_i1, scale = None, zero_point = None, out_dtype = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32。


max
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def max(tensor_i0, tensor_i1, scale = None, zero_point = None, out_dtype = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


min
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def min(tensor_i0, tensor_i1, scale = None, zero_point = None, out_dtype = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。

copy
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def copy(tensor_i, out_name=None):
          #pass

功能描述
"""""""""""
copy，将输入数据复制到输出Tensor中.
该操作属于 **全局操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gt(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

lt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def lt(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

ge
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ge(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

le
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def le(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

eq
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def eq(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

ne
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ne(tensor_i0, tensor_i1, scale = None, zero_point = None, out_name = None):
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
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

gts
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gts(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素大于比较运算。 :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

lts
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def lts(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素小于比较运算。 :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

ges
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ges(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素大于等于比较运算。 :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
les
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def les(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素小于等于比较运算。 :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

eqs
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def eqs(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素等于比较运算。 :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

nes
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def nes(tensor_i0, scalar_i1, scale = None, zero_point = None, out_name = None):
          #pass

功能描述
"""""""""""
张量和标量的按元素不等于比较运算。 :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Tensor类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

Activation Operator
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
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


prelu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def prelu(input: Tensor, slope : Tensor, out_name=None):
          #pass

功能描述
"""""""""""
prelu激活函数，逐元素实现功能 :math:`y =\begin{cases}x\quad x>0\\x*slope \quad x<=0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor。
* slope：Tensor类型，表示slope Tensor。仅支持slope为coeff Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


leaky_relu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def leaky_relu(input, negative_slope=0.01, out_name=None):
          #pass

功能描述
"""""""""""
leaky_relu激活函数，逐元素实现功能 :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* negative_slope：FLOAT类型，表示输入的负斜率。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

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
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

ln
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ln(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
ln激活函数，逐元素实现功能 :math:`y = log(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

ceil
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def ceil(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
ceil向上取整激活函数，逐元素实现功能 :math:`y = \left \lfloor x \right \rfloor`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

floor
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def floor(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
floor向下取整激活函数，逐元素实现功能 :math:`y = \left \lceil x \right \rceil`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


sin
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sin(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
sin正弦激活函数，逐元素实现功能 :math:`y = sin(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。


cos
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def cos(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
cos余弦激活函数，逐元素实现功能 :math:`y = cos(x)`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

exp
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def exp(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
exp指数激活函数，逐元素实现功能 :math:`y = e^{x}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

tanh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def tanh(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
tanh双曲正切激活函数，逐元素实现功能 :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

sigmoid
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sigmoid(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
sigmoid激活函数，逐元素实现功能 :math:`y = 1 / (1 + e^{-x})`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

log_sigmoid
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def log_sigmoid(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
log_sigmoid激活函数，逐元素实现功能 :math:`y = log(1 / (1 + e^{-x}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

elu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def elu(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
elu激活函数，逐元素实现功能 :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

square
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def square(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
square平方激活函数，逐元素实现功能 :math:`y = \square{x}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

sqrt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sqrt(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
sqrt平方根激活函数，逐元素实现功能 :math:`y = \sqrt{x}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

rsqrt
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def rsqrt(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
rsqrt平方根取反激活函数，逐元素实现功能 :math:`y = 1 / (sqrt{x})`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

silu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def silu(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
silu激活函数，逐元素实现功能 :math:`y = x * (1 / (1 + e^{-x}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。


erf
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def erf(tensor, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
erf激活函数，对于输入输出Tensor对应位置的元素x和y，逐元素实现功能 :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


softmax_int
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def softmax_int(input: Tensor,
                      axis: int,
                      scale: List[float],
                      zero_point: List[int] = None,
                      out_name: str = None):
          #pass

功能描述
"""""""""""
softmax定点运算。可参考各框架下的softmax定义。

    ::

      for i in range(256)
        table[i] = exp(scale[0] * i)

      for n,h,w in N,H,W
        max_val = max(input[n,c,h,w] for c in C)
        sum_exp = sum(table[max_val - input[n,c,h,w]] for c in C)
        for c in C
          prob = table[max_val - input[n,c,h,w]] / sum_exp
          output[n,c,h,w] = saturate(int(round(prob * scale[1])) + zero_point[1]),    其中saturate饱和到output数据类型


其中table表示查表。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* axis：int型，表示进行运算的轴。
* scale：List[float]型，表示输入和输出的量化系数。长度必须时2。
* zero_point：List[int]型或None型，表示输入和输出偏移。如果为None，则取[0, 0]。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。
* BM1684X：输入数据类型可以是INT8/UINT8。


mish
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def mish(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
mish激活函数，逐元素实现功能 :math:`y = x * tanh(ln(1 + e^{x}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。



hswish
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def hswish(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
hswish激活函数，逐元素实现功能 :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。



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

      def sinh(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
sinh双曲正弦激活函数，逐元素实现功能 :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。



cosh
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def cosh(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
cosh双曲余弦激活函数，逐元素实现功能 :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。


sign
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def sign(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
sign激活函数，逐元素实现功能 :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


gelu
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def gelu(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
gelu激活函数，逐元素实现功能 :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32。

hsigmoid
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def hsigmoid(tensor_i, scale = None, zero_point = None, out_name=None):
          #pass

功能描述
"""""""""""
hsigmoid激活函数，逐元素实现功能 :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

broadcast
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def broadcast(tensor_i, reps, out_name=None):
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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


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
* tensors：List[Tensor]类型，存放多个Tensor，所有的Tensor要求数据格式一致并具有相同的shape维度数，且除了待拼接的那一维，shape其他维度的值应该相等。若数据类型包含scale或zero_point，则scale，zero_point必须一致。
* axis：int型，表示进行拼接运算的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

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
* padding：List[int]或Tuple[int]型或None。padding为None时使用一个长度为2*len(tensor.shape)的全 0 list。例如，一个hw的二维Tensor对应的padding是 [h_top, w_left, h_bottom,  w_right]。
* value：Saclar或Variable型或None，表示待填充的数值。数据类型和tensor一致；
* method：string类型，表示填充方法，可选方法"constant"，"reflect"，"symmetric"，"edge"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。

extract
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def extract(input: Tensor,
                    start: Union[List[int], Tuple[int]] = None,
                    end: Union[List[int], Tuple[int]] = None,
                    stride: Union[List[int], Tuple[int]] = None,
                    out_name: str = None)

功能描述
"""""""""""
对输入tensor进行切片提取操作。

参数说明
"""""""""""
* input：Tensor类型，表示输入张量。
* start：整数的列表或者元组或None，表示切片的起始位置，为None时表示全为0。
* end：整数的列表或者元组或None，表示切片的终止位置，为None时表示输出张量的形状。
* stride：整数的列表或者元组或None，表示切片的步长，为None时表示全为1。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，数据类型与输入Tensor的数据类型相同。

处理器支持
"""""""""""
* BM1688： 输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

Sort Operator
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
返回两个Tensor，INT32的indices和FLOAT32的values。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

topk
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def topk(input: Tensor,
                 axis: int,
                 k: int,
                 out_name: str = None):

功能描述
"""""""""""
按某个轴排序后前K个数。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor。
* axis：int型，表示排序所使用的轴。
* k：int型，表示沿着轴排序靠前的数的个数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回两个Tensor，第一个Tensor表示前几个数，其数据类型与输入类型相同，第二个Tensor表示前几个数在输入中的索引。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

sort
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def sort(input: Tensor,
                 axis: int = -1,
                 descending : bool = True,
                 out_name = None)

功能描述
"""""""""""
沿某个轴的输入张量进行排序，输出排序后的张量以及该张量的数据在输入张量中的索引。

参数说明
"""""""""""
* input：Tensor类型，表示输入张量。
* axis：int类型，表示指定的轴。(暂时只支持axis==-1)
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回两个Tensor，第一个张量的数据类型与输入张量的数据类型相同，第二个张量的数据类型为INT32。

处理器支持
"""""""""""
* BM1688：输入张量的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入张量的数据类型可以是FLOAT32/FLOAT16。


argsort
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def argsort(input: Tensor,
                    axis: int = 0,
                    descending : bool = True,
                    out_name = None)

功能描述
"""""""""""
沿某个轴的输入张量进行排序，输出排序后的张量的数据在输入张量中的索引。

参数说明
"""""""""""
* input：Tensor类型，表示输入张量。
* axis：int类型，表示指定的轴。(暂时只支持axis==-1)
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，其数据类型为INT32。

处理器支持
"""""""""""
* BM1688：输入张量的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入张量的数据类型可以是FLOAT32/FLOAT16。


sort_by_key
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def sort_by_key(input: Tensor,
                        key: Tensor,
                        axis: int = -1,
                        descending : bool = True,
                        out_name = None)

功能描述
"""""""""""
沿某个轴按键对输入张量进行排序，输出排序后的张量以及相应的键。

参数说明
"""""""""""
* input：Tensor类型，表示输入。
* key：Tensor类型，表示键。
* axis：int类型，表示指定的轴。
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回两个Tensor，第一个张量的数据类型与输入的数据类型相同，第二个张量的数据类型与键的数据类型相同。

处理器支持
"""""""""""
* BM1688：输入和键的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入和键的数据类型可以是FLOAT32/FLOAT16。


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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

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
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

unsqueeze
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def unsqueeze(tensor_i, axis, out_name=None):
          #pass

功能描述
"""""""""""
增维操作。在axis指定的位置增加1。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* axis：int型，表示指定的轴，设tensor_i的维度长度是D，则axis范围[-D,D-1)。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


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
                              requant_mode, #unused
                              out_dtype,
                              out_name = None,
                              round_mode='half_away_from_zero'):

功能描述
"""""""""""
对输入tensor进行量化处理。

该操作对应的计算式为

    ::

        output = saturate(int(round(input * scale)) + offset)，
        其中saturate为饱和到output的数据类型


该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型；表示输出偏移。
* requant_mode：int型，表示量化模式。废弃。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。(TODO)
* out_dtype：string类型，表示输入Tensor的类型.数据类型可以是"int16"/"uint16"/"int8"/"uint8"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


requant_fp
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def requant_fp(tensor_i,
                       scale,
                       offset,
                       out_dtype,
                       out_name = None,
                       round_mode='half_away_from_zero'):

功能描述
"""""""""""
对输入tensor进行量化处理。

该操作对应的计算式为：

    ::

        output = saturate(int(round(float(input) * scale + offset)))，
        其中saturate为饱和到output的数据类型


该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型。表示输出偏移。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* out_dtype：string类型，表示输入Tensor的类型。数据类型可以是"int16"/"uint16"/"int8"/"uint8"
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是INT32/INT16/UINT16。
* BM1684X：输入数据类型可以是INT32/INT16/UINT16。

requant_int
:::::::::::::::::::

对输入tensor进行量化处理。

    .. code-block:: python

        def requant_int(tensor_i,
                        mul,
                        shift,
                        offset,
                        requant_mode,
                        out_dtype = None,
                        out_name = None,
                        round_mode='half_up'):

功能描述
"""""""""""
对输入tensor进行量化处理。

当requant_mode==0时，该操作对应的计算式为：

    ::

        output = shift > 0 ? (input << shift) : input
        output = saturate((output * multiplier) >> 31),     其中 >> 为round_half_up, saturate饱和到INT32
        output = shift < 0 ? (output >> -shift) : output,   其中 >> 的舍入模式由round_mode确定。
        output = saturate(output + offset),                 其中saturate饱和到output数据类型

    * BM1684X：input数据类型可以是INT32, output数据类型可以是INT32/INT16/INT8
    * BM1688：input数据类型可以是INT32, output数据类型可以是INT32/INT16/INT8

当requant_mode==1时，该操作对应的计算式为：

    ::

        output = saturate((input * multiplier) >> 31)，     其中 >> 为round_half_up, saturate饱和到INT32
        output = saturate(output >> -shift + offset)，      其中 >> 的舍入模式由round_mode确定, saturate饱和到output数据类型

    * BM1684X：input数据类型可以是INT32, output数据类型可以是INT32/INT16/INT8
    * BM1688：input数据类型可以是INT32, output数据类型可以是INT32/INT16/INT8

当requant_mode==2时，该操作对应的计算式为(建议使用)：

    ::

        output = input * multiplier
        output = shift > 0 ? (output << shift) : (output >> -shift),    其中 >> 的舍入模式由round_mode确定
        output = saturate(output + offset),                             其中 saturate饱和到output数据类型

    * BM1684X：input数据类型可以是INT32/INT16/UINT16, output数据类型可以是INT16/UINT16/INT8/UINT8
    * BM1688：input数据类型可以是INT32/INT16/UINT16, output数据类型可以是INT16/UINT16/INT8/UINT8

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* mul：List[int]型或int型，表示量化乘子系数。
* shift:List[int]型或int型，表示量化移位系数。右移为正，左移为负。
* offset：List[int]型或int型，表示输出偏移。
* requant_mode：int型，表示量化模式。
* round_mode：string型，表示舍入模式。默认为“half_up”。
* out_dtype：string类型或None，表示输出Tensor的类型。None代表输出数据类型为“int8”
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

芯片支持
"""""""""""
* BM1684X

dequant_int_to_fp32
:::::::::::::::::::

    .. code-block:: python

        def dequant_int_to_fp(tensor_i,
                              scale,
                              offset,
                              out_dtype: str="float32",
                              out_name = None):

功能描述
"""""""""""
对输入tensor进行反量化处理。

该操作对应的计算式为：

    ::

        output = (input - offset) * scale


该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型，表示输出偏移。
* out_dtype：string类型，表示输出Tensor的类型。默认输出数据类型为“float32”。当输入数据类型为int8/uint8时，取值范围为“float16”，“float32”。当输入类型为int16/uint16时，输出类型只能为“float32”。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype指定。

芯片支持
"""""""""""
* BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8。

dequant_int
:::::::::::::::::::

    .. code-block:: python

        def dequant_int(tensor_i,
                        mul,
                        shift,
                        offset,
                        lshift,
                        quant_mode,
                        out_dtype = None,
                        out_name = None,
                        round_mode='half_up'):


功能描述
"""""""""""
对输入tensor进行反量化处理。

当quant_mode==0时，该操作对应的计算式为：

    ::

        output = (intpu - offset) * multiplier
        output = saturate(output >> -shift),                其中 >> 的舍入模式由round_mode确定, saturate饱和到INT32

    * BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8, output数据类型可以是INT32/INT16/UINT16

当quant_mode==1时，该操作对应的计算式为：

    ::

        output = ((input - offset) * multiplier) << lshift
        output = saturate(output >> 31)，                   其中 >> 为round_half_up, saturate饱和到INT32
        output = saturate(output >> -shift)，               其中 >> 的舍入模式由round_mode确定, saturate饱和到output数据类型

    * BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8, output数据类型可以是INT32/INT16/INT8

该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* mul：List[int]型或int型，表示量化乘子系数。
* shift:List[int]型或int型，表示量化移位系数。右移为负，左移为正。
* offset：List[int]
* lshift：int型，表示左移位系数。
* requant_mode：int型，表示量化模式。
* round_mode：string型，表示舍入模式。默认为“half_up”。
* out_dtype：string类型，表示输入Tensor的类型.
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

芯片支持
"""""""""""
* BM1684X
* BM1688

Up/Down Scaling Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

maxpool2d
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def maxpool2d(input: Tensor,
                    kernel: List[int]=None,
                    stride: List[int] = None,
                    pad: List[int] = None,
                    ceil_mode: bool = False,
                    scale: List[float] = None,
                    zero_point: List[int] = None,
                    out_name: str = None):
          #pass

功能描述
"""""""""""
对输入Tensor进行Max池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* ceil：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

芯片支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


maxpool2d_with_mask
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def maxpool2d_with_mask(input: Tensor,
                              kernel: List[int]=None,
                              stride: List[int] = None,
                              pad: List[int] = None,
                              ceil_mode: bool = False,
                              out_name: str = None,
                              mask_name: str = None):
          #pass

功能描述
"""""""""""
对输入Tensor进行Max池化处理，并输出其mask index。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* ceil：bool型，表示计算output shape时是否向上取整。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* mask_name：string类型或None，表示输出Mask的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回两个Tensor，一个Tensor的数据类型与输入Tensor相同。另一个返回一个坐标Tensor，该Tensor是记录使用比较运算池化时所选择的坐标。

芯片支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

avgpool2d
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def avgpool2d(input: Tensor,
                    kernel: List[int]=None,
                    stride: List[int] = None,
                    pad: List[int] = None,
                    ceil_mode: bool = False,
                    scale: List[float] = None,
                    zero_point: List[int] = None,
                    out_name: str = None):
          #pass

功能描述
"""""""""""
对输入Tensor进行Avg池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* ceil：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

芯片支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

upsample
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def upsample(tensor_i, scale = 1, out_name=None):
          #pass

功能描述
"""""""""""
在h和w维度对输入tensor数据进行scale倍重复扩展输出。
该操作属于 **本地操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* scale：int型，表示扩展倍数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

reduce
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def reduce(tensor_i, method='sum', axis=0, keep_dims=False, out_name=None):
          #pass

功能描述
"""""""""""
依据axis_list，对输入的tensor做reduce操作。
该操作属于 **受限本地操作** ；仅当输入数据类型为FLOAT32时是 **本地操作**。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* method：string类型，表示reduce方法，目前可选”mean”,”max”,”min”,”sum”,”prod”,"L1","L2"。
* axis：List[int]或Tuple[int]或int，表示需要reduce的轴。
* keep_dims：bool型，表示是否要保留原先的维度。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


Normalization Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

batch_norm
:::::::::::::::::::

接口定义
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


功能描述
"""""""""""
该batch_norm算子先完成输入值的批归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的batch_norm算子。

该操作属于 **本地操作** 。

参数说明
"""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* mean：Tensor类型，表示输入的均值，shape为[c]。
* variance：Tensor类型，表示输入的方差值，shape为[c]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全1Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全0Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回Tensor类型，表示输出待归一化的Tensor。

芯片支持
"""""""""""
* BM1684：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

layer_norm
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def layer_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int,
                     out_name: str = None):
          #pass


功能描述
"""""""""""
该layer_norm算子先完成输入值的归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的layer_norm算子。

该操作属于 **本地操作** 。

参数说明
"""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全 1 Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全 0 Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* axis：int型，第一个标准化的维度。 如果rank(X)为r，则axis的允许范围为[-r, r)。 负值表示从后面开始计算维度。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回Tensor类型，表示输出待归一化的Tensor。

芯片支持
"""""""""""
* BM1684：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

group_norm
:::::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def group_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     num_groups: int,
                     out_name: str = None):
          #pass


功能描述
"""""""""""
该group_norm算子先完成输入值的归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的group_norm算子。

该操作属于 **本地操作** 。

参数说明
"""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全1Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全0Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* num_groups：int型，表示分组的数量。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回Tensor类型，表示输出待归一化的Tensor。

芯片支持
"""""""""""
* BM1684：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。



Vision Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nms
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def nms(boxes: Tensor,
                scores: Tensor,
                format: str = 'PYTORCH',
                max_box_num_per_class: int = 0,
                out_name: str = None)

功能描述
"""""""""""
对输入tensor进行非极大值抑制处理。

参数说明
"""""""""""
* boxes：Tensor类型，表示输入框的列表。必须是三维张量，第一维为批的个数，第二维为框的个数，第三维为框的4个坐标。
* scores：Tensor类型，表示输入得分的列表。必须是三维张量，第一维为批的个数，第二维为类的个数，第三维为框的个数。
* format：string类型，'TENSORFLOW'表示Tensorflow格式[y1, x1, y2, x2]，'PYTORCH'表示Pytorch格式[x_center, y_center, width, height]。
* max_box_num_per_class：int型，表示每个类中的输出框的最大个数。默认为0。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，表示从框列表中选出的框的索引的列表，它是一个2维张量，格式为[num_selected_indices, 3], 其中每个索引的格式为[batch_index, class_index, box_index]。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


interpolate
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def interpolate(input: Tensor,
                        scale_h: float,
                        scale_w: float,
                        method: str = 'nearest',
                        coord_mode: str = "pytorch_half_pixel",
                        out_name: str = None)

功能描述
"""""""""""
对输入tensor进行插值。

参数说明
"""""""""""
* input：Tensor类型，表示输入Tensor。
* scale_h：float型，表示h方向的放缩系数。
* scale_w：float型，表示w方向的放缩系数。
* method: string类型，表示插值方法，可选项为"nearest"或"linear"。
* coord_mode: string类型, 表示输出坐标的计算方法，可选项为"align_corners"/"pytorch_half_pixel"/ "half_pixel"/"asymmetric"等。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

其中， `coord_mode` 的意义跟onnx的 `Resize` 算子的参数 `coordinate_transformation_mode` 的意义时一样的。若h/w方向的放缩因子为 `scale` ，输入坐标为 `x_in` ，输入尺寸为 `l_in` ，输出坐标为 `x_out` ，输出尺寸为 `l_out` ，则逆映射定义如下：

* `"half_pixel"`：

    ::

        x_in = (x_out + 0.5) / scale - 0.5

* `"pytorch_half_pixel"`：

    ::

        x_in = len > 1 ? (x_out + 0.5) / scale - 0.5 : 0

* `"align_corners"`：

    ::

        x_in = x_out * (l_in - 1) / (l_out - 1)

* `"asymmetric"`：

    ::

        x_in = x_out / scale


返回值
"""""""""""
返回一个Tensor，数据类型与输入类型相同。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16(TODO)。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16(TODO)。


Select Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nonzero
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

      def nonzero(tensor_i, dtype = 'int32', out_name=None):
          #pass

功能描述
"""""""""""
抽取输入Tensor data为true时对应的位置信息信息。
该操作属于 **全局操作** 。

参数说明
"""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* dtype：string型，表示输出数据类型，目前仅可使用默认值"int32"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，数据类型为INT32。

处理器支持
"""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

lut
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def lut(input: Tensor,
                table: Tensor,
                out_name: str = None)

功能描述
"""""""""""
对输入tensor进行查找表查找操作。

参数说明
"""""""""""
* input：Tensor类型，表示输入。
* table：Tensor类型，表示查找表。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
"""""""""""
返回一个Tensor，数据类型与张量 `table` 的数据类型相同。

处理器支持
"""""""""""
* BM1688： `input` 的数据类型可以是INT8/UINT8， `table` 的数据类型可以是INT8/UINT8。
* BM1684X： `input` 的数据类型可以是INT8/UINT8， `table` 的数据类型可以是INT8/UINT8。

select
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def select(lhs: Tensor,
                   rhs: Tensor,
                   tbrn: Tensor,
                   fbrn: Tensor,
                   type: str,
                   out_name = None)

功能描述
"""""""""""
根据 `lhs` 与 `rhs` 的数值比较结果来选择，条件为真时，选择 `tbrn` ，条件为假时，选择 `fbrn` 。

参数说明
"""""""""""
* lhs：Tensor类型，表示左边的张量。
* rhs：Tensor类型，表示右边的张量。
* tbrn：Tensor类型，表示条件为真时取的值。
* fbrn：Tensor类型，表示条件为假时取的值。
* type: string类型，表示比较符。可选项为"Greater"/"Less"/"GreaterOrEqual"/"LessOrEqual"/"Equal"/"NotEqual"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

约束条件：要求 `lhs` 与 `rhs` 的形状和数据类型相同， `tbrn` 与 `fbrn` 的形状和数据类型相同。

返回值
"""""""""""
返回一个Tensor，数据类型与张量 `tbrn`的数据类型相同。

处理器支持
"""""""""""
* BM1688： `lhs` / `rhs` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。
* BM1684X： `lhs` / `rhs` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。


cond_select
:::::::::::::::::

接口定义
"""""""""""

    .. code-block:: python

        def cond_select(cond: Tensor,
                        tbrn: Union[Tensor, Scalar, float, int],
                        fbrn: Union[Tensor, Scalar, float, int],
                        out_name = None)

功能描述
"""""""""""
根据条件 `cond` 来选择，条件为真时，选择 `tbrn` ，条件为假时，选择 `fbrn` 。

参数说明
"""""""""""
* cond：Tensor类型，表示条件。
* tbrn：Tensor类型或Scalar类型，表示条件为真时取的值。
* fbrn：Tensor类型或Scalar类型，表示条件为假时取的值。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

约束条件：若 `tbrn` 和 `fbrn` 皆为张量，则要求 `tbrn` 与 `fbrn` 的形状和数据类型相同。

返回值
"""""""""""
返回一个Tensor，数据类型与张量 `tbrn` 的数据类型相同。

处理器支持
"""""""""""
* BM1688： `cond` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。
* BM1684X： `cond` / `tbrn` / `fbrn` 的输入数据类型可以是FLOAT32/FLOAT16(TODO)。
