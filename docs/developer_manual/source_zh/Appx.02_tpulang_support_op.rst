附录02：TpuLang的基本元素
==============================


本章将介绍TpuLang程序的基本元素：Tensor、Scalar、Control Functions和Operator。

.. _tensor:

张量(Tensor)
--------------------------------------

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
* data：Tensor的数据，ndarray或None，默认值为None，此时Tensor将根据指定的形状初始化为全零。当ttype为coeff时，不可以为None，data为ndarray，此时data的shape，dtype必须与输入shape，dtype一致。
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
--------------------------------------

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

如上所示，TpuLang中Tensor的preprocess有如下几个参数。

* mean：Tensor的每个channel的平均值，默认值为[0, 0, 0]；
* scale：Tensor的每个channel的scale值，默认值为[1, 1, 1]；
* pixel_format：Tensor的pixel的方式，默认值为'bgr'，取值范围为：'rgb'，'bgr'，'gray'，'rgba'，'gbrg'，'grbg'，'bggr'，'rggb'；
* channel_format：Tensor的格式，channel维在前还是在最后。默认值为'nchw'，取值范围为"nchw"，"nhwc"。
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
--------------------------------------

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
--------------------------------------

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

* device：string类型。取值范围"BM1684X"\|"BM1688"\|"CV183X"。

.. _compile:
compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

接口定义
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


功能描述
:::::::::::::::::::::::::::::::::::::::::::::::::

用于将TpuLang模型编译为bmodel。

参数说明
:::::::::::::::::::::::::::::::::::::::::::::::::

* name：string类型。模型名称。
* inputs：List[Tensor]，表示编译网络的所有输入Tensor；
* outputs：List[Tensor]，表示编译网络的所有输出Tensor；
* cmp：bool类型，True表示需要结果比对，False表示仅编译；如果mlir_inference为False，cmp参数无效。
* refs：List[Tensor]，表示编译网络的所有需要比对验证的Tensor；
* mode：string类型，废弃。
* dynamic：bool类型，是否进行动态编译。
* no_save：bool类型，是否将中间文件暂存到共享内存并随进程释放，启用该项时Compile会返回生成的bmodel文件的bytes-like object，用户需要自行接收和处理，如使用f.write(bmodel_bin)保存。
* asymmetric：bool类型，是否为非对称量化。
* opt：int类型，表示编译器group优化级别。0，表示不需要进行group；1，表示尽可能进行group；2，表示根据动态规划进行group。默认值为2。
* mlir_inference: bool类型，是否执行mlir的推理，如果为False, cmp参数无效。
* bmodel_inference: bool类型，是否执行bmodel的推理。
* log_level 用来控制日志等级，目前支持only-pass、only-layer-group、normal、quiet:
  - only-pass: 主要打印图优化pattern匹配情况。
  - only-layer-group: 主要打印layer group 信息。
  - normal: 编译生成bmodel的日志都会打印出来
  - quiet: 什么都不打印
* embed_debug_info: bool类型，是否开启profile模式。

.. _deinit:

反初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在网络构建之后，需要进行反初始化结束。只有在反初始化后，之前
Tpulang的数据才会得到释放

    .. code-block:: python

       def deinit():
          #pass

.. _reset_default_graph:

重置默认图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在网络构建之前，需要进行重置默认图操作。如果输入graph为None，重置默认图后，当前graph为空图。
如果设置输入graph，会设置graph为默认图。
如果只有一个子图，可以不需要显示调用reset_default_graph。因为init函数会调用该函数。

    .. code-block:: python

       def reset_default_graph(graph = None):
          #pass

.. _get_default_graph:

获取当前默认图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在网络构建之后，如果需要得到默认的子图，调用该函数可以得到默认的graph。

    .. code-block:: python

       def get_default_graph():
          #pass

.. _reset_graph:

重置图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果需要清除graph以及其保存的Tensor信息，可以调用该函数。graph为None时，清除当前默认图的信息。

    .. code-block:: python

       def reset_graph(graph = None):
          #pass

注意：如果graph中的Tensor还被其他graph使用，不要调用该函数清除graph信息

.. _RoundingMode:

舍入模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
舍入是指按照一定的规则舍去某些数字后面多余的尾数的过程，以得到更简短、明确的数字表示。
给定 x，舍入结果是 y，有下面的舍入模式供选择。

邻近偶数四舍五入(Half to Even)
""""""""""""""""""""""""""""""""""""""""""""""

    四舍五入，当小数值为0.5时舍入到邻近的偶数，对应的值是 :cpp:enumerator:`half_to_even`。

远离原点四舍五入(Half Away From Zero)
""""""""""""""""""""""""""""""""""""""""""""""

    四舍五入，正数接近于正无穷，负数接近于负无穷，对应的值是 :cpp:enumerator:`half_away_from_zero`，公式如下

    .. math:: \mathsf{y = \mathrm{sign}(x)\left\lfloor|x| + 0.5\right\rfloor = -\mathrm{sign}(x)\left\lceil-|x| - 0.5\right\rceil}

截断取整(Towards Zero)
""""""""""""""""""""""""""""""""""""""""""""""

    无条件舍去，接近于原点，对应的值是 :cpp:enumerator:`towards_zero`，公式如下

    .. math:: \mathsf{y = \mathrm{sign}(x)\left\lfloor|x|\right\rfloor = -\mathrm{sign}(x)\left\lceil-|x|\right\rceil} = {\begin{cases}\mathsf{\lfloor x\rfloor}&{\text{if}}\mathsf{\ \ x > 0,}\\ \mathsf{\lceil x\rceil}&{\text{otherwise}}.\end{cases}}

下取整(Down)
""""""""""""""""""""""""""""""""""""""""""""""

    接近于负无穷，对应的值是 :cpp:enumerator:`down`，公式如下

    .. math:: \mathsf{y = \lfloor x\rfloor = -\lceil-x\rceil}

上取整(Up)
""""""""""""""""""""""""""""""""""""""""""""""

    接近于正无穷，对应的值是 :cpp:enumerator:`up`，公式如下

    .. math:: \mathsf{y = \lceil x\rceil = -\lfloor-x\rfloor}

向上四舍五入(Half Up)
""""""""""""""""""""""""""""""""""""""""""""""

    四舍五入，接近于正无穷，对应的值是 :cpp:enumerator:`half_up`，公式如下

    .. math:: \mathsf{y = \lceil x + 0.5\rceil = -\lfloor-x - 0.5\rfloor = \left\lceil\frac{\lfloor 2x\rfloor}{2}\right\rceil}

向下四舍五入(Half Down)
""""""""""""""""""""""""""""""""""""""""""""""

    四舍五入，接近于正无穷，对应的值是 :cpp:enumerator:`half_down`，公式如下

    .. math:: \mathsf{y = \lfloor x - 0.5\rfloor = -\lceil-x + 0.5\rceil = \left\lfloor\frac{\lceil 2x\rceil}{2}\right\rfloor}

例子
""""""""""""""""""""""""""""""""""""""""""""""

下表列出不同舍入模式下 x 与 y 的对应关系。

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
--------------------------------------

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
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
二维卷积运算。可参考各框架下的二维卷积定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1]，不为None时要求长度2。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1]，不为None时要求长度为2。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0]，不为None时要求长度为4。
* groups：int型，表示卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。


conv_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
二维卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    output = (input - izp) Conv (weight - wzp) + bias[c];

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。bias的数据类型为int32。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1]，不为None时要求长度为2。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1]，不为None时要求长度为2。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0]，不为None时要求长度为4。
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。当前不支持List[int]型。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


conv_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。bias的数据类型为int32。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1]，不为None时要求长度为2。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1]，不为None时要求长度为2。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0]，不为None时要求长度为4。
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv。
* input_scale：List[float]型或float型，表示输入量化参数。取None则使用input Tensor中的量化参数，取List时要求长度为ic。当前不支持List[float]型。
* weight_scale：List[float]型或float型，表示卷积核量化参数。取None则使用weight Tensor中的量化参数，取List时要求长度为oc。
* output_scale：List[float]型或float型，表示卷积核量化参数。不可以取None，取List时要求长度为oc。当前不支持List[float]型。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。当前不支持List[int]型。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* output_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。当前不支持List[int]型。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。

deconv
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
二维反卷积运算。可参考各框架下的二维反卷积定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]，oc表示输出Channel数。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1]，不为None时要求长度为2。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1]，不为None时要求长度为2。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0]，不为None时要求长度为4。
* output_padding：List[int]，表示输出每个空间维度的填充大小，取None则表示[0,0]，不为None时要求长度为2。
* group：int类型，表示表示反卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。

deconv_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
二维反卷积定点运算。可参考各框架下的二维卷积定义。
::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    wzp = is_wzp_const ? wzp_val : wzp_vec[c];
    output = (input - izp) Deconv (weight - wzp) + bias[c];

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，4维NCHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维[oc, ic, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1]。bias的数据类型为int32
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1]，不为None时要求长度为2。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1]，不为None时要求长度为2。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0]，不为None时要求长度为4。
* output_padding：List[int]，表示输出的填充大小，取None则表示[0,0]，不为None时要求长度为1或2。
* groups：int型，表示反卷积层的组数。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。当前不支持List[int]型。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。

conv3d
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
三维卷积运算。可参考各框架下的三维卷积定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor，5维NCDHW格式。
* weight：Tensor类型，表示卷积核Tensor，4维NCDHW格式。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1, 1]或[oc]，oc表示输出Channel数。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1,1]，不为None时要求长度为3。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1,1]，不为None时要求长度为3。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0,0,0]，不为None时要求长度为6。
* groups：int型，表示卷积层的组数。
* out_dtype：string类型或None，为None时与input数据类型一致。取值为范围为“float32”，“float16”。表示输出Tensor的数据类型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与weight的数据类型必须一致。bias的数据类型必须是FLOAT32。


conv3d_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
三维卷积定点运算。可参考各框架下的三维卷积定义。

::

  for c in channel
    izp = is_izp_const ? izp_val : izp_vec[c];
    kzp = is_kzp_const ? kzp_val : kzp_vec[c];
    output = (input - izp) Conv3d (weight - kzp) + bias[c];

其中Conv3d表示3D卷积计算。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，5维NCTHW格式。
* weight：Tensor类型，表示卷积核Tensor，5维[oc, ic, kt, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kt是kernel_t，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1, 1]。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1,1]，不为None时要求长度为3。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1,1]，不为None时要求长度为3。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0,0,0]。不为None时要求长度6。
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv3d。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。当前不支持List[int]型。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为ic，其中ic表示输入的Channel数。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


conv3d_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，5维NCTHW格式。
* weight：Tensor类型，表示卷积核Tensor，5维[oc, ic, kt, kh, kw]格式。其中oc表示输出Channel数，ic表示输入channel数，kt是kernel_t，kh是kernel_h，kw是kernel_w。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[1, oc, 1, 1， 1]。bias的数据类型为int32。
* stride：List[int]，表示每个空间维度的步长大小，取None则表示[1,1,1]，不为None时要求长度为3。
* dilation：List[int]，表示每个空间维度的空洞大小，取None则表示[1,1,1]，不为None时要求长度为3。
* pad：List[int]，表示每个空间维度的填充大小，遵循[x1_begin, x2_begin…x1_end, x2_end,…]顺序。取None则表示[0,0,0,0,0,0]。不为None时要求长度6。
* groups：int型，表示卷积层的组数。若ic=oc=groups时，则卷积为depthwise conv3d
* input_scale：List[float]型或float型，表示输入量化参数。取None则使用input Tensor中的量化参数，取List时要求长度为ic。当前不支持List[float]型。
* weight_scale：List[float]型或float型，表示卷积核量化参数。取None则使用weight Tensor中的量化参数，取List时要求长度为oc。
* output_scale：List[float]型或float型，表示卷积核量化参数。不可以取None，取List时要求长度为oc。当前不支持List[float]型。
* input_zp：List[int]型或int型，表示输入偏移。取None则表示0，取List时要求长度为ic。当前不支持List[int]型。
* weight_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。
* output_zp：List[int]型或int型，表示卷积核偏移。取None则表示0，取List时要求长度为oc。当前不支持List[int]型。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入和权重的数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


matmul
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
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
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求input.shape[-1] == right.shape[-2]，input.shape[:-2]和right.shape[:-2]需要满足广播规则。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。input与right的数据类型必须一致。bias的数据类型必须是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。input与right,bias的数据类型必须一致。


matmul_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* right：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* left_transpose：bool型，默认为False。表示计算时是否对左矩阵进行转置。
* right_transpose：bool型，默认为False。表示计算时是否对右矩阵进行转置。
* output_transpose：bool型，默认为False。表示计算时是否对输出矩阵进行转置。
* keep_dims：bool型，默认为True。表示结果是否保持原来的dim，False则shape为2维。
* input_zp：List[int]型或int型，表示input的偏移。取None则表示0。当前不支持List[int]型。
* right_zp：List[int]型或int型，表示right的偏移。取None则表示0。当前不支持List[int]型。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int32。取值范围：int32/uint32
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求input.shape[-1] == right.shape[-2]，input.shape[:-2]和right.shape[:-2]需要满足广播规则。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。

matmul_quant
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
量化的矩阵乘运算。可参考各框架下的矩阵乘定义。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入左操作数，大于或等于2维，设最后两维shape=[m,k]。
* right：Tensor类型，表示输入右操作数，大于或等于2维，设最后两维shape=[k,n]。
* bias：Tensor类型，表示偏置Tensor。为None时表示无偏置，反之则要求shape为[n]。
* right_transpose：bool型，默认为False。表示计算时是否对右矩阵进行转置。
* keep_dims：bool型，默认为True。表示结果是否保持原来的dim，False则shape为2维。
* input_scale：List[float]型或float型，表示input的量化参数。取None则使用input Tensor中的量化参数。当前不支持List[float]型。
* right_scale：List[float]型float型，表示right的量化参数。取None则使用right Tensor中的量化参数。当前不支持List[float]型。
* output_scale：List[float]型float型，表示output的量化参数。不可以取None。当前不支持List[float]型。
* input_zp：List[int]型或int型，表示input的偏移。取None则表示0。当前不支持List[int]型。
* right_zp：List[int]型或int型，表示right的偏移。取None则表示0。当前不支持List[int]型。
* output_zp：List[int]型或int型，表示output的偏移。取None则表示0。当前不支持List[int]型。
* out_dtype：string类型或None，表示输入Tensor的类型，取None表示为int8。取值范围：int8/uint8
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

要求左右Tensor的维度长度一致。
当Tensor的维度长度为2时，表示矩阵和矩阵乘运算。
当Tensor的维度长度大于2时，表示批矩阵乘运算。要求input.shape[-1] == right.shape[-2]，input.shape[:-2]和right.shape[:-2]需要满足广播规则。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。
* BM1684X：输入数据类型可以是INT8/UINT8。偏置的数据类型为INT32。


Base Element-wise Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def add(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素加法运算。 :math:`tensor\_o = tensor\_i0 + tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致（当其中一个输入为'int8'则输出默认为'int8'类型）。当输入为'float32'/'float16'时，输出数据类型必须与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


sub
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sub(tensor_i0: Union[Tensor, Scalar, int, float],
                tensor_i1: Union[Tensor, Scalar, int, float],
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_dtype: str = None,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素减法运算。 :math:`tensor\_o = tensor\_i0 - tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当输入为'float32'/'float16'时，输出数据类型必须与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


mul
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mul(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素乘法运算。 :math:`tensor\_o = tensor\_i0 * tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'/'uint8'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致（当其中一个输入为'int8'则输出默认为'int8'类型）。当输入为'float32'/'float16'时，输出数据类型必须与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。


div
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def div(tensor_i0: Union[Tensor, Scalar],
            tensor_i1: Union[Tensor, Scalar],
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素除法运算。 :math:`tensor\_o = tensor\_i0 / tensor\_i1`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32。


max
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def max(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素取最大值。 :math:`tensor\_o = max(tensor\_i0, tensor\_i1)`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT16/UINT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT16/UINT16/INT8/UINT8。


min
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def min(tensor_i0: Union[Tensor, Scalar, int, float],
            tensor_i1: Union[Tensor, Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_dtype: str = None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素取最小值。 :math:`tensor\_o = min(tensor\_i0, tensor\_i1)`。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int、float，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int、float，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。
* out_dtype：string类型或None，表示输出Tensor的数据类型，为None时会与输入数据类型一致。可选参数为'float32'/'float16'/'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。当数据类型为FLOAT16/FLOAT32时，tensor_i0与tensor_i1的数据类型必须一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT16/UINT16/INT32/UINT32/INT8/UINT8。



add_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def add_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
运算公式 :math:`tensor\_o = (tensor\_i0 + tensor\_i1) << shift`。
张量和张量的按元素相加后再舍入算术移shift位，shift为正时，左移，shift为负时，右移。舍入模式由round_mode确定。
add_shift数据相加后，以INT64为中间结果保存，然后在INT64基础上做一次舍入的算数移位操作；
结果支持饱和处理；当tensor_i0、tensor_i1为signed，且tensor_o为unsigned时，结果必须饱和处理。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* shift：int型，表示移位的位数。
* round_mode：String型，表示舍入模式。默认值为'half_away_from_zero'。取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* is_saturate：Bool型，表示结果是否需要饱和处理，默认饱和处理。
* out_dtype：String或None，表示输出Tensor的数据类型，取默认值时则和tensor_i0的类型一致。可选参数为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。
该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。
* BM1684X：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。


sub_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sub_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
运算公式 :math:`tensor\_o = (tensor\_i0 - tensor\_i1) << shift`。
张量和张量的按元素相减后再舍入算术移shift位，shift为正时，左移，shift为负时，右移。舍入模式由round_mode确定。
sub_shift数据相减后，以INT64为中间结果保存，然后在INT64基础上做一次舍入的算数移位操作；结果支持饱和处理。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* shift：int型，表示移位的位数。
* round_mode：String型，表示舍入模式。默认值为'half_away_from_zero'。取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* is_saturate：Bool型，表示结果是否需要饱和处理，默认饱和处理。
* out_dtype：String或None，表示输出Tensor的数据类型，取默认值时则和tensor_i0的类型一致。可选参数为'int8'/'int16'/'int32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。
该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。
* BM1684X：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。


mul_shift
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mul_shift(tensor_i0: Union[Tensor, Scalar, int],
                    tensor_i1: Union[Tensor, Scalar, int],
                    shift: int,
                    out_dtype: str,
                    round_mode: str='half_away_from_zero',
                    is_saturate: bool=True,
                    out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
运算公式 :math:`tensor\_o = (tensor\_i0 * tensor\_i1) << shift`。
张量和张量的按元素相减后再舍入算术移shift位，shift为正时，左移，shift为负时，右移。舍入模式由round_mode确定。
mul_shift数据相乘后，以INT64为中间结果保存，然后在INT64基础上做一次舍入的算数移位操作；
结果支持饱和处理；当tensor_i0、tensor_i1为signed，且tensor_o为unsigned时，结果必须饱和处理。
该操作支持广播。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型或Scalar、int，表示输入左操作Tensor或Scalar。
* tensor_i1：Tensor类型或Scalar、int，表示输入右操作Tensor或Scalar。tensor_i0和tensor_i1至少有一个是Tensor。
* shift：int型，表示移位的位数。
* round_mode：String型，表示舍入模式。默认值为'half_away_from_zero'。取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* is_saturate：Bool型，表示结果是否需要饱和处理，默认饱和处理。
* out_dtype：String或None，表示输出Tensor的数据类型，取默认值时则和tensor_i0的类型一致。可选参数为'int8'/'uint8'/'int16'/'uint16'/'int32'/'uint32'。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。
该Tensor的数据类型由out_dtype指定，或与输入数据类型一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。
* BM1684X：输入数据类型可以是INT32/UINT32/INT16/UINT6/INT8/UINT8。


copy
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def copy(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
copy，将输入数据复制到输出Tensor中.
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

clamp
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def clamp(input: Tensor, min: float, max: float, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
将输入Tensor中所有元素的值都限定在设置的最大最小值范围内，大于最大值则截断为最大值，小于最大值则截断为最小值。
要求所有输入Tensor及Scalar的dtype一致。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor。
* min：float类型，表示阶段的下限。
* max：float类型，表示阶段的上限。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

Element-wise Compare Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gt
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def gt(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素大于比较运算。 :math:`tensor\_o = tensor\_i0 > tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

lt
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def lt(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素小于比较运算。 :math:`tensor\_o = tensor\_i0 < tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

ge
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ge(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素大于等于比较运算。 :math:`tensor\_o = tensor\_i0 >= tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

le
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def le(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素小于等于比较运算。 :math:`tensor\_o = tensor\_i0 <= tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

eq
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def eq(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素等于比较运算。 :math:`tensor\_o = tensor\_i0 == tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

ne
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ne(tensor_i0: Tensor,
            tensor_i1: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和张量的按元素不等于比较运算。 :math:`tensor\_o = tensor\_i0 != tensor\_i1 ? 1 : 0`。
该操作支持广播。
tensor_i0或者tensor_i1可以被指定为COEFF_TENSOR。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作Tensor。
* tensor_i1：Tensor类型，表示输入右操作Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的scale。tensor_i0与tensor_i1的scale必须一致。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为3，分别为tensor_i0，tensor_i1，output的zero_point。tensor_i0与tensor_i1的zero_point必须一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。tensor_i0与tensor_i1的数据类型必须一致。

gts
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def gts(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素大于比较运算。 :math:`tensor\_o = tensor\_i0 > scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

lts
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def lts(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素小于比较运算。 :math:`tensor\_o = tensor\_i0 < scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

ges
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ges(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素大于等于比较运算。 :math:`tensor\_o = tensor\_i0 >= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。


les
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def les(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素小于等于比较运算。 :math:`tensor\_o = tensor\_i0 <= scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

eqs
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def eqs(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素等于比较运算。 :math:`tensor\_o = tensor\_i0 == scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

nes
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nes(tensor_i0: Tensor,
            scalar_i1: Union[Scalar, int, float],
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
张量和标量的按元素不等于比较运算。 :math:`tensor\_o = tensor\_i0 != scalar\_i1 ? 1 : 0`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i0：Tensor类型，表示输入左操作数。
* scalar_i1：Scalar，int或float类型，表示输入右操作数。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。scalar_i1数据类型为FLOAT32。

Activation Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

relu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def relu(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
relu激活函数，逐元素实现功能 :math:`y = max(0, x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


prelu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def prelu(input: Tensor, slope : Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
prelu激活函数，逐元素实现功能 :math:`y =\begin{cases}x\quad x>0\\x*slope \quad x<=0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor。
* slope：Tensor类型，表示slope Tensor。仅支持slope为coeff Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


leaky_relu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def leaky_relu(input: Tensor,
                    negative_slope: float = 0.01,
                    out_name: str = None,
                    round_mode : str="half_away_from_zero",):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
leaky_relu激活函数，逐元素实现功能 :math:`y =\begin{cases}x\quad x>0\\x*params_[0] \quad x<=0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor。
* negative_slope：float类型，表示输入的负斜率，默认值为0.01。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

abs
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def abs(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
abs绝对值激活函数，逐元素实现功能 :math:`y = \left | x \right |`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。若输入是quantized类型，输出的scale与zero_point与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

ln
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ln(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
ln激活函数，逐元素实现功能 :math:`y = log(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

ceil
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def ceil(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
ceil向上取整激活函数，逐元素实现功能 :math:`y = \left \lfloor x \right \rfloor`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

floor
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def floor(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
floor向下取整激活函数，逐元素实现功能 :math:`y = \left \lceil x \right \rceil`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

round
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def round(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
round四舍五入整激活函数，逐元素实现功能 :math:`y = round(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


sin
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sin(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
sin正弦激活函数，逐元素实现功能 :math:`y = sin(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。


cos
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cos(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
cos余弦激活函数，逐元素实现功能 :math:`y = cos(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

exp
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def exp(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
exp指数激活函数，逐元素实现功能 :math:`y = e^{x}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

tanh
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tanh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None,
            round_mode : str="half_away_from_zero"):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
tanh双曲正切激活函数，逐元素实现功能 :math:`y=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

sigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sigmoid(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None,
                round_mode : str="half_away_from_zero"):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
sigmoid激活函数，逐元素实现功能 :math:`y = 1 / (1 + e^{-x})`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

log_sigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def log_sigmoid(input: Tensor,
                    scale: List[float]=None,
                    zero_point: List[int]=None,
                    out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
log_sigmoid激活函数，逐元素实现功能 :math:`y = log(1 / (1 + e^{-x}))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

elu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def elu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
elu激活函数，逐元素实现功能 :math:`y =  \begin{cases}x\quad x>=0\\e^{x}-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

square
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def square(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
square平方激活函数，逐元素实现功能 :math:`y = \square{x}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

sqrt
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sqrt(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
sqrt平方根激活函数，逐元素实现功能 :math:`y = \sqrt{x}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

rsqrt
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rsqrt(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
rsqrt平方根取反激活函数，逐元素实现功能 :math:`y = 1 / (sqrt{x})`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

silu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def silu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
silu激活函数，逐元素实现功能 :math:`y = x * (1 / (1 + e^{-x}))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

swish
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def swish(input: Tensor,
              beta: float,
              scale: List[float]=None,
              zero_point: List[int]=None,
              round_mode: str = "half_away_from_zero",
              out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
swish激活函数，逐元素实现功能 :math:`y = x * (1 / (1 + e^{-x * beta}))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor。
* beta: Scalar或float类型，表示beta值。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。


erf
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def erf(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
erf激活函数，对于输入输出Tensor对应位置的元素x和y，逐元素实现功能 :math:`y = \frac{2}{\sqrt{\pi }}\int_{0}^{x}e^{-\eta ^{2}}d\eta`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

tan
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tan(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
tan正切激活函数，逐元素实现功能 :math:`y = tan(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。


softmax
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def softmax(input: Tensor,
                axis: int,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
softmax激活函数，实现功能 :math:`tensor\_o = exp(tensor\_i)/sum(exp(tensor\_i),axis)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* axis：int型，表示进行运算的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


softmax_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def softmax_int(input: Tensor,
                    axis: int,
                    scale: List[float],
                    zero_point: List[int] = None,
                    out_name: str = None,
                    round_mode : str="half_away_from_zero"):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* axis：int型，表示进行运算的轴。
* scale：List[float]型，表示输入和输出的量化系数。长度必须时2。
* zero_point：List[int]型或None型，表示输入和输出偏移，长度与scale一致。如果为None，则取[0, 0]。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT8/UINT8。
* BM1684X：输入数据类型可以是INT8/UINT8。


mish
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def mish(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
mish激活函数，逐元素实现功能 :math:`y = x * tanh(ln(1 + e^{x}))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。



hswish
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hswish(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
hswish激活函数，逐元素实现功能 :math:`y =\begin{cases}0\quad x<=-3\\x \quad x>=3\\x*((x+3)/6) \quad -3<x<3\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


arccos
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arccos(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
arccos反余弦激活函数，逐元素实现功能 :math:`y = arccos(x)`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。


arctanh
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def arctanh(input: Tensor, out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
arctanh反双曲正切激活函数，逐元素实现功能 :math:`y = arctanh(x)=\frac{1}{2}ln(\frac{1+x}{1-x})`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。FLOAT16数据会自动转换为FLOAT32。


sinh
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sinh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
sinh双曲正弦激活函数，逐元素实现功能 :math:`y = sinh(x)=\frac{e^{x}-e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。



cosh
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cosh(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
cosh双曲余弦激活函数，逐元素实现功能 :math:`y = cosh(x)=\frac{e^{x}+e^{-x}}{2}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。


sign
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def sign(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
sign激活函数，逐元素实现功能 :math:`y =\begin{cases}1\quad x>0\\0\quad x=0\\-1\quad x<0\\\end{cases}`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


gelu
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def gelu(input: Tensor,
            scale: List[float]=None,
            zero_point: List[int]=None,
            out_name: str = None,
            round_mode : str="half_away_from_zero"):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
gelu激活函数，逐元素实现功能 :math:`y = x* 0.5 * (1+ erf(\frac{x}{\sqrt{2}}))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

hsigmoid
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def hsigmoid(input: Tensor,
                scale: List[float]=None,
                zero_point: List[int]=None,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
hsigmoid激活函数，逐元素实现功能 :math:`y = min(1, max(0, \frac{x}{6} + 0.5))`。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入Tensor。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的scale。
* zero_point：List[int]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为tensor_i0，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的形状和数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32/INT8/UINT8。FLOAT16数据会自动转换为FLOAT32。

Data Arrange Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

permute
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def permute(input:tensor,
                  order:Union[List[int], Tuple[int]],
                  out_name:str=None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据置换参数对输入Tensor进行重排。
例如：输入shape为（6，7，8，9），置换参数order为（1，3，2，0），则输出的shape为（7，9，8，6）。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* order：List[int]或Tuple[int]型，表示置换参数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

tile
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def tile(tensor_i: Tensor,
               reps: Union[List[int], Tuple[int]],
               out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
在指定的维度重复复制数据。
该操作属于 **受限本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* reps：List[int]或Tuple[int]型，表示每个维度的复制份数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

broadcast
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def broadcast(input: Tensor,
                    reps: Union[List[int], Tuple[int]],
                    out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
在指定的维度重复复制数据。
该操作属于 **受限本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* reps：List[int]或Tuple[int]型，表示每个维度的复制份数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8/INT16/UINT16。


concat
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

    def concat(inputs: List[Tensor],
               scales: Optional[Union[List[float],List[int]]] = None,
               zero_points: Optional[List[int]] = None,
               axis: int = 0,
               out_name: str = None,
               dtype="float32",
               round_mode: str="half_away_from_zero"):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对多个张量在指定的轴上进行拼接, 以及支持不同量纲输入、输出。

该操作属于 **受限本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* inputs：List[Tensor]类型，存放多个Tensor，所有的Tensor要求数据格式一致并具有相同的shape维度数，且除了待拼接的那一维，shape其他维度的值应该相等。
* scales：Optional[Union[List[float],List[int]]]类型，存放多个输入和一个输出scale，最后一个为输出的scale。
* zero_points：Optional[List[int]]类型，存放多个输入和一个输出的zero_point, 最后一个为输出的zero_point。
* axis：int型，表示进行拼接运算的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* dtype：string类型,默认是"float32"。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

split
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def split(input:tensor,
                axis:int=0,
                num:int=1,
                size:Union[List[int], Tuple[int]]=None,
                out_name:str=None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor在指定的轴上拆成多个Tensor。如果size不为空，则由分裂后的大小由size决定，反之则会根据tensor尺寸和num计算平均分裂后的大小。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示将要进行切分的Tensor。
* axis：int型，表示进行切分运算的轴。
* num：int型，表示切分的份数；
* size：List[int]或Tuple[int]型，非平均分裂时，指定每一份大小，平均分裂时，设置为空即可。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个List[Tensor]，其中每个Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

pad
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def pad(input:tensor,
              method='constant',
              value:Union[Scalar, Variable, None]=None,
              padding:Union[List[int], Tuple[int], None]=None,
              out_name:str=None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行填充。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示将要进行填充的Tensor。
* method：string类型，表示填充方法，可选方法"constant"，"reflect"，"symmetric"，"edge"。
* value：Saclar或Variable型或None，表示待填充的数值。数据类型和tensor一致；
* padding：List[int]或Tuple[int]型或None。padding为None时使用一个长度为2*len(tensor.shape)的全 0 list。例如，一个hw的二维Tensor对应的padding是 [h_top, w_left, h_bottom,  w_right]。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

repeat
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def repeat(tensor_i:Tensor,
                 reps:Union[List[int], Tuple[int]],
                 out_name:str=None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
在指定的维度重复复制数据。功能同tile。
该操作属于 **受限本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* reps：List[int]或Tuple[int]型，表示每个维度的复制份数。要求order长度和tensor维度一致。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。

extract
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def extract(input: Tensor,
                    start: Union[List[int], Tuple[int]] = None,
                    end: Union[List[int], Tuple[int]] = None,
                    stride: Union[List[int], Tuple[int]] = None,
                    out_name: str = None)

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行切片提取操作。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入张量。
* start：整数的列表或者元组或None，表示切片的起始位置，为None时表示全为0。
* end：整数的列表或者元组或None，表示切片的终止位置，为None时表示输出张量的形状。
* stride：整数的列表或者元组或None，表示切片的步长，为None时表示全为1。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型与输入Tensor的数据类型相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688： 输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8/INT16/UINT16。


roll
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def roll(input:Tensor,
                shifts: Union[int, List[int], Tuple[int]],
                dims: Union[int, List[int], Tuple[int]]   = None,
                out_name:str=None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
沿给定维度滚动输入张量。移出最后一个位置的元素将在第一个位置重新引入。如果 dims 为 None，则张量将在滚动之前展平，然后恢复到原始形状。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* shifts：int，List[int]或Tuple[int]型，张量元素移动的位数。如果 shifts 是元组/列表，则 dims 必须是相同大小的元组/列表，并且每个维度将按相应的值滚动。
* dims：int，List[int]，Tuple[int]型或None, 滚动的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8/INT16/UINT16。


Sort Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arg
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

          def arg(input: Tensor,
                method: str = "max",
                axis: int = 0,
                keep_dims: bool = True,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor的指定的axis求最大或最小值，输出对应的index，并将该axis的dim设置为1。
该操作属于 **受限本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入的操作Tensor。
* method：string类型，表示操作的方法，可选'max'，'min'。
* axis：int型，表示指定的轴。默认值为0。
* keep_dims：bool型，表示是否保留运算后的指定轴，默认值为True表示保留（此时该轴长度为1）。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。


返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回两个Tensor，第一个Tensor表示indices，类型为int32；第二个Tensor表示values，类型会和input的类型一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

topk
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def topk(input: Tensor,
                 axis: int,
                 k: int,
                 out_name: str = None):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
按某个轴排序后前K个数。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入Tensor。
* axis：int型，表示排序所使用的轴。
* k：int型，表示沿着轴排序靠前的数的个数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回两个Tensor，第一个Tensor表示前几个数，其数据类型与输入类型相同，第二个Tensor表示前几个数在输入中的索引。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。

sort
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sort(input: Tensor,
                 axis: int = -1,
                 descending : bool = True,
                 out_name = None)

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
沿某个轴的输入张量进行排序，输出排序后的张量以及该张量的数据在输入张量中的索引。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入张量。
* axis：int类型，表示指定的轴。(暂时只支持axis==-1)
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回两个Tensor，第一个张量的数据类型与输入张量的数据类型相同，第二个张量的数据类型为INT32。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入张量的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入张量的数据类型可以是FLOAT32/FLOAT16。


argsort
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def argsort(input: Tensor,
                    axis: int = -1,
                    descending : bool = True,
                    out_name : str = None)


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
沿某个轴的输入张量进行排序，输出排序后的张量的数据在输入张量中的索引。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入张量。
* axis：int类型，表示指定的轴。(暂时只支持axis==-1)
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，其数据类型为INT32。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入张量的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入张量的数据类型可以是FLOAT32/FLOAT16。


sort_by_key
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def sort_by_key(input: Tensor,
                        key: Tensor,
                        axis: int = -1,
                        descending : bool = True,
                        out_name = None)

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
沿某个轴按键对输入张量进行排序，输出排序后的张量以及相应的键。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入。
* key：Tensor类型，表示键。
* axis：int类型，表示指定的轴。
* descending：bool类型，表示是否按从大到小排列。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回两个Tensor，第一个张量的数据类型与输入的数据类型相同，第二个张量的数据类型与键的数据类型相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入和键的数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入和键的数据类型可以是FLOAT32/FLOAT16。


Shape About Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

squeeze
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def squeeze(tensor_i: Tensor, axis: Union[Tuple[int], List[int]], out_name: str = None):
          #pass


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
降维操作，去掉输入shape指定的某些1维的轴，如果没有指定轴(axis)则去除所有是1维的轴。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* axis：List[int]或Tuple[int]型，表示指定的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

reshape
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def reshape(tensor: Tensor, new_shape: Union[Tuple[int], List[int], Tensor], out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor做reshape的操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* new_shape：List[int]或Tuple[int]或Tensor类型，表示转化后的形状。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

shape_fetch
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def shape_fetch(tensor_i: Tensor,
                begin_axis: int = None,
                end_axis: int = None,
                step: int = 1,
                out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor取指定轴(axis)之间的shape信息。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* begin_axis：int型，表示指定开始的轴。
* end_axis：int型，表示指定结束的轴。
* step：int型，表示步长。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型为INT32。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。

unsqueeze
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def unsqueeze(input: Tensor, axes: List[int] = [1,2], out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
增维操作。在axis指定的位置增加1。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* axis：int型，表示指定的轴，设tensor_i的维度长度是D，则axis范围[-D,D-1)。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


Quant Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

requant_fp_to_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def requant_fp_to_int(tensor_i,
                              scale,
                              offset,
                              requant_mode, #unused
                              out_dtype,
                              out_name = None,
                              round_mode='half_away_from_zero'):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行量化处理。

该操作对应的计算式为

    ::

        output = saturate(int(round(input * scale)) + offset)，
        其中saturate为饱和到output的数据类型


该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型；表示输出偏移。
* requant_mode：int型，表示量化模式。废弃。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。(TODO)
* out_dtype：string类型，表示输入Tensor的类型.数据类型可以是"int16"/"uint16"/"int8"/"uint8"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


requant_fp
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def requant_fp(tensor_i: Tensor,
               scale: Union[float, List[float]],
               offset: Union[float, List[float]],
               out_dtype: str,
               out_name: str=None,
               round_mode: str='half_away_from_zero',
               first_round_mode: str='half_away_from_zero'):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行量化处理。

该操作对应的计算式为：

    ::

        output = saturate(int(round(float(input) * scale + offset)))，
        其中saturate为饱和到output的数据类型


该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型。表示输出偏移。
* out_dtype：string类型，表示输入Tensor的类型。数据类型可以是"int16"/"uint16"/"int8"/"uint8"
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* first_round_mode：string型，表示之前量化tensor_i时使用的舍入模式。默认为“half_away_from_zero”。first_round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是INT32/INT16/UINT16。
* BM1684X：输入数据类型可以是INT32/INT16/UINT16。

requant_int
:::::::::::::::::::::::::::::::::::::::::::::::::

对输入tensor进行量化处理。

    .. code-block:: python

        def requant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                requant_mode: int,
                out_dtype: str="int8",
                out_name=None,
                round_mode='half_away_from_zero', rq_axis:int = 1, fuse_rq_to_matmul: bool = False):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* mul：List[int]型或int型，表示量化乘子系数。
* shift:List[int]型或int型，表示量化移位系数。右移为负，左移为正。
* offset：List[int]型或int型，表示输出偏移。
* requant_mode：int型，表示量化模式。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”, 范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* out_dtype：string类型或None，表示输出Tensor的类型。None代表输出数据类型为“int8”
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* rq_axis：int型，表示在rq_axis维度做requant。
* fuse_rq_to_matmul：bool类型，表示是否将requant 融合到matmul,默认是False。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X
* BM1688

dequant_int_to_fp
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
    .. code-block:: python

        def dequant_int_to_fp(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int], float, List[float]],
                      out_dtype: str="float32",
                      out_name: str=None,
                      round_mode: str='half_away_from_zero'):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行反量化处理。

该操作对应的计算式为：

    ::

        output = (input - offset) * scale


该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* scale：List[float]型或float型，表示量化系数。
* offset：List[int]型或int型，表示输出偏移。
* out_dtype：string类型，表示输出Tensor的类型。默认输出数据类型为“float32”。当输入数据类型为int8/uint8时，取值范围为“float16”，“float32”。当输入类型为int16/uint16时，输出类型只能为“float32”。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：string型，表示舍入模式。默认为“half_away_from_zero”。round_mode取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。(TODO)


返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype指定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8。

dequant_int
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
    .. code-block:: python

        def dequant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                lshift: int,
                requant_mode: int,
                out_dtype: str="int8",
                out_name=None,
                round_mode='half_up'):


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行反量化处理。

当requant_mode==0时，该操作对应的计算式为：

    ::

        output = (intpu - offset) * multiplier
        output = saturate(output >> -shift),                其中 >> 的舍入模式由round_mode确定, saturate饱和到INT32

    * BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8, output数据类型可以是INT32/INT16/UINT16

当requant_mode==1时，该操作对应的计算式为：

    ::

        output = ((input - offset) * multiplier) << lshift
        output = saturate(output >> 31)，                   其中 >> 为round_half_up, saturate饱和到INT32
        output = saturate(output >> -shift)，               其中 >> 的舍入模式由round_mode确定, saturate饱和到output数据类型

    * BM1684X：input数据类型可以是INT16/UINT16/INT8/UINT8, output数据类型可以是INT32/INT16/INT8

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入Tensor，3-5维。
* mul：List[int]型或int型，表示量化乘子系数。
* shift:List[int]型或int型，表示量化移位系数。右移为负，左移为正。
* offset：List[int]型或int型，表示输出偏移。
* lshift：int型，表示左移位系数。
* requant_mode：int型，表示量化模式。取值为0和1，0表示“Normal”，1表示“TFLite”。
* round_mode：string型，表示舍入模式。默认为“half_up”, 范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。
* out_dtype：string类型，表示输入Tensor的类型。默认为“int8”。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor。该Tensor的数据类型由out_dtype确定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X

cast
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def cast(tensor_i: Tensor,
         out_dtype: str = 'float32',
         out_name: str = None,
         round_mode: str = 'half_away_from_zero'):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
将输入张量 `tensor_i` 转换为指定的数据类型 `out_dtype`，并根据指定的舍入模式 `round_mode` 对数据进行舍入。
注意本算子不能单独使用，必须配合其他算子。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* out_dtype: str = 'float32'，输出张量的数据类型，默认为 `float32`。
* out_name: str = None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode: str = 'half_away_from_zero', 舍入模式，默认为 `half_away_from_zero`。取值范围为“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”。注意，此函数round_mode不支持“half_up”与“half_down”。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型由输入的out_dtype决定。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/UINT8/INT8。


Up/Down Scaling Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

maxpool2d
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行Max池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* ceil：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：String型，当输入输出Tensor为量化时，表示舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


maxpool2d_with_mask
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def maxpool2d_with_mask(input: Tensor,
                              kernel: Union[List[int],Tuple[int],None] = None,
                              stride: Union[List[int],Tuple[int],None] = None,
                              pad:    Union[List[int],Tuple[int],None] = None,
                              ceil_mode: bool = False,
                              out_name: str = None,
                              mask_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行Max池化处理，并输出其mask index。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* ceil：bool型，表示计算output shape时是否向上取整。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* mask_name：string类型或None，表示输出Mask的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回两个Tensor，一个Tensor的数据类型与输入Tensor相同。另一个返回一个坐标Tensor，该Tensor是记录使用比较运算池化时所选择的坐标。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


maxpool3d
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行Max池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或int或None，输入None表示使用global_pooling，不为None时若输入单个整数，表示在3个维度上的kernel大小相同，若输入List或Tuple，要求该参数长度为3。
* stride：List[int]或Tuple[int]型或int或None，表示步长尺寸，输入None使用默认值[1,1,1]，不为None时若输入单个整数，表示在3个维度上的stride大小相同，若输入List或Tuple，要求该参数长度为3。
* pad：List[int]或Tuple[int]型或int或None，表示填充尺寸，输入None使用默认值[0,0,0,0,0,0]，不为None时若输入单个整数，表示在3个维度上的pad大小相同，若输入List或Tuple，要求该参数长度为6。
* ceil_mode：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* round_mode：String型，当输入输出Tensor为量化时，表示舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。



avgpool2d
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行Avg池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或None，输入None表示使用global_pooling，不为None时要求该参数长度为2。
* stride：List[int]或Tuple[int]型或None，表示步长尺寸，输入None使用默认值[1,1]，不为None时要求该参数长度为2。
* pad：List[int]或Tuple[int]型或None，表示填充尺寸，输入None使用默认值[0,0,0,0]，不为None时要求该参数长度为4。
* ceil_mode：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* count_include_pad：Bool类型，表示在计算平均值时，是否将pad值计算在内，默认值为False。
* round_mode：String型，当输入输出Tensor为量化时，表示舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。
* first_round_mode：String型，当输入输出Tensor为量化时，表示第一次的舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


avgpool3d
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor进行Avg池化处理。请参考各大框架下的池化操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor：Tensor类型，表示输入操作Tensor。
* kernel：List[int]或Tuple[int]型或int或None，输入None表示使用global_pooling，不为None时若输入单个整数，表示在3个维度上的kernel大小相同，若输入List或Tuple，要求该参数长度为3。
* pad：List[int]或Tuple[int]型或int或None，表示填充尺寸，输入None使用默认值[0,0,0,0,0,0]，不为None时若输入单个整数，表示在3个维度上的pad大小相同，若输入List或Tuple，要求该参数长度为6。
* stride：List[int]或Tuple[int]型或int或None，表示步长尺寸，输入None使用默认值[1,1,1]，不为None时若输入单个整数，表示在3个维度上的stride大小相同，若输入List或Tuple，要求该参数长度为3。
* ceil_mode：bool型，表示计算output shape时是否向上取整。
* scale：List[float]类型或None，量化参数。取None代表非量化计算。若为List，长度为2，分别为input，output的scale。
* zero_point：List[int]类型或None，偏移参数。取None代表非量化计算。若为List，长度为2，分别为input，output的zero_point。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* count_include_pad：Bool类型，表示在计算平均值时，是否将pad值计算在内，默认值为False。
* round_mode：String型，当输入输出Tensor为量化时，表示第二次的舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。
* first_round_mode：String型，当输入输出Tensor为量化时，表示第一次的舍入模式。默认值为'half_away_from_zero'。round_mode取值范围为"half_away_from_zero", "half_to_even", "towards_zero", "down", "up"。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8/UINT8。


upsample
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def upsample(tensor_i: Tensor,
                   scale: int = 2,
                   out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
在h和w维度对输入tensor数据进行scale倍重复扩展输出。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* scale：int型，表示扩展倍数。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16/INT8。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16/INT8。

reduce
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def reduce(tensor_i: Tensor,
                 method: str = 'ReduceSum',
                 axis: Union[List[int],Tuple[int],int] = None,
                 keep_dims: bool = False,
                 out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
依据axis_list，对输入的tensor做reduce操作。
该操作属于 **受限本地操作** ；仅当输入数据类型为FLOAT32时是 **本地操作**。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* method：string类型，表示reduce方法，目前可选"ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1","ReduceSum"。
* axis：List[int]或Tuple[int]或int，表示需要reduce的轴。
* keep_dims：bool型，表示是否要保留原先的维度。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入Tensor相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


Normalization Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

batch_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""""""""""
该batch_norm算子先完成输入值的批归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的batch_norm算子。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* mean：Tensor类型，表示输入的均值，shape为[c]。
* variance：Tensor类型，表示输入的方差值，shape为[c]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全1Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全0Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回Tensor类型，表示输出归一化后的Tensor，数据类型与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

layer_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def layer_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int,
                     out_name: str = None):
          #pass


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
该layer_norm算子先完成输入值的归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的layer_norm算子。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全 1 Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全 0 Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* axis：int型，第一个标准化的维度。 如果rank(X)为r，则axis的允许范围为[-r, r)。 负值表示从后面开始计算维度。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回Tensor类型，表示输出归一化后的Tensor，数据类型与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

group_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def group_norm(input: Tensor,
                     gamma: Tensor = None,
                     beta: Tensor = None,
                     epsilon: float = 1e-5,
                     num_groups: int,
                     out_name: str = None):
          #pass


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
该group_norm算子先完成输入值的归一化，完成归一化之后再进行缩放和平移。
批归一化运算过程可参考各框架的group_norm算子。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限，如果x只有1维，c为1，否则c等于x的shape[1]。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape为[c]，取None时相当于shape为[c]的全1Tensor。
* beta：Tensor类型或None，表示批归一化和缩放之后进行的平移，不为None时要求shape为[c]，取None时相当于shape为[c]的全0Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* num_groups：int型，表示分组的数量。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回Tensor类型，表示输出归一化后的Tensor，数据类型与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

rms_norm
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def rms_norm(input: Tensor,
                     gamma: Tensor = None,
                     epsilon: float = 1e-5,
                     axis: int = -1,
                     out_name: str = None):
          #pass


功能描述
""""""""""""""""""""""""""""""""""""""""""""""
该rms_norm算子先完成输入值最后一个维度的归一化，完成归一化之后再进行缩放。运算过程可参考各框架的RMSNorm算子。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""

* input：Tensor类型，表示输入待归一化的Tensor，维度不限。
* gamma：Tensor类型或None，表示批归一化之后进行的缩放，不为None时要求shape与input最后一维 `w` 相等，取None时相当于shape为[w]的全1Tensor。
* epsilon：FLOAT类型，表示为了除法运算数值稳定加在分母上的值。
* axis: int型，第一个标准化的维度。 如果rank(X)为r，则axis的允许范围为[-r, r)。 负值表示从后面开始计算维度。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回Tensor类型，表示输出归一化的后的Tensor，数据类型与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


normalize
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
    .. code-block:: python

      def normalize(input: Tensor,
              p: float = 2.0,
              axes: Union[List[int], int] = 1,
              eps : float = 1e-12,
              out_name: str = None):

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入张量input的指定维度进行 :math:`L_p`  归一化。

对于大小为 :math:`(n_0, ..., n_{dim}, ..., n_k)` 的张量输入，每个 :math:`n_{dim}`元素向量:math:`v`沿维度:attr:`axes` 的变换为

.. math::
  v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

在默认参数下，它对向量的维度1上进行L2归一化。

该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input: Tensor类型。表示输入Tensor。数据类型为float32, float16。
* p: float类型，默认值为2.0。表示是归一化过程中的指数值。
* axes: Union[List[int], int]类型，默认为1。表示要归一化的维度。如果是list，那么list内的值必须是连续的。注意，axes = [0,-1]并不是连续的。
* eps : float类型，默认值为1e-12。 一个极小值，用来避免除以0。
* out_name: string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回Tensor类型，表示输出归一化的后的Tensor，数据类型与输入一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

Vision Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nms
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def nms(boxes: Tensor,
                scores: Tensor,
                format: str = 'PYTORCH',
                max_box_num_per_class: int = 1,
                out_name: str = None)

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行非极大值抑制处理。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* boxes：Tensor类型，表示输入框的列表。必须是三维张量，第一维为批的个数，第二维为框的个数，第三维为框的4个坐标。
* scores：Tensor类型，表示输入得分的列表。必须是三维张量，第一维为批的个数，第二维为类的个数，第三维为框的个数。
* format：string类型，'TENSORFLOW'表示Tensorflow格式[y1, x1, y2, x2]，'PYTORCH'表示Pytorch格式[x_center, y_center, width, height], 默认值为'PYTORCH'。
* max_box_num_per_class：int型，表示每个类中的输出框的最大个数。必须大于0，默认值为1。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，表示从框列表中选出的框的索引的列表，它是一个2维张量，格式为[num_selected_indices, 3], 其中每个索引的格式为[batch_index, class_index, box_index]。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32。
* BM1684X：输入数据类型可以是FLOAT32。


interpolate
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def interpolate(input: Tensor,
                        scale_h: float,
                        scale_w: float,
                        method: str = 'nearest',
                        coord_mode: str = "pytorch_half_pixel",
                        out_name: str = None)

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行插值。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入的Tensor。必须是至少2维的张量。
* scale_h：float型，表示高度方向的缩放系数，必须大于0。
* scale_w：float型，表示宽度方向的缩放系数，必须大于0。
* method: string类型，表示插值方法，可选项为"nearest"或"linear"。默认值为"nearest"。
* coord_mode: string类型，表示输出坐标的计算方法，可选项为"align_corners"、"pytorch_half_pixel"、"half_pixel"、"asymmetric"。默认值为"pytorch_half_pixel"。
* out_name：string类型或None，表示输出Tensor的名称。如果为None，内部会自动生成名称。

其中， `coord_mode` 的意义跟onnx的 `Resize` 算子的参数 `coordinate_transformation_mode` 的意义是一样的。若h/w方向的放缩因子为 `scale` ，输入坐标为 `x_in` ，输入尺寸为 `l_in` ，输出坐标为 `x_out` ，输出尺寸为 `l_out` ，则逆映射定义如下：

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
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，表示插值后的结果。数据类型与输入类型相同，形状根据缩放系数进行调整。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：支持的输入数据类型为FLOAT32/FLOAT16/INT8。
* BM1684X：支持的输入数据类型为FLOAT32/FLOAT16/INT8。

yuv2rgb
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行yuv转rgb格式，输入的Tensor要求shape=[n,h*3/2,w]，其中n为批个数，h为图像的像素高，w为图像的像素宽。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* inputs：Tensor类型，表示输入的yuv矩阵。必须是三维张量，第一维为批的个数，第二维为输入矩阵的高，第三维为输入矩阵的宽。
* src_format：Int类型，表示输入的格式。FORMAT_MAPPING_YUV420P_YU12=0，FORMAT_MAPPING_YUV420P_YV12=1，FORMAT_MAPPING_NV12=2，FORMAT_MAPPING_NV21=3。
* dst_format：Int类型，表示输出的格式。FORMAT_MAPPING_RGB=4，FORMAT_MAPPING_BGR=5。
* ImageOutFormatAttr：str型，目前只支持"UINT8"。
* formula_mode：string类型，表示使用的yuv2rgb转换公式，目前支持"_601_limited"、"_601_full"。
* round_mode：string类型，表示使用的舍入模式，目前支持"HalfAwayFromZero", "HalfToEven"。
* out_name：string类型，表示输出Tensor的名称，非必选，默认为None。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，表示转换出的rgb格式Tensor，shape=[n,3,h,w]，其中n为批个数，h为图像的像素高，w为图像的像素宽。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型是INT8/UINT8, 输出UINT8。
* BM1688：输入数据类型是INT8/UINT8，输出UINT8。

roiExtractor
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
给定4个featrue map，根据target_lvls索引从rois中抽取对应的roi，并与对应的featrue map做roi align，得到最终输出。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* rois：Tensor类型，表示所有的rois。
* target_lvls：Tensor类型，表示roi对应哪层feature map。
* feats：List[Tensor]型，表示多层feature map。
* PH：int类型，表示输出的height。
* PW：int类型，表示输出的width。
* sampling_ratio：int类型，表示每层feature map的sample ratio。
* list_spatial_scale：int，List[int]或Tuple[int]型，表示每层feature map对应的spatial scale。
                      请注意，spatial scale遵循mmdetection风格，最初给定一个整数值，但其浮点倒数最終被用于RoIAlign。
* mode: string类型, 表示Op执行模式, 目前支持DynNormal, DynFuse。
        请注意，在DynFuse模式下，输入rois的坐标  支持2类风格,1)遵循mmdetection的风格，即5长度[batch_id, x0, y0, x1, y1]。
                                                       2)自定义的7长度[a, b, x0, y0, x1, y1, c], 特別注意如果batch_id和a,b,c难以匹配, 建议另外重新生成batch_id。
               在DynNormal模式下，输入rois的坐标风格是一种自定义的7长度[a, b, x0, y0, x1, y1, c]风格，以便应用客户独特的模型。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，该Tensor的数据类型与输入rois相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。


Select Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nonzero
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def nonzero(tensor_i:Tensor,
                  dtype: str = 'int32',
                  out_name: str = None):
          #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
抽取输入Tensor data为true时对应的位置信息信息。
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* tensor_i：Tensor类型，表示输入操作Tensor。
* dtype：string型，表示输出数据类型，目前仅可使用默认值"int32"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型为INT32。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：输入数据类型可以是FLOAT32/FLOAT16。
* BM1684X：输入数据类型可以是FLOAT32/FLOAT16。

lut
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def lut(input: Tensor,
                table: Tensor,
                out_name: str = None):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入tensor进行查找表查找操作。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入。
* table：Tensor类型，表示查找表。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型与张量 `table` 的数据类型相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688： `input` 的数据类型可以是INT8/UINT8， `table` 的数据类型可以是INT8/UINT8。
* BM1684X： `input` 的数据类型可以是INT8/UINT8， `table` 的数据类型可以是INT8/UINT8。

select
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def select(lhs: Tensor,
                   rhs: Tensor,
                   tbrn: Tensor,
                   fbrn: Tensor,
                   type: str,
                   out_name = None):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据 `lhs` 与 `rhs` 的数值比较结果来选择，条件为真时，选择 `tbrn` ，条件为假时，选择 `fbrn` 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* lhs：Tensor类型，表示左边的张量。
* rhs：Tensor类型，表示右边的张量。
* tbrn：Tensor类型，表示条件为真时取的值。
* fbrn：Tensor类型，表示条件为假时取的值。
* type: string类型，表示比较符。可选项为"Greater"/"Less"/"GreaterOrEqual"/"LessOrEqual"/"Equal"/"NotEqual"。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

约束条件：要求 `lhs` 与 `rhs` 的形状和数据类型相同， `tbrn` 与 `fbrn` 的形状和数据类型相同。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型与张量 `tbrn`的数据类型相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688： `lhs` / `rhs` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。
* BM1684X： `lhs` / `rhs` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。


cond_select
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

        def cond_select(cond: Tensor,
                        tbrn: Union[Tensor, Scalar],
                        fbrn: Union[Tensor, Scalar],
                        out_name:str = None):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据条件 `cond` 来选择，条件为真时，选择 `tbrn` ，条件为假时，选择 `fbrn` 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* cond：Tensor类型，表示条件。
* tbrn：Tensor类型或Scalar类型，表示条件为真时取的值。
* fbrn：Tensor类型或Scalar类型，表示条件为假时取的值。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

约束条件：若 `tbrn` 和 `fbrn` 皆为张量，则要求 `tbrn` 与 `fbrn` 的形状和数据类型相同。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型与张量 `tbrn` 的数据类型相同。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688： `cond` / `tbrn` / `fbrn` 的数据类型可以是FLOAT32/FLOAT16(TODO)。
* BM1684X： `cond` / `tbrn` / `fbrn` 的输入数据类型可以是FLOAT32/FLOAT16(TODO)。


bmodel_inference_combine
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据生成的bmodel进行推理和逐层Tensor数据打印，配合 `npz_tool.py` 进行bmodel正确性验证。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* bmodel_file: String类型，表示bmodel绝对路径。
* final_mlir_fn: String类型，表示bmodel对应的final.mlir的绝对路径。
* input_data_fn: String类型或dict类型，表示输入数据的格式，支持 字典格式、.dat格式、.npz格式。
* tensor_loc_file: String类型，表示bmodel对应的tensor_location.json文件的绝对路径。
* reference_data_fn: String,类型，表示 `module.state = "TPU_LOWERED"`的.mlir文件或对应的.npz推理结果的绝对路径。bmodel推理时会将原本一个算子的shape拆散，该参数用于恢复原本的shape。
* dump_file: Bool类型，表示逐层Tensor数据是否以.npz文件形似保存，或直接返回字典。
* save_path: String类型，表示 `dump_file=True` 时的主机(host)端保存逐层推理的.npz文件的绝对路径。
* out_fixed: Bool类型，表示逐层Tensor数据输出是否保持为定点格式。
* dump_cmd_info: Bool类型，表示将当前bmodel中包含的所有原子指令对应的final.mlir的信息保存成txt文件，保存路径在save_path下。
* skip_check: Bool类型，启用此项可禁用数据对比，提高推理速度。soc模式下默认不进行数据对比。
* run_by_op: Bool类型，启用后按OP粒度运行，禁用时为按原子指令粒度运行。按OP粒度运行速度较快，但当一个OP中包含过多原子指令时可能会引发timeout错误。
* desire_op: List类型，其中当传入多个String类型的名字时，只会dump出给定名字的tensor。默认dump所有层tensor。
* is_soc: Bool类型，表示是否启用soc模式进行推理。
* using_memory_opt: Bool类型，启用后会减小在device端的内存消耗，但会增加耗时。推荐在大模型时启用。
* enable_soc_log: Bool类型，启用此项打印并在save_path下保存log日志。
* soc_tmp_path: String类型，表示soc模式下，板卡(device)端存放临时文件与推理工具的绝对路径。
* hostname: String类型，表示soc模式下，device端的ip地址。
* port: Int类型，表示soc模式下，device端的端口号。
* username: String类型，表示soc模式下，device端的用户名。
* password: String类型，表示soc模式下，device端的密码。

注意:

* 当使用pcie或soc模式进行逐层dump时，需先使用 `/tpu-mlir/envsetup.sh` 中的use_chip切换环境变量。当使用cmodel模式时，使用use_cmodel。
* 当使用soc模式时：reference_data_fn必须是.npz格式。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
* cmodel/pcie模式下：如果 `dump_file=True`，则在save_path下生成bmodel_infer_xxx.npz文件，否则返回python字典。
* soc模式下：在save_path下生成soc_infer_xxx.npz文件。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1688：  cmodel模式。
* BM1684X： cmodel/pcie/soc模式。


scatter
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def scatter(input: Tensor,
            index: Tensor,
            updates: Tensor,
            axis: int = 0,
            out_name: str = None):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据指定的索引，将输入数据写入目标Tensor的特定位置。该操作允许将更新输入Tensor的元素散布到输出Tensor的指定位置。请参考各大框架下的ScatterElements操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor，即需要更新的目标Tensor。
* index：Tensor类型，表示指定更新位置的索引Tensor。
* updates：Tensor类型，表示要写入目标Tensor的值。
* axis：int型，表示更新的轴。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。


返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个新的Tensor，该Tensor在指定位置上进行了更新操作，其他位置保持了原始输入的Tensor值。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32,FLOAT16,INT8。
* BM1688：输入数据类型可以是FLOAT32,FLOAT16,INT8。

scatterND
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""

    .. code-block:: python

      def scatterND(input: Tensor,
            indices: Tensor,
            updates: Tensor,
            out_name: str = None):
        #pass

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
根据指定的索引，将输入数据写入目标Tensor的特定位置。该操作允许将更新输入Tensor的元素散布到输出Tensor的指定位置。请参考ONNX 11下的ScatterND操作。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor，即需要更新的目标Tensor。
* indices：Tensor类型，表示指定更新位置的索引Tensor。 数据类型必须是uint32。
* updates：Tensor类型，表示要写入目标Tensor的值。Rank(updates) = Rank(input) + Rank(indices) - shape(indices)[-1] -1 。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。


返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个新的Tensor，该Tensor在指定位置上进行了更新操作，其他位置保持了原始输入的Tensor值。形状与数据类型和input一致。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32,FLOAT16,INT8。
* BM1688：输入数据类型可以是FLOAT32,FLOAT16,INT8。


Preprocess Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mean_std_scale
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor 进行预处理操作。
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。必须是4维或5维。
* std：List[float]类型，表示数据集的标准差。mean,std维度必须和input的channel维度一致,即input的第二维。
* mean: List[float]类型，表示数据集的均值。mean,std维度必须和input的channel维度一致，即input的第二维。
* scale: Optional[Union[List[float],List[int]]]类型或None，缩放系数。
* zero_points: Optional[List[int]]类型或None，表示零点。
* out_name：string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* odtype：String类型，表示接口输出Tensor数据类型。默认值为"float16"。目前支持float16， int8。
* round_mode：String类型，表示取整方法。默认值为"half_away_from_zero",范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型为odtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32/UINT8/INT8,输出类型可以为INT8/FLOAT16。


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rope
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入Tensor 进行旋转编码（RoPE）操作。
该操作属于 **全局操作**

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input：Tensor类型，表示输入操作Tensor。必须是4维。
* weight0: Tensor, 表示输入操作Tensor。
* weight1: Tensor, 表示输入操作Tensor。
* is_permute_optimize：bool类型, 表示是否做permute下沉，进行permute下沉shape的检查。# unused
* mul1_round_mode: String类型, 表示RoPE中mul1的取整方法。默认值为"half_away_from_zero",范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”，“half_up”，“half_down”。
* mul2_round_mode: String类型, 表示RoPE中mul2的取整方法。默认值为"half_away_from_zero",范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”，“half_up”，“half_down”。
* add_round_mode: String类型, 表示RoPE中add的取整方法。默认值为"half_away_from_zero",范围是“half_away_from_zero”，“half_to_even”，“towards_zero”，“down”，“up”，“half_up”，“half_down”。
* mul1_shift: int型，表示RoPE中mul1的移位的位数。
* mul2_shift: int型，表示RoPE中mul2的移位的位数。
* add_shift: int型，表示RoPE中add的移位的位数。
* mul1_saturation: bool 类型, 表示RoPE中的mul1计算结果是否需要饱和处理, 默认为True饱和处理, 非必要不修改。
* mul2_saturation: bool 类型, 表示RoPE中的mul2计算结果是否需要饱和处理, 默认为True饱和处理, 非必要不修改。
* add_saturation: bool 类型, 表示RoPE中的add计算结果是否需要饱和处理, 默认为True饱和处理, 非必要不修改。
* out_name: output name, string类型，默认为None。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型为odtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X：输入数据类型可以是FLOAT32,FLOAT16和INT类型。


multi_scale_deformable_attention
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入进行多尺度可变形注意力机制，具体功能可参考https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/multi_scale_deform_attn.py:MultiScaleDeformableAttention:forward，该操作的实现方式与官方有所不同。
目前只支持batch_size=1的情况。
该操作属于 **本地操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* query：Tensor类型，Transformer的查询张量，形状为 (1, num_query, embed_dims)。
* value：Tensor类型，值投影张量，形状为 (1, num_key, embed_dims)。
* key_padding_mask: Tensor类型，查询张量的mask，形状为 (1, num_key)。
* reference_points: Tensor类型，归一化的参考点，形状为 (1, num_query, num_levels, 2)，所有元素的范围在 [0, 1] 之间，左上角为 (0,0)，右下角为 (1,1)，包括填充区域。
* sampling_offsets_weight: Tensor类型，计算采样偏移量全连接层的权重，形状为 (embed_dims, num_heads\*num_levels\*num_points\*2)。
* sampling_offsets_bias_ori: Tensor类型，计算采样偏移量全连接层的偏置，形状为 (num_heads\*num_levels\*num_points\*2)。
* attention_weights_weight: Tensor类型，计算注意力权重全连接层的权重，形状为 (embed_dims, num_heads\*num_levels\*num_points)。
* attention_weights_bias_ori: Tensor类型，计算注意力权重全连接层的偏置，形状为 (num_heads\*num_levels\*num_points)。
* value_proj_weight: Tensor类型，计算值投影全连接层的权重，形状为 (embed_dims, embed_dims)。
* value_proj_bias_ori: Tensor类型，计算值投影全连接层的偏置，形状为 (embed_dims)。
* output_proj_weight: Tensor类型，计算输出投影全连接层的权重，形状为 (embed_dims, embed_dims)。
* output_proj_bias_ori: Tensor类型，计算输出投影全连接层的偏置，形状为 (embed_dims)。
* spatial_shapes: List[List[int]]类型，不同层级特征的空间形状，形状为 (num_levels, 2)，最后一个维度表示 (h, w)。
* embed_dims: int类型，查询、键、值的hidden_size。
* num_heads: int类型，注意力头数，默认值为8。
* num_levels: int类型，多尺度注意力的层级数，默认值为4。
* num_points: int类型，每个层级的采样点数，默认值为4。
* out_name: string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型为query.dtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X: 输入数据类型可以是FLOAT32,FLOAT16类型。
* BM1688: 输入数据类型可以是FLOAT32,FLOAT16类型。

Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a16matmul
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
对输入进行W4A16/W8A16 MatMul。
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* input:Tensor类型，表示输入tensor。
* weight: Tensor类型，表示4bits/8bits量化后权重，以int32类型存储。
* scale: Tensor类型，表示权重量化缩放因子，以float32类型存储。
* zp: Tensor类型，表示权重量化零点，以int32类型存储。
* bias: Tensor类型，表示偏置，以float32类型存储。
* right_transpose: Bool类型，表示权重矩阵是否转置，目前仅支持为True。
* out_dtype: string类型，表示输出张量的数据类型。
* out_name: string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。
* group_size: int类型，表示量化的group大小。
* bits: int类型，表示量化位宽，仅支持4bits/8bits。
* g_idx: Tensor类型，量化重排系数，目前不支持。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回一个Tensor，数据类型为out_dtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X: 输入数据类型可以是FLOAT32,FLOAT16类型。
* BM1688: 输入数据类型可以是FLOAT32,FLOAT16类型。


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qwen2_block
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
qwen2在prefill阶段的一个block layer。
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* hidden_states: Tensor类型，表示激活值，形状为 (1, seq_length, hidden_size)。
* position_ids: Tensor类型，表示位置索引，形状为 (3, 1, seq_length)。
* attention_mask: Tensor类型，表示注意力掩码，形状为 (1, 1, seq_length, seq_length)。
* q_proj_weights: Tensor类型，表示query量化后权重，以int32类型存储。
* q_proj_scales: Tensor类型，表示query量化缩放因子，以float32类型存储。
* q_proj_zps: Tensor类型，表示query量化零点，以int32类型存储。
* q_proj_bias: Tensor类型，表示query偏置，以float32类型存储。
* k_proj_weights: Tensor类型，表示key量化后权重，以int32类型存储。
* k_proj_scales: Tensor类型，表示key量化缩放因子，以float32类型存储。
* k_proj_zps: Tensor类型，表示key量化零点，以int32类型存储。
* k_proj_bias: Tensor类型，表示key偏置，以float32类型存储。
* v_proj_weights: Tensor类型，表示value量化后权重，以int32类型存储。
* v_proj_scales: Tensor类型，表示value量化缩放因子，以float32类型存储。
* v_proj_zps: Tensor类型，表示value量化零点，以int32类型存储。
* v_proj_bias: Tensor类型，表示value偏置，以float32类型存储。
* o_proj_weights: Tensor类型，表示输出投影层量化后权重，以int32类型存储。
* o_proj_scales: Tensor类型，表示输出投影层量化缩放因子，以float32类型存储。
* o_proj_zps: Tensor类型，表示输出投影层量化零点，以int32类型存储。
* o_proj_bias: Tensor类型，表示输出投影层偏置，以float32类型存储。
* down_proj_weights: Tensor类型，表示降维投影层量化后权重，以int32类型存储。
* down_proj_scales: Tensor类型，表示降维投影层量化缩放因子，以float32类型存储。
* down_proj_zps: Tensor类型，表示降维投影层量化零点，以int32类型存储。
* gate_proj_weights: Tensor类型，表示门投影层量化后权重，以int32类型存储。
* gate_proj_scales: Tensor类型，表示门投影层量化缩放因子，以float32类型存储。
* gate_proj_zps: Tensor类型，表示门投影层量化零点，以int32类型存储。
* up_proj_weights: Tensor类型，表示升维投影层量化后权重，以int32类型存储。
* up_proj_scales: Tensor类型，表示升维投影层量化缩放因子，以float32类型存储。
* up_proj_zps: Tensor类型，表示升维投影层量化零点，以int32类型存储。
* input_layernorm_weight: Tensor类型，表示对input做layernorm的权重，以int32类型存储。
* post_attention_layernorm_weight: Tensor类型，表示对attention层输出做layernorm的权重，以int32类型存储。
* cos: List[Tensor]型类型，表示cos位置编码。
* sin: List[Tensor]型类型，表示sin位置编码。
* out_dtype: string类型，表示输出张量的数据类型。
* group_size: int类型，表示量化的group大小。
* weight_bits: int类型，表示量化位宽，仅支持4bits/8bits。
* hidden_size: int类型，表示query/key/value的hidden_size。
* rms_norm_eps: float类型，表示layernorm中的eps参数。
* num_attention_heads: int类型，表示注意力头的个数。
* num_key_value_heads: int类型，表示key/value头的个数。
* mrope_section: List[int]类型，表示位置编码的三个维度大小。
* quant_method: str类型，表示量化方式，目前仅支持GPTQ量化。
* out_name: string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回3个Tensor，分别为激活输出、key cache、value cache，数据类型为out_dtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X: 输入数据类型可以是FLOAT32,FLOAT16类型。
* BM1688: 输入数据类型可以是FLOAT32,FLOAT16类型。


Transform Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qwen2_block_cache
:::::::::::::::::::::::::::::::::::::::::::::::::

接口定义
""""""""""""""""""""""""""""""""""""""""""""""
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

功能描述
""""""""""""""""""""""""""""""""""""""""""""""
qwen2在decode阶段的一个block layer。
该操作属于 **全局操作** 。

参数说明
""""""""""""""""""""""""""""""""""""""""""""""
* hidden_states: Tensor类型，表示激活值，形状为 (1, 1, hidden_size)。
* position_ids: Tensor类型，表示位置索引，形状为 (3, 1, 1)。
* attention_mask: Tensor类型，表示注意力掩码，形状为 (1, 1, 1, seq_length + 1)。
* k_cache: Tensor类型，表示key cache，形状为 (1, seq_length, num_key_value_heads, head_dim)。
* v_cache: Tensor类型，表示value cache，形状为 (1, seq_length, num_key_value_heads, head_dim)。
* q_proj_weights: Tensor类型，表示query量化后权重，以int32类型存储。
* q_proj_scales: Tensor类型，表示query量化缩放因子，以float32类型存储。
* q_proj_zps: Tensor类型，表示query量化零点，以int32类型存储。
* q_proj_bias: Tensor类型，表示query偏置，以float32类型存储。
* k_proj_weights: Tensor类型，表示key量化后权重，以int32类型存储。
* k_proj_scales: Tensor类型，表示key量化缩放因子，以float32类型存储。
* k_proj_zps: Tensor类型，表示key量化零点，以int32类型存储。
* k_proj_bias: Tensor类型，表示key偏置，以float32类型存储。
* v_proj_weights: Tensor类型，表示value量化后权重，以int32类型存储。
* v_proj_scales: Tensor类型，表示value量化缩放因子，以float32类型存储。
* v_proj_zps: Tensor类型，表示value量化零点，以int32类型存储。
* v_proj_bias: Tensor类型，表示value偏置，以float32类型存储。
* o_proj_weights: Tensor类型，表示输出投影层量化后权重，以int32类型存储。
* o_proj_scales: Tensor类型，表示输出投影层量化缩放因子，以float32类型存储。
* o_proj_zps: Tensor类型，表示输出投影层量化零点，以int32类型存储。
* o_proj_bias: Tensor类型，表示输出投影层偏置，以float32类型存储。
* down_proj_weights: Tensor类型，表示降维投影层量化后权重，以int32类型存储。
* down_proj_scales: Tensor类型，表示降维投影层量化缩放因子，以float32类型存储。
* down_proj_zps: Tensor类型，表示降维投影层量化零点，以int32类型存储。
* gate_proj_weights: Tensor类型，表示门投影层量化后权重，以int32类型存储。
* gate_proj_scales: Tensor类型，表示门投影层量化缩放因子，以float32类型存储。
* gate_proj_zps: Tensor类型，表示门投影层量化零点，以int32类型存储。
* up_proj_weights: Tensor类型，表示升维投影层量化后权重，以int32类型存储。
* up_proj_scales: Tensor类型，表示升维投影层量化缩放因子，以float32类型存储。
* up_proj_zps: Tensor类型，表示升维投影层量化零点，以int32类型存储。
* input_layernorm_weight: Tensor类型，表示对input做layernorm的权重，以int32类型存储。
* post_attention_layernorm_weight: Tensor类型，表示对attention层输出做layernorm的权重，以int32类型存储。
* cos: List[Tensor]型类型，表示cos位置编码。
* sin: List[Tensor]型类型，表示sin位置编码。
* out_dtype: string类型，表示输出张量的数据类型。
* group_size: int类型，表示量化的group大小。
* weight_bits: int类型，表示量化位宽，仅支持4bits/8bits。
* hidden_size: int类型，表示query/key/value的hidden_size。
* rms_norm_eps: float类型，表示layernorm中的eps参数。
* num_attention_heads: int类型，表示注意力头的个数。
* num_key_value_heads: int类型，表示key/value头的个数。
* mrope_section: List[int]类型，表示位置编码的三个维度大小。
* quant_method: str类型，表示量化方式，目前仅支持GPTQ量化。
* out_name: string类型或None，表示输出Tensor的名称，为None时内部会自动产生名称。

返回值
""""""""""""""""""""""""""""""""""""""""""""""
返回3个Tensor，分别为激活输出、key cache、value cache，数据类型为out_dtype。

处理器支持
""""""""""""""""""""""""""""""""""""""""""""""
* BM1684X: 输入数据类型可以是FLOAT32,FLOAT16类型。
* BM1688: 输入数据类型可以是FLOAT32,FLOAT16类型。
