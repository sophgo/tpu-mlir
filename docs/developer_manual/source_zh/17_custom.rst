用户自定义算子
===============

概述
-------
tpu-mlir当前已经包含了丰富的算子库，可以满足大部分神经网络模型的编译需求。但在某些场景下，
可能需要用户自定义算子来实现对张量的计算。如：

1. tpu-mlir 还未支持的算子，且无法通过其它算子组合实现
2. 算子为用户私有，未对公众开源
3. 使用多个算子 API 组合无法取得最佳计算性能，直接从 tpu-kernel 层自定义运算可以提高运行效率

自定义算子功能允许用户自由使用tpu-kernel中的接口，自定义tensor在tpu上的计算，并将这一计算过程封装为后端算子（后端算子开发请参考TPU-KERNEL开发参考手册）。其中，后端算子计算涉及到global layer与local layer相关操作：

  a. 算子必须实现 global layer ，global layer 的输入和输出数据都放在 ddr 中，数据需要
  从 global mem 搬运到 local mem 中执行运算，再将结果搬运至 global mem。其优点是
  local mem 可以任意使用，比较灵活；缺点是会产生较多的 gdma 搬运，tpu 利用率较
  低。

  b. 算子根据需要实现 local layer，local layer 的输入和输出的数据都放在 local mem 中，
  可以与其他 layer 组合进行 layer group 优化，避免该 layer 计算时数据要搬入搬出到
  global mem 中。其优点是可以节省 gdma 搬运, 运算效率高；缺点是比较复杂，local
  mem 在模型部署时会提前分配好，不可任意使用，在部分算子中无法实现。

  c. 用户还需要实现自定义算子的形状推断函数，以根据输入数据类型和形状完成对输出类型和形状的推断。

完成后端算子的封装后，前端可以通过tpulang或caffe构建包含自定义算子的模型,并最终通过 tpu-mlir 的模型转换接口完成模型部署。本章主要介绍在tpu-mlir发布包中使用自定义算子的流程。


自定义算子添加流程
---------------------------

注意: 在下文中, {op_name} 表示算子的名字, 且字符串长度应不超过 20 。{processor_arch} 表示架构名称，当前可选 `bm1684x` 和 `bm1688` 。

TpuLang自定义算子添加
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 加载tpu-mlir

.. include:: ./../../quick_start/source_zh/env_var.rst


2. 基于tpu-kernel编写后端算子

  假定当前处于 $TPUC_ROOT/customlayer 路径下，在./include/tpu_impl_custom_ops.h 头文件中，声明global layer与local layer的自定义算子函数 void tpu_impl_{op_name}_global 和 void tpu_impl_{op_name}_local (可选) ， 并在 ./src 下添加 tpu_impl_{op_name}.c 文件，在其中调用tpu-kernel接口实现相应的函数。

3. 定义算子参数结构体与编写算子调用接口

  a. 在 ./src 目录下添加自定义算子函数的调用接口:

    void api_{op_name}_global 和 api_{op_name}_local (可选) (分别用于调用 void tpu_impl_{op_name}_global 和 void tpu_impl_{op_name}_local)

    void shape_infer_{op_name} (从输入的形状和数据类型推理出输出的形状和数据类型)

  b. 同时，用户需要根据算子所需的参数实现相应的函数用于解析由工具链前端传递过来的参数。工具链前端的参数是通过 `custom_param_t` 数组的指针进行传递，其中，数组的第一个元素是保留的，从第二个元素开始，每个元素对应前端的一个属性，每个 `custom_param_t` 结构体中包含了一个参数的信息，参数值会被存放在相应数据类型（其中包含了整数，浮点数，整数数组与浮点数数组）的 `custom_param_t` 成员变量中。参数的顺序与用户后续在调用tpulang接口（第6步）时提供的参数顺序相同。 `custom_param_t` 结构体的定义如下：

  .. code-block:: c

    typedef union {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;


4. 注册自定义算子

  在 register_ops.cmake 中添加算子的名字以注册自定义算子：

  .. code-block:: shell

    register_custom_op({op_name})


5. 编译并安装动态库

  先初始化环境：

  .. code-block:: shell

    source $TPUC_ROOT/customlayer/envsetup.sh


  然后需要完成自定义算子后端接口的编译（得到 `libbackend_custom.so`）：

  .. code-block:: shell

    rebuild_custom_backend

  之后根据实际使用场景编译对应的固件：

  a. CMODEL模式（得到 `libfirmware_custom_xxx.so`）

  .. code-block:: shell

    rebuild_custom_firmware_cmodel {processor_arch}

  b. SOC模式（得到 `libxxx_kernel_module_custom_soc.so`）

  .. code-block:: shell

    rebuild_custom_firmware_soc {processor_arch}

  c. PCIE模式（得到 `libxxx_kernel_module_custom_pcie.so`）

  .. code-block:: shell

    rebuild_custom_firmware_pcie {processor_arch}


  至此我们就完成了自定义算子后端部分的工作。


6. 利用TpuLang构建自定义算子

  关于TpuLang的使用方式请参考TpuLang接口章节。

  TpuLang中提供了 `TpuLang.custom` 接口用于在工具链前端构建自定义算子（请保证 `op_name` 部分与后端算子的名称一致）。需要注意的是， `params` 是一个python字典，其中，键 *必须* 是字符串，表示属性的名字，值 *只能* 是整数/浮点数，或整数/浮点数的列表(要求长度不超过16)。在构建神经网络时，对于同一个自定义算子，键的个数和排列顺序必须保持一致；对于同一个键，如果它的值是个列表，则列表的长度也必须保持一致。

  .. code-block:: python

    TpuLang.custom(tensors_in: List[TpuLang.Tensor],
                   shape_func,
                   op_name: str,
                   out_dtypes: List[str],
                   out_names: List[str] = None,
                   params: dict = None)
                   -> List[TpuLang.Tensor]
    '''
        自定义算子
        参数:
            tensors_in: 输入张量的列表 (包括权重)
            shape_func: 形状推理函数，输入为输入张量的形状的列表，
                        输出为输出张量的形状的列表
            op_name: 自定义算子的名字
            out_dtypes: 输出张量的数据类型的列表
            out_names: 输出张量的名字列表
            params: 自定义算子的属性字典

        Return:
            tensors_out: 输出张量的列表
    '''

  a. 定义自定义算子的tpulang接口

  为方便起见，可在 $TPUC_ROOT/customlayer/python/my_tpulang_layer.py 中标准化定义你的自定义算子，格式如下：

  .. code-block:: python

    class xxx:
      @staticmethod
      def native(...):
          ...
          return ...
      @staticmethod
      def tpulang(inputs, ...):
          def shape_func(tensors_in:list, ...):
              ...
              return ...
          params = dict(...)
          outputs = tpul.custom(
              tensors_in=inputs,
              shape_func=shape_func,
              op_name=...,
              params=params,
              out_dtypes=...)
          return outputs

  其中， `native` 函数可用于计算输出张量的参考数据，而 `tpulang` 函数则用 `TpuLang.custom` 函数来构建自定义层。

  b. 单元测试

  在定义完自定义算子前端接口后，需要测试一下这个接口是否可靠。在目录 `$TPUC_ROOT/customlayer/test_if/unittest` 下，创建名为"test_xxx.py"的python文件，并创建一个测试类，继承 `TestTPULangCustom` 类，并创建名为"test_xxx"的方法来测试你的自定义算子。

  通过运行以下shell命令来进行自定义算子的单元测试：

  .. code-block:: shell

    run_custom_unittest {processor_arch}


7. 上卡测试
  当网络中存在自定义动态算子时，bmodel中包含的固件可能无法使bmrt_test正常工作，这时就需要替换固件了，使用shell命令可以达到这一目标：

  .. code-block:: shell

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_soc.so #SOC模式下

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_pcie.so #PCIE模式下


Caffe自定义算子添加
~~~~~~~~~~~~~~~~~~~~~

1. 定义Caffe的自定义算子

  要定义 Caffe 的自定义算子，你需要在$TPUC_ROOT/customlayer/python/my_caffe_layer.py 文件中定义一个类，该类继承自 caffe.Layer，并根据需要重写 `setup`, `reshape`, `forward` 和 `backward` 函数。

2. 实现自定义算子前端转换函数

  通过Python实现的自定义算子的caffe层类型为 "Python"，需要在 $TPUC_ROOT/customlayer/python/my_converter.py 中的 `MyCaffeConverter` 类里根据之前的自定义算子定义一个针对caffe层类型为 "Python" 的前端算子转换函数。完成转换函数后便可通过 `MyCaffeConverter` 对包含自定义算子的Caffe模型进行前端转换。

  定义完成后，可以调用my_converter.py接口进行Top MLIR转换:

  .. code-block:: shell

    my_converter.py \
    --model_name # the model name \
    --model_def # .prototxt file \
    --model_data # .caffemodel file \
    --input_shapes # list of input shapes (e.g., [[1,2,3],[3,4,5]]) \
    --mlir # output mlir file

后面步骤与TpuLang自定义算子添加中的第2-6步相同，此处不再赘述。

自定义算子示例
--------------

本节内容假定已经完成了tpu-mlir发布包加载。

TpuLang示例
~~~~~~~~~~~~

本小节提供了一个swapchanel算子实现与通过TpuLang接口应用的样例。

1. 算子参数解析与后端接口

  在文件 ${TPUC_ROOT}/customlayer/include/backend_custom_param.h 中定义参数结构体 `swapchannel_param_t`：

  .. code-block:: c

    typedef struct swapchannel_param {
      int order[3];
    } swapchannel_param_t;

  其中，这里的字段order对应前端的属性order 。

  值得注意的是，从编译器传递到后端的是一个 `custom_param_t` 的数组A，它的第一个元素是保留的，从第二个元素开始，每个元素对应前端的一个属性。为方便起见，在头文件 {TPUC_ROOT}/customlayer/include/api_common.h 中，提供了一个宏来完成了一个对应： `PARSE_PARAM(swapchannel, sc_param, param)` , 其中， `param` 表示数组A， `sc_param` 表示后端参数结构体。用户需要在文件 ${TPUC_ROOT}/customlayer/include/param_parser.h 中定义一个swapchannel_parse_param解析函数来完成这种转换，其输入实际上是数组A的剔除第一个元素后的子数组的指针：

  .. code-block:: c

    static swapchannel_param_t swapchannel_parse_param(const void* param) {
        swapchannel_param_t sc_param = {0};
        for (int i = 0; i < 3; i++) {
            sc_param.order[i] = ((custom_param_t *)param)[0].int_arr_t[i];
        }
        return sc_param;
    }

  接着，在文件 ${TPUC_ROOT}/customlayer/src/interface_swapchannel.c 中定义函数void shape_infer_swapchannel和void api_swapchannel_global：

  .. code-block:: c

    #include <string.h>
    #include "tpu_utils.h"
    #include "tpu_impl_custom_ops.h"
    #include "param_parser.h"

    // shape infer function
    void shape_infer_swapchannel(
        const global_tensor_spec_t *input,
        global_tensor_spec_t *output,
        const void *param) {
        output->dtype = input->dtype;
        output->dims = input->dims;
        memcpy(output->shape, input->shape, output->dims);
        output->elem_num = input->elem_num;
    }

    // global api function
    void api_swapchannel_global(
        const global_tensor_spec_t *input,
        global_tensor_spec_t *output,
        const void *param) {
        PARSE_PARAM(swapchannel, sc_param, param);
        tpu_impl_swapchannel_global(
            input->addr,
            output->addr,
            input->shape,
            sc_param.order,
            tpu_type_convert(input->dtype));
    }

2. 后端算子实现

  ${TPUC_ROOT}/customlayer/include/tpu_impl_custom_ops.h 头文件添加如下声明：

  .. code-block:: c

    void tpu_impl_swapchannel_global(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int *shape,
        const int *order,
        data_type_t dtype);


  ${TPUC_ROOT}/customlayer/src/tpu_impl_swapchannel.c 代码如下：

  .. code-block:: c

    #include "tpu_impl_custom_ops.h"

    void tpu_impl_swapchannel_global(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int *shape,
        const int *order,
        data_type_t dtype)
    {
        dim4 channel_shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
        dim4 stride = {0};
        stride.w = 1, stride.h = channel_shape.w;
        stride.c = stride.h * channel_shape.h;
        stride.n = stride.c * channel_shape.c;
        channel_shape.c = 1;
        int data_size = tpu_data_type_size(dtype);
        int offset = channel_shape.w * channel_shape.h * data_size;
        for (int i = 0; i < 3; i++) {
            tpu_gdma_cpy_S2S(
                output_global_addr + i * offset,
                input_global_addr + order[i] * offset,
                &channel_shape,
                &stride,
                &stride,
                dtype);
        }
    }

3. 后端算子注册

  ${TPUC_ROOT}/customlayer/register_ops.cmake 添加如下代码，可用于注册后端算子：

  .. code-block:: c

    register_custom_op(swapchannel)

  完成后，可参看《TpuLang自定义算子添加》中第5步中列出的命令进行编译。


4. 前端准备

  在文件 ${TPUC_ROOT}/customlayer/python/my_tpulang_layer.py 中调用TpuLang接口构建自定义算子swapChannel， 它只有一个输入和一个输出，且有一个属性order，是一个长度为3的整数列表：

   .. code-block:: python

      import transform.TpuLang as tpul

      class swapChannel:
          @staticmethod
          def native(data):
              return data[:, [2, 1, 0], :, :]
          @staticmethod
          def tpulang(inputs, dtype="float32"):
              def shape_func(tensors_in:list):
                  return [tensors_in[0].shape]
              params = {"order": [2, 1, 0]}
              outs = tpul.custom(
                  tensors_in=inputs,
                  shape_func=shape_func,
                  # op_name should be consistent with the backend
                  op_name="swapchannel",
                  params=params,
                  out_dtypes=[dtype])
              return outs

  在文件 ${TPUC_ROOT}/customlayer/test_if/unittest/test_swapchannel.py 中, 对自定义的swapChannel算子进行单元测试：

   .. code-block:: python

      import numpy as np
      import unittest
      from tpulang_custom_test_base import TestTPULangCustom
      import transform.TpuLang as tpul
      import my_tpulang_layer

      class TestSwapChannel(TestTPULangCustom):
          def _test(self, dtype):
              shape = [4, 32, 36, 36]
              self.data_in = np.random.random(shape).astype(dtype)
              x = tpul.Tensor(name="in", dtype=dtype, shape=shape, data=self.data_in)
              y = my_tpulang_layer.swapChannel.tpulang(inputs=[x],
                    dtype=dtype)[0]
              self.compile('SwapChannel', [x], [y], dtype)
          def test_fp32(self):
              self._test('float32')
          def test_fp16(self):
              self._test('float16')

      if __name__ == '__main__':
          unittest.main()


Caffe示例
~~~~~~~~~~

本小节提供了Caffe中 absadd 和 ceiladd 自定义算子的应用示例。

1. 定义 Caffe 自定义算子

  absadd 和 ceiladd 在 $TPUC_ROOT/customlayer/python/my_caffe_layer.py 中的定义如下：

  .. code-block:: python

    import caffe
    import numpy as np

    # Define the custom layer
    class AbsAdd(caffe.Layer):

        def setup(self, bottom, top):
            params = eval(self.param_str)
            # define attributes here
            self.b_val = params['b_val']

        def reshape(self, bottom, top):
            top[0].reshape(*bottom[0].data.shape)

        def forward(self, bottom, top):
            top[0].data[...] = np.abs(np.copy(bottom[0].data)) + self.b_val

        def backward(self, top, propagate_down, bottom):
            pass

    class CeilAdd(caffe.Layer):

        def setup(self, bottom, top):
            params = eval(self.param_str)
            # define attributes here
            self.b_val = params['b_val']

        def reshape(self, bottom, top):
            top[0].reshape(*bottom[0].data.shape)

        def forward(self, bottom, top):
            top[0].data[...] = np.ceil(np.copy(bottom[0].data)) + self.b_val

        def backward(self, top, propagate_down, bottom):
            pass

  Caffe prototxt 中相应算子的表达如下：

  .. code-block:: text

    layer {
      name: "myabsadd"
      type: "Python"
      bottom: "input0_bn"
      top: "myabsadd"
      python_param {
        module: "my_caffe_layer"
        layer: "AbsAdd"
        param_str: "{ 'b_val': 1.2}"
      }
    }

    layer {
      name: "myceiladd"
      type: "Python"
      bottom: "input1_bn"
      top: "myceiladd"
      python_param {
        module: "my_caffe_layer"
        layer: "CeilAdd"
        param_str: "{ 'b_val': 1.5}"
      }
    }


2. 实现算子前端转换函数

  在 $TPUC_ROOT/customlayer/python/my_converter.py 中的 MyCaffeConverter 类里定义一个 convert_python_op 函数，代码如下：

  .. code-block:: python

    def convert_python_op(self, layer):
        assert (self.layerType(layer) == "Python")
        in_op = self.getOperand(layer.bottom[0])
        p = layer.python_param

        dict_attr = dict(eval(p.param_str))
        params = dict_attr_convert(dict_attr)

        # p.layer.lower() to keep the consistency with the backend op name
        attrs = {"name": p.layer.lower(), "params": params, 'loc': self.get_loc(layer.top[0])}

        # The output shape compute based on reshape function in my_caffe_layer
        out_shape = self.getShape(layer.top[0])
        outs = top.CustomOp([self.mlir.get_tensor_type(out_shape)], [in_op],
                            **attrs,
                            ip=self.mlir.insert_point).output
        # add the op result to self.operands
        self.addOperand(layer.top[0], outs[0])

3. Caffe前端转换

  通过调用 my_converter.py 接口完成对 $TPUC_ROOT/customlayer/test 目录下的 my_model.prototxt, my_model.caffemodel Caffe模型进行转换 （该Caffe模型中包含了absadd与ceiladd算子），命令如下：

  .. code-block:: shell

    my_converter.py \
    --model_name caffe_test_net \
    --model_def $TPUC_ROOT/customlayer/test/my_model.prototxt \
    --model_data $TPUC_ROOT/customlayer/test/my_model.caffemodel \
    --input_shapes [[1,3,14,14],[1,3,24,26]] \
    --mlir caffe_test_net.mlir

  通过以上步骤可获得caffe_test_net.mlir的Top层mlir文件，后续的模型部署过程请参考用户接口章节。

4. 后端算子与接口实现

  absadd 与 ceiladd 的实现部分和 swapchannel 算子相似，可在 $TPUC_ROOT/customlayer/include 和  $TPUC_ROOT/customlayer/src 目录下找到相应代码。
