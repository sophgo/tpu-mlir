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

2. 前端准备

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

3. 参数结构体解析

  用户需要根据算子所需的参数实现相应的函数用于解析由工具链前端传递过来的参数。工具链前端的参数是通过 `custom_param_t` 数组的指针进行传递，其中，数组的第一个元素是保留的，从第二个元素开始，每个元素对应前端的一个属性，每个 `custom_param_t` 结构体中包含了一个参数的信息，参数值会被存放在相应数据类型（其中包含了整数，浮点数，整数数组与浮点数数组）的 `custom_param_t` 成员变量中。参数的顺序与用户在调用tpulang接口时提供的参数顺序相同。 `custom_param_t` 结构体的定义如下：

  .. code-block:: c

    typedef union {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;

4. 编译器补丁

  有时候，需要对编译器进行修改，以对不同的自定义算子在不同参数下的编译行为进行控制，这时候就需要添加一些补丁。现时，已支持以下补丁函数(在文件夹 ./plugin中定义)：

  a. 强制动态运行。在某些场合，如形状动态变化时，需要强制动态运行。补丁函数形式如下：

  .. code-block:: c

    bool force_dynamic_run_{op_name}(void* param, int param_size);

  若不提供该函数，则即使在静态编译模式下，仍默认{op_name}对应的算子必须静态运行。

  b. 是否支持与其他算子融合。补丁函数形式如下：

  .. code-block:: c

    bool local_gen_support_{op_name}(void* param, int param_size);

  若不提供该函数，则默认{op_name}对应的算子不支持与其他算子融合。否则，需要在允许的参数情形下实现 `api_xxx_local`和 `api_xxx_local_bfsz`(可选)接口。

  c. 在与其他算子融合时，是否允许对轴axis进行切割。补丁函数形式如下：

  .. code-block:: c

    bool allow_data_split_{op_name}(void* param, int param_size, int axis, group_type_t group_type);

  若不提供该函数，则默认与其他算子融合时，{op_name}对应的算子允许对所有的轴进行切割。

  d. 切片反向推导函数。补丁函数形式如下：

  .. code-block:: c

    bool backwardh_{op_name}(void* param, int param_size, int* in_idx, int* in_slice, int out_idx, int out_slice);

    bool backwardw_{op_name}(void* param, int param_size, int* in_idx, int* in_slice, int out_idx, int out_slice);

  其中，in_idx和in_slice分别表示指向该层输入张量切片的索引和大小的指针，out_idx和out_slice表示该层输出张量切片的索引索引和大小。若不提供该函数，则in_idx指向的数值与out_idx相同，in_slice指向的数值与out_slice相同。

  e. 推理函数。此补丁函数用于TOP层和TPU层的数据比对，必须实现。补丁函数形式如下：

  .. code-block:: c

    void inference_absadd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
      const int* input_dims, const float** inputs, float** outputs);

  其中，input_shapes为输入张量形状的数组，input_dims为输入张量维度的数组。inputs表示输入张量的指针数组，outputs表示输出张量的指针数组。

  e. 形状推断函数。此补丁函数用于TOP层形状推断，若不实现，默认只有一个输入一个输出，且输出形状跟输入形状相同。补丁函数形式如下：

  .. code-block:: c

    void shape_inference_absadd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
      const int* input_dims, int (*output_shapes)[MAX_SHAPE_DIMS], int* output_dims);

  其中，input_shapes/output_shapes为输入/出张量形状的数组，input_dims/output_dims为输入/出张量维度的数组。

5. 基于tpu-kernel编写后端算子

  假定当前处于 $TPUC_ROOT/customlayer 路径下，在./include/tpu_impl_custom_ops.h 头文件中，声明global layer与local layer的自定义算子函数 void tpu_impl_{op_name}_global 和 void tpu_impl_{op_name}_local (可选) ， 并在 ./src 下添加 tpu_impl_{op_name}.c 文件，在其中调用tpu-kernel接口实现相应的函数。

6. 定义算子参数结构体与编写算子调用接口

  注意: 在下文中, {op_name} 表示算子的名字, 且字符串长度应不超过 20 。{processor_arch} 表示架构名称，当前可选 `bm1684x` 和 `bm1686` 。

  在 ./src 目录下添加自定义算子函数的调用接口:

    void api_{op_name}_global 和 api_{op_name}_local (可选) (分别用于调用 void tpu_impl_{op_name}_global 和 void tpu_impl_{op_name}_local)

    int64_t api_{op_name}_global_bfsz (可选，计算global layer需要的缓存大小)

    int api_{op_name}_local_bfsz (可选，计算local layer需要的缓存大小)

    void type_infer_{op_name} (动态时使用，从输入的形状和数据类型推理出输出的形状和数据类型，若不实现，则默认只有一个输入和一个输出，且输出的形状和数据类型与输入的形状和数据类型相同)

    void slice_infer_{op_name} (动态时使用，从输入的切片推理出输出的切片，若不实现，则默认只有一个输入和一个输出，且输出的切片与输入的切片相同)

7. 定义后端调用接口

  在 register_ops.cmake 中添加算子的名字以注册自定义算子：

  .. code-block:: shell

    register_custom_op({op_name})

  假如自定义算子存在local layer，则需要注册一下:

  .. code-block:: shell

    register_custom_local_op({op_name})

  假如自定义算子global layer需要缓存，则需要注册一下:

  .. code-block:: shell

    register_custom_global_bfsz({op_name})

  假如自定义算子local layer需要缓存，则需要注册一下:

  .. code-block:: shell

    register_custom_local_bfsz({op_name})


8. 编译并安装动态库

  先初始化环境：

  .. code-block:: shell

    source $TPUC_ROOT/customlayer/envsetup.sh

  然后需要完成补丁的编译（得到 `libplugin_custom.so`）：

  .. code-block:: shell

    rebuild_custom_plugin

  自定义算子后端接口的编译（得到 `libbackend_custom.so`）：

  .. code-block:: shell

    rebuild_custom_backend

  之后根据实际使用场景编译对应的固件：

  a. CMODEL模式（得到 `libcmodel_custom_xxx.so`）

  .. code-block:: shell

    rebuild_custom_firmware_cmodel {processor_arch}

  b. SOC模式（得到 `libxxx_kernel_module_custom_soc.so`）

  .. code-block:: shell

    rebuild_custom_firmware_soc {processor_arch}

  c. PCIE模式（得到 `libxxx_kernel_module_custom_pcie.so`）

  .. code-block:: shell

    rebuild_custom_firmware_pcie {processor_arch}

9. 上卡测试
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

后面步骤与TpuLang自定义算子添加中的第3-9步相同，此处不再赘述。

自定义算子示例
--------------

本节内容假定已经完成了tpu-mlir发布包加载。

TpuLang示例
~~~~~~~~~~~~

本小节提供了一个swapchanel算子实现与通过TpuLang接口应用的样例。

1. 算子参数解析

  在文件 ${TPUC_ROOT}/customlayer/include/backend_custom_param.h 中定义参数结构体 `swapchannel_param_t`：

  .. code-block:: c

    typedef struct swapchannel_param {
      int order[3];
    } swapchannel_param_t;

  其中，这里的字段order对应前端的属性order 。

  值得注意的是，从编译器传递到后端的是一个 `custom_param_t` 的数组A，它的第一个元素是保留的，从第二个元素开始，每个元素对应前端的一个属性。为方便起见，在头文件 {TPUC_ROOT}/customlayer/include/api_common.h 中，提供了一个宏来完成了一个对应： `PARSE_PARAM(swapchannel, sc_param, param)` , 其中， `param` 表示数组A， `sc_param` 表示后端参数结构体。用户需要在文件 ${TPUC_ROOT}/customlayer/include/param_parser.h 中定义一个swapchannel_parse_param解析函数来完成这种转换，其输入实际上是数组A的剔除第一个元素后的子数组的指针。 在文件 ${TPUC_ROOT}/customlayer/include/param_parser.h 中，实现参数解析代码：

  .. code-block:: c

    static swapchannel_param_t swapchannel_parse_param(const void* param) {
        swapchannel_param_t sc_param = {0};
        for (int i = 0; i < 3; i++) {
            sc_param.order[i] = ((custom_param_t *)param)[0].int_arr_t[i];
        }
        return sc_param;
    }

  参数解析在补丁函数和后端实现中都会被用到。

2. 补丁函数

  在文件 ${TPUC_ROOT}/customlayer/plugin/plugin_swapchannel.c 中：

  .. code-block:: c

    #include <string.h>
    #include <assert.h>
    #include "param_parser.h"

    void inference_swapchannel(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
      const int* input_dims, const float** inputs, float** outputs) {
      PARSE_PARAM(swapchannel, sc_param, param);
      int in_num = 1;
      for (int i = 2; i < input_dims[0]; ++i) {
        in_num *= input_shapes[0][i];
      }
      int N = input_shapes[0][0];
      int C = input_shapes[0][1];
      assert(C == 3);
      for (int n = 0; n < N; ++n) {
        for (int c = 0; c < 3; ++c) {
          for (int x = 0; x < in_num; ++x) {
            memcpy(outputs[0] + n * C * in_num + sc_param.order[c] * in_num,
                  inputs[0] + n * C * in_num + c * in_num, in_num * sizeof(float));
          }
        }
      }
    }

3. 后端接口

  在文件 ${TPUC_ROOT}/customlayer/src/interface_swapchannel.c 中定义函数 `void type_infer_swapchannel`和 `void api_swapchannel_global`：

  .. code-block:: c

    #include <string.h>
    #include "tpu_utils.h"
    #include "tpu_impl_custom_ops.h"
    #include "param_parser.h"

    // type infer function
    void type_infer_swapchannel(
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

4. 后端算子实现

  在 ${TPUC_ROOT}/customlayer/include/tpu_impl_custom_ops.h 头文件中添加如下声明：

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

5. 后端算子注册

  在文件 ${TPUC_ROOT}/customlayer/register_ops.cmake 添加如下代码，可用于注册后端算子：

  .. code-block:: c

    register_custom_op(swapchannel)

  完成后，可参看《TpuLang自定义算子添加》中第5步中列出的命令进行编译。


6. 前端准备

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
