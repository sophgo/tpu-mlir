用户自定义算子
===============

概述
-------
tpu-mlir当前已经包含了丰富的算子库，可以满足大部分神经网络模型的编译需求。但在某些场景下，
可能需要用户自定义算子来实现对 Tensor 的计算。如：

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

完成后端算子的封装后，前端可以通过tpulang或caffe构建包含自定义算子的模型,并最终通过 tpu-mlir 的模型转换接口完成模型部署。本章主要介绍在tpu-mlir发布包中使用自定义算子的流程。


自定义算子添加流程
---------------------------

TpuLang自定义算子添加
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 加载tpu-mlir

.. include:: ./../../quick_start/source_zh/env_var.rst

2. 基于tpu-kernel编写后端算子

  假定当前处于 $TPUC_ROOT/customlayer 路径下，在./include目录下添加 nodechip_{op_name}.h 头文件，声明global layer与local layer的自定义算子函数 void nodechip_{op_name}_global 和 void nodechip_{op_name}_local ， 并在 ./src 下添加 nodechip_{op_name}.c 文件，在其中调用tpu-kernel接口实现相应的函数。

3. 定义算子参数结构体与编写算子调用接口

  a. 根据算子所需的参数在 ./include/backend_custom_param.h 头文件中添加相应的结构体 {op_name}_param_t ，用于接收来自工具链前端的参数。

  b. 在 ./include 目录下添加 api_{op_name}.h 头文件，声明自定义算子函数的调用接口 void api_{op_name}_global 和 api_{op_name}_local（分别用于调用 void nodechip_{op_name}_global 和 void nodechip_{op_name}_local ），并在 ./src 目录下添加 api_{op_name}.c 文件实现相应的接口函数。

  c. 同时，用户需要根据算子所需的参数实现相应的函数用于解析由工具链前端传递过来的参数。工具链前端的参数是通过custom_param_t数组的指针进行传递，每个custom_param_t结构体中包含了一个参数的信息，参数值会被存放在相应数据类型（其中包含了整数，浮点数，整数数组与浮点数数组）的custom_param_t成员变量中。参数的顺序与用户在调用tpulang接口时提供的参数顺序相同。custom_param_t结构体的定义如下：

  .. code-block:: c

    typedef struct {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;


4. 定义后端调用接口

  在 ./src/backend_custom_api.cpp 中通过宏定义建立后端接口，该接口将在工具链前端进行Codegen时被调用。格式如下：

  .. code-block:: C

    IMPL_CUSTOM_API_GLB({op_name}, {op_name}_param_t)

    IMPL_CUSTOM_API_LOC({op_name}, {op_name}_param_t)


5. 编译并安装动态库

  通过运行 $TPUC_ROOT/customlayer/build.sh 脚本，便可以完成自定义算子的编译，生成 backend_custom_api.so 动态库并安装到 $TPUC_ROOT/lib 中。

6. 调用TpuLang构建模型

  关于TpuLang的使用方式请参考TpuLang接口章节。

  TpuLang中提供了 TpuLang.custom 接口用于在工具链前端构建自定义算子（请保证op_name部分与后端算子的名称一致）：

  .. code-block:: python

    TpuLang.custom(tensors_in: list,
                    shape_func,
                    op_name: str,
                    out_dtypes: list,
                    out_names: list = None,
                    params: dict = None)
    '''
        The custom op
        Arguments:
            tensors_in: list of input tensors (including weight tensors)
            shape_func: function for doing shape inference, taking tensors_in as the
                        parameter, return is the list of output tensors shape
            op_name: name of the custom operator,
            out_dtypes: list of outputs' data type
            out_name: list of output names
            params: parameters of the custom op

        Return:
            tensors_out: list of output tensors
    '''

Caffe自定义算子添加
~~~~~~~~~~~~~~~~~~~~~

1-5步TpuLang自定义算子添加中相同，此处不再赘述。

6. 定义Caffe的自定义算子

  要定义 Caffe 的自定义算子，你需要在$TPUC_ROOT/customlayer/python/ my_layer.py 文件中定义一个类，该类继承自 caffe.Layer，并根据需要重写 setup, reshape, forward 和 backward 函数。


7. 实现自定义算子前端转换函数

  通过Python实现的自定义算子type为 "Python"，需要在 $TPUC_ROOT/customlayer/python/ my_converter.py 中的 MyCaffeConverter 类里根据之前的自定义算子定义一个针对type为 "Python" 的前端算子转换函数。完成转换函数后便可通过 MyCaffeConverter 对包含自定义算子的Caffe模型进行前端转换。

  定义完成后，可以调用my_converter.py接口进行Top MLIR转换:

  .. code-block:: shell

    my_converter.py \
    --model_name # the model name \
    --model_def # .prototxt file \
    --model_data # .caffemodel file \
    --input_shapes # list of input shapes (e.g., [[1,2,3],[3,4,5]]) \
    --mlir # output mlir file

自定义算子示例
--------------

本节内容假定已经完成了tpu-mlir发布包加载。

TpuLang示例
~~~~~~~~~~~~

本小节提供了一个swapchanel算子实现与通过TpuLang接口应用的样例。

1. 后端算子实现

  ${TPUC_ROOT}/customlayer/include/nodechip_swapchannel.h 头文件声明如下：

  .. code-block:: cpp

    #ifndef NODECHIP_ABSADD_H_
    #define NODECHIP_ABSADD_H_

    #include "tpu_kernel.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

    void nodechip_swapchannel_global(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int *shape,
        const int *order,
        data_type_t dtype);

    #ifdef __cplusplus
    }
    #endif

    #endif


  ${TPUC_ROOT}/customlayer/src/nodechip_swapchannel.c 代码如下：

  .. code-block:: c

    #include "nodechip_swapchannel.h"
    #include "common.h"

    void nodechip_swapchannel_global(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int *shape,
        const int *order,
        data_type_t dtype)
    {
        dim4 channel_shape = {.n = shape[0], .c = 1, .h = shape[2], .w = shape[3]};
        int data_size = tpu_data_type_size(dtype);
        int offset = channel_shape.w * channel_shape.h * data_size;
        for (int i = 0; i < 3; i++) {
            tpu_gdma_cpy_S2S(
                output_global_addr + i * offset,
                input_global_addr + order[i] * offset,
                &channel_shape,
                NULL,
                NULL,
                dtype);
        }
    }


2. 算子参数结构体与算子调用接口实现

  ${TPUC_ROOT}/customlayer/include/backend_custom_param.h 中 swapchannel_param_t 定义如下：

  .. code-block:: c

    typedef struct swapchannel_param {
      int order[3];
    } swapchannel_param_t;


  ${TPUC_ROOT}/customlayer/include/api_swapchannel.h 头文件声明如下：

  .. code-block:: cpp

    #pragma once
    #include "api_common.h"
    #include "backend_custom_param.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

    void api_swapchannel_global(
        global_tensor_spec_t *input,
        global_tensor_spec_t *output,
        custom_param_t *param);

    #ifdef __cplusplus
    }
    #endif


  ${TPUC_ROOT}/customlayer/src/api_swapchannel.c 代码如下：

  .. code-block:: c

    #include "tpu_utils.h"
    #include "api_swapchannel.h"
    #include "nodechip_swapchannel.h"

    // parse param function
    swapchannel_param_t parsParam(custom_param_t* param) {
        swapchannel_param_t sc_param = {0};
        for (int i = 0; i < 3; i++) {
            sc_param.order[i] = param[0].int_arr_t[i];
        }
        return sc_param;
    }

    // global api function
    void api_swapchannel_global(
        global_tensor_spec_t *input,
        global_tensor_spec_t *output,
        custom_param_t *param)
    {
        swapchannel_param_t sc_param = parsParam(param);

        nodechip_swapchannel_global(
            input->addr,
            output->addr,
            input->shape,
            sc_param.order,
            tpu_type_convert(input->dtype));
    }


3. 后端调用接口

  ${TPUC_ROOT}/customlayer/src/backend_custom_api.cpp 代码如下：

  .. code-block:: cpp

    #include "backend_helper.h"
    #include "common_def.h"
    #include "api_common.h"

    // 1. include head file of api function
    #include "api_swapchannel.h"

    // 2. global backend api functions
    IMPL_CUSTOM_API_GLB(swapchannel, swapchannel_param_t)

  完成后端调用接口后运行 $TPUC_ROOT/customlayer/build.sh 编译并安装自定义算子动态库。

4. TpuLang接口调用

  调用TpuLang接口构建自定义算子模型的python代码如下：

   .. code-block:: python

      import numpy as np
      import transform.TpuLang as tpul

      # 1. initialize tpulang
      tpul.init("BM1684X", True)

      # 2. prepare the input
      dtype = "float32"
      input_shape = [1, 3, 14, 14]
      x_data = np.random.random(input_shape).astype(np.float32)
      x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)

      # 3. build model
      def shape_func(tensors_in):
          # the shape inference function
          return [tensors_in[0].shape]

      out_names = ["out"]
      params = {"order": [2, 1, 0]}

      outs = tpul.custom(
              tensors_in=[x],
              shape_func=shape_func,
              # op_name should be consistent with the backend
              op_name="swapchannel",
              params=params,
              out_dtypes=[dtype],
              out_names=out_names)

      # 4. compile to Top mlir file, the input will be saved in {top_mlir}_in_f32.npz
      top_mlir = "tpulang_test_net"
      tpul.compile(top_mlir, [x], outs, False, 2, has_custom=True)

  通过以上代码可获得 tpulang_test_net.mlir 的Top层mlir文件，后续的模型部署过程请参考用户接口章节。

Caffe示例
~~~~~~~~~~

本小节提供了Caffe中 absadd 和 ceiladd 自定义算子的应用示例。

1. 后端算子与接口实现

  absadd 与 ceiladd 的实现部分和 swapchannel 算子相似，可在 $TPUC_ROOT/customlayer/include 和  $TPUC_ROOT/customlayer/src 目录下找到相应代码。

2. 定义 Caffe 自定义算子

  absadd 和 ceiladd 在 $TPUC_ROOT/customlayer/python/my_layer.py 中的定义如下：

  .. code-block:: python

    import caffe
    import numpy as np

    # Define the custom layer
    class AbsAdd(caffe.Layer):

        def setup(self, bottom, top):
            params = eval(self.param_str)
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
        module: "my_layer"
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
        module: "my_layer"
        layer: "CeilAdd"
        param_str: "{ 'b_val': 1.5}"
      }
    }


3. 实现算子前端转换函数

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

        # The output shape compute based on reshape function in my_layer
        out_shape = self.getShape(layer.top[0])
        outs = top.CustomOp([self.mlir.get_tensor_type(out_shape)], [in_op],
                            **attrs,
                            ip=self.mlir.insert_point).output
        # add the op result to self.operands
        self.addOperand(layer.top[0], outs[0])

4. Caffe前端转换

  通过调用 my_converter.py 接口完成对 $TPUC_ROOT/customlayer/test 目录下的 my_model.prototxt, my_model.caffemodel Caffe模型进行转换 （该Caffe模型中包含了absadd与ceiladd算子），命令如下：

  .. code-block:: shell

    my_converter.py \
    --model_name caffe_test_net \
    --model_def $TPUC_ROOT/customlayer/test/my_model.prototxt \
    --model_data $TPUC_ROOT/customlayer/test/my_model.caffemodel \
    --input_shapes [[1,3,14,14],[1,3,24,26]] \
    --mlir caffe_test_net.mlir

  通过以上步骤可获得caffe_test_net.mlir的Top层mlir文件，后续的模型部署过程请参考用户接口章节。


