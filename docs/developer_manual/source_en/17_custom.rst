Custom Operators
=================

Overview
---------
TPU-MLIR already includes a rich library of operators that can fulfill the needs of most neural network models. However, in certain scenarios, there may be a requirement for users to define their own custom operators to perform computations on tensors. This need arises when:

1. TPU-MLIR does not support a specific operator, and it cannot be achieved by combining existing operators.
2. The operator is private.
3. Combining multiple operator APIs does not yield optimal computational performance, and custom operations at the TPU-Kernel level can improve execution efficiency.

The functionality of custom operators allows users to freely use the interfaces in TPU-Kernel to compute tensors on the TPU, and encapsulate this computation process as backend operators (refer to the TPU-KERNEL Technical Reference Manual for backend operator development). The backend operator calculation involves operations related to the global layer and local layer:

  a. The operator must implement the global layer. The input and output data of the global layer are stored in DDR. The data needs to be transferred from global memory to local memory for execution and then transferred back to global memory. The advantage is that local memory can be used flexibly, but it has the disadvantage of generating a considerable number of GDMA transfers, resulting in lower TPU utilization.


  b. The operator can optionally implement the local layer. The input and output data of the local layer are stored in local memory. It can be combined with other layers for layer group optimization, avoiding the need to transfer data to and from global memory during the calculation of this layer. The advantage is that it saves GDMA transfers and achieves higher computational efficiency. However, it is more complex to implement. The local memory needs to be allocated in advance during model deployment, limiting its usage and making it impractical for certain operators.

The frontend can build models containing custom operators using tpulang or Caffe, and finally deploy the models through the model conversion interface of TPU-MLIR. This chapter primarily introduces the process of using custom operators in the TPU-MLIR release package.


Custom Operator Addition Process
--------------------------------

Add TpuLang Custom Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load TPU-MLIR

.. include:: ./../../quick_start/source_en/env_var.rst

2. Develop backend operators based on TPU-Kernel

  Assuming the current path is $TPUC_ROOT/customlayer, add the nodechip_{op_name}.h header file in the ./include directory to declare the custom operator functions for the global layer and local layer (void nodechip_{op_name}_global and void nodechip_{op_name}_local, respectively). Then, add the nodechip_{op_name}.c file in the ./src directory and invoke the TPU-Kernel interfaces to implement the corresponding functions.

3. Define the operator's parameter structure and write the operator's interface

  a. Add the corresponding structure {op_name}_param_t in the ./include/backend_custom_param.h header file to receive parameters from the frontend of toolchain, based on the parameters required by the operator.

  b. Add the api_{op_name}.h header file in the ./include directory to declare the interfaces for the custom operator functions (void api_{op_name}_global and void api_{op_name}_local). Then, add the api_{op_name}.c file in the ./src directory and implement the corresponding interfaces.

  c. Additionally, users need to implement corresponding functions to parse the parameters passed from the frontend of toolchain based on the parameters required by the operator. Parameters are passed through a pointer to a custom_param_t array, where each custom_param_t structure contains information about a parameter, and the parameter value is stored in the corresponding member variables in custom_param_t (which includes integer, floating-point number, integer array, and floating-point array variables). The order of the parameters is the same as the order in which the user provides them when calling the TpuLang interface. The definition of the custom_param_t is as follows:

  .. code-block:: c

    typedef struct {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;


4. Define the backend interface

  In ./src/backend_custom_api.cpp, build the backend interface using macro definitions. This interface will be called during Codegen in the frontend of toolchain. The format is as follows:

  .. code-block:: C

    IMPL_CUSTOM_API_GLB({op_name}, {op_name}_param_t)

    IMPL_CUSTOM_API_LOC({op_name}, {op_name}_param_t)


5. Compile and install the dynamic library

  By running the build.sh script in $TPUC_ROOT/customlayer, the compilation of the custom operator will be completed. It will generate the backend_custom_api.so dynamic library and install it in $TPUC_ROOT/lib.

6. Invoke TpuLang to build the model

  Refer to the TPULang Interface section for instructions on how to use TpuLang.

  TpuLang provides the TpuLang.custom interface to build custom operators in the frontend of toolchain (ensure that the op_name part matches the name of the backend operator):

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

Add Caffe Custom Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steps 1-5 are the same as in Add TpuLang Custom Operator section, and will not be repeated here.

6. Defining custom operators in Caffe

  To define custom operators in Caffe, you need to define a class in the file $TPUC_ROOT/customlayer/python/my_layer.py that inherits from caffe.Layer and override the setup, reshape, forward, and backward functions as needed.

7. Implementing the frontend conversion function

  The type of custom operators implemented in python is "Python", so you need to implement a corresponding conversion function of MyCaffeConverter class defined in the file $TPUC_ROOT/customlayer/python/my_converter.py, based on the definition in step 6.

  After the definition, you can call my_converter.py interface for Top MLIR conversion:

  .. code-block:: shell

    my_converter.py \
    --model_name # the model name \
    --model_def # .prototxt file \
    --model_data # .caffemodel file \
    --input_shapes # list of input shapes (e.g., [[1,2,3],[3,4,5]]) \
    --mlir # output mlir file


Custom Operator Example
-----------------------

This section assumes that the tpu-mlir release package has been loaded.

Example of TpuLang
~~~~~~~~~~~~~~~~~~~

This subsection provides a sample of swapchanel operator implementation and application through TpuLang interface.

1. Backend Operator Implementation

  The following is the declaration in the header file

  ${TPUC_ROOT}/customlayer/include/nodechip_swapchannel.h:

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


  The code of ${TPUC_ROOT}/customlayer/src/nodechip_swapchannel.c:

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


2. Operator Parameter Structure and Implementation of the Operator Interface

  The definition of swapchannel_param_t in

  ${TPUC_ROOT}/customlayer/include/backend_custom_param.h is as follows:

  .. code-block:: c

    typedef struct swapchannel_param {
      int order[3];
    } swapchannel_param_t;


  The following is the declaration in the header file

  ${TPUC_ROOT}/customlayer/include/api_swapchannel.h:

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


  The code of ${TPUC_ROOT}/customlayer/src/api_swapchannel.c:

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


3. Backend Interface

  The code of ${TPUC_ROOT}/customlayer/src/backend_custom_api.cpp:

  .. code-block:: cpp

    #include "backend_helper.h"
    #include "common_def.h"
    #include "api_common.h"

    // 1. include head file of api function
    #include "api_swapchannel.h"

    // 2. global backend api functions
    IMPL_CUSTOM_API_GLB(swapchannel, swapchannel_param_t)

  After completing the implementation of the backend interface, you can run $TPUC_ROOT/customlayer/build.sh to compile and install the custom operator dynamic library.

4. TpuLang Interface Invocation

  Here is an example of Python code that utilizes the TpuLang interface to build a custom operator model:

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

  By running the above code, you can obtain the Top MLIR file tpulang_test_net.mlir. For the subsequent model deployment process, please refer to the User Interface chapter.

Example of Caffe
~~~~~~~~~~~~~~~~~

This subsection provides application examples of custom operators absadd and ceiladd in Caffe.

1. Backend operator and interface implementation

  The implementation of absadd and ceiladd is similar to the swapchannel operator and can be found in  $TPUC_ROOT/customlayer/include and $TPUC_ROOT/customlayer/src.

2. Defining Caffe custom operators

  The definition of absadd and ceiladd in Caffe can be found in $TPUC_ROOT/customlayer/python/my_layer.py as follows:

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

  The expression of corresponding operators in Caffe prototxt is as follows:

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

3. Implement operator front-end conversion functions

  Define a convert_python_op function of the MyCaffeConverter class in $TPUC_ROOT/customlayer/python/my_converter.py, the code is as follows:

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

4. Caffe front-end conversion

  Complete the conversion of Caffe model in the $TPUC_ROOT/customlayer/test directory (i.e., my_model.prototxt and my_model.caffemodel, which contain absadd and ceiladd operators) by calling the my_converter.py interface, the command is as follows:

  .. code-block:: shell

    my_converter.py \
    --model_name caffe_test_net \
    --model_def $TPUC_ROOT/customlayer/test/my_model.prototxt \
    --model_data $TPUC_ROOT/customlayer/test/my_model.caffemodel \
    --input_shapes [[1,3,14,14],[1,3,24,26]] \
    --mlir caffe_test_net.mlir

  So far, the Top MLIR file caffe_test_net.mlir has been obtained. For the subsequent model deployment process, please refer to the user interface chapter.
