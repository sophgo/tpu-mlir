Custom Operators
=================

Overview
---------
TPU-MLIR already includes a rich library of operators that can fulfill the needs of most neural network models. However, in certain scenarios, there may be a requirement for users to define their own custom operators to perform computations on tensors. This need arises when:

1. TPU-MLIR does not support a specific operator, and it cannot be achieved by combining existing operators.
2. The operator is private.
3. Combining multiple operator APIs does not yield optimal computational performance, and custom operations at the TPU-Kernel level can improve execution efficiency.

The functionality of custom operators allows users to freely use the interfaces in TPU-Kernel to compute tensors on the TPU, and encapsulate this computation process as backend operators (refer to the TPU-KERNEL Technical Reference Manual for backend operator development). The backend operator calculation involves operations related to the global layer and local layer:

  a. The operator must implement the global layer. The input and output data of
     the global layer are stored in DDR. The data needs to be transferred from
     global memory to local memory for execution and then transferred back to
     global memory. The advantage is that local memory can be used flexibly, but
     it has the disadvantage of generating a considerable number of GDMA
     transfers, resulting in lower the Tensor Competing Processor utilization.

  b. The operator can optionally implement the local layer. The input and output data of the local layer are stored in local memory. It can be combined with other layers for layer group optimization, avoiding the need to transfer data to and from global memory during the calculation of this layer. The advantage is that it saves GDMA transfers and achieves higher computational efficiency. However, it is more complex to implement. The local memory needs to be allocated in advance during model deployment, limiting its usage and making it impractical for certain operators.

  c. The operator should implement the function of shape inference, which can infer the data type and shape of outputs from those of inputs.

The frontend can build models containing custom operators using tpulang or Caffe, and finally deploy the models through the model conversion interface of TPU-MLIR. This chapter primarily introduces the process of using custom operators in the TPU-MLIR release package.


Custom Operator Addition Process
--------------------------------

Notice: in the following context, {op_name} represent the name of operator, whose length is limited to 20. {processor_arch} represents architecture of processor, whose optional values are `bm1684x` or `bm1688`.

Add TpuLang Custom Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load TPU-MLIR

.. include:: ./../../quick_start/source_en/env_var.rst


2. Develop backend operators based on TPU-Kernel

  Assuming the current path is $TPUC_ROOT/customlayer, add the ./include/tpu_impl_custom_ops.h header file in the ./include directory to declare the custom operator functions for the global layer and local layer (void tpu_impl_{op_name}_global and void tpu_impl_{op_name}_local, respectively). Then, add the tpu_impl_{op_name}.c file in the ./src directory and invoke the TPU-Kernel interfaces to implement the corresponding functions.

3. Define the operator's parameter structure and write the operator's interface

  a. Add the interface_{op_name}.c file in the ./src directory and implement the corresponding interfaces:

    void api_{op_name}_global, void api_{op_name}_local (calling void tpu_impl_{op_name}_global and void tpu_impl_{op_name}_local respectively)

    void shape_infer_{op_name} (infer shape and dtype of outputs from those of inputs).

  b. Additionally, users need to implement corresponding functions to parse the parameters passed from the frontend of toolchain based on the parameters required by the operator. Parameters are passed through a pointer to a `custom_param_t` array. Starting from the second element of the array, a `custom_param_t` structure contains information about a parameter, and the parameter value is stored in the corresponding member variables in `custom_param_t` (which includes integer, floating-point number, integer array, and floating-point array variables). The order of the parameters is the same as the order in which the user provides them when calling the TpuLang interface (6th step). The definition of the `custom_param_t` is as follows:

  .. code-block:: c

    typedef union {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;


4. Register the operator

  In file register_ops.cmake, add op name for registering your operator:

  .. code-block:: shell

    register_custom_op({op_name})


5. Compile and install the dynamic library

  Firstly, initialize your environment by running the shell command:

  .. code-block:: shell

    source envsetup.sh in $TPUC_ROOT/customlayer

  Then compile the backend apis (target: `libbackend_custom.so`):

  .. code-block:: shell

    rebuild_custom_backend

  After that, compile the corresponding firmware according to the actual usage scenario:

  a. CMODEL mode (target: `libfirmware_custom_xxx.so`):

  .. code-block:: shell

    rebuild_custom_firmware_cmodel {processor_arch}

  b. SOC mode (target: `libxxx_kernel_module_custom_soc.so`):

  .. code-block:: shell

    rebuild_custom_firmware_soc {processor_arch}

  c. PCIE mode (target: `libxxx_kernel_module_custom_pcie.so`):

  .. code-block:: shell

    rebuild_custom_firmware_pcie {processor_arch}

  At this point we have completed the work on the backend part of the custom operator.

6. Invoke TpuLang to build the model

  Refer to the TPULang Interface section for instructions on how to use TpuLang.

  TpuLang provides the `TpuLang.custom` interface to build custom operators in the frontend of toolchain (please ensure that the `op_name` matches the name of the backend operator): Note that, `params` should be dictionary in python, whose key should be a string representing the name of parameter and value should be a integer or floating-point number, or a list of integer or floating-point number (the length of list should be no greater than 16). When building the neural network, the number and order of keys should keep the same for the same custom operator and for the same key, if its value is a list, the length should keep the same.

  .. code-block:: python

    TpuLang.custom(tensors_in: List[TpuLang.Tensor],
                   shape_func,
                   op_name: str,
                   out_dtypes: List[str],
                   out_names: List[str] = None,
                   params: dict = None)
                   -> List[TpuLang.Tensor]
    '''
        The custom op
        Arguments:
            tensors_in: list of input tensors (including weight tensors).
            shape_func: function for doing shape inference, taking shape of
                        tensors_in as inputs, and returning a list of shape
                        of output tensors.
            op_name: name of the custom operator.
            out_dtypes: list of data type of outputs.
            out_names: list of name of outputs.
            params: parameters of the custom op.

        Return:
            tensors_out: list of output tensors.
    '''

  a. Define the tpulang interface of custom operator

  For convenient, one could standardize custom operator in file $TPUC_ROOT/customlayer/python/my_tpulang_layer.py :

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
          outputs = TpuLang.custom(
              tensors_in=inputs,
              shape_func=shape_func,
              op_name=...,
              params=params,
              out_dtypes=...)
          return outputs

  where `native` function is used to calculate the reference output data of custom layer. `tpulang` function constructs the custom layer using `TpuLang.custom` function.

  b. Unit test

  After defining the custom operator, one should test whether this inferface is reliable. In the directory `$TPUC_ROOT/customlayer/test_if/unittest`, create a python file named "test_xxx.py". In this file, create a class, which is derived from class `TestTPULangCustom` and create a method named "test_xxx" for testing custom layer.

  The shell command below would tries to automatically perform the unit tests:

  .. code-block:: shell

    run_custom_unittest {processor_arch}


7. On-Processor test
  When at least a dynamic subnet exists in the network, the firmware containing in bmodel might be not useful since shell command `bmrt_test` does not work. In this case, one might need the following shell command to replace the old firmware with new one:

  .. code-block:: shell

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_soc.so # SOC mode

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_pcie.so #PCIE mode

Add Caffe Custom Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Defining custom operators in Caffe

  To define custom operators in Caffe, you need to define a class in the file $TPUC_ROOT/customlayer/python/my_caffe_layer.py that inherits from caffe.Layer and override the `setup`, `reshape`, `forward`, and `backward` functions as needed.

2. Implementing the frontend conversion function

  Provided that the caffe layer type of custom operators is "Python". One needs to implement a corresponding conversion function of `MyCaffeConverter` class defined in the file $TPUC_ROOT/customlayer/python/my_converter.py.

  After the definition, you can call my_converter.py interface for Top MLIR conversion:

  .. code-block:: shell

    my_converter.py \
    --model_name # the model name \
    --model_def # .prototxt file \
    --model_data # .caffemodel file \
    --input_shapes # list of input shapes (e.g., [[1,2,3],[3,4,5]]) \
    --mlir # output mlir file

The next steps are the same as steps 2-6 in "Add TpuLang Custom Operator" section, and will not be repeated here.

Custom Operator Example
-----------------------

This section assumes that the tpu-mlir release package has been loaded.

Example of TpuLang
~~~~~~~~~~~~~~~~~~~

This subsection provides a sample of swapchanel operator implementation and application through TpuLang interface.

1. Parameter Parser and Backend Interface

  The definition of `swapchannel_param_t` in

  ${TPUC_ROOT}/customlayer/include/backend_custom_param.h is as follows:

  .. code-block:: c

    typedef struct swapchannel_param {
      int order[3];
    } swapchannel_param_t;

  where the field `order` is corresponding to the frontend attribute named "order".

  Note that there is an array (here alias A) passing from compiler. Starting from the second element of the array, each element is corresponding to an frontend attribute of frontend. For convenient, file {TPUC_ROOT}/customlayer/include/api_common.h, provides a macro `PARSE_PARAM(swapchannel, sc_param, param)` to transform `param`(pointer to array A) into `sc_param`(pointer to backend custom param). One should define a parser function, whose parameter is a pointer of array B (constructing by dropping the first element from array A) in file ${TPUC_ROOT}/customlayer/include/param_parser.h and output type is `swapchannel_parse_param`.

  .. code-block:: c

    static swapchannel_param_t swapchannel_parse_param(const void* param) {
        swapchannel_param_t sc_param = {0};
        for (int i = 0; i < 3; i++) {
            sc_param.order[i] = ((custom_param_t *)param)[0].int_arr_t[i];
        }
        return sc_param;
    }

  In file ${TPUC_ROOT}/customlayer/src/interface_swapchannel.c, one should define tow functions: void shape_infer_swapchannel and void api_swapchannel_global：

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

2. Backend Operator Implementation

  The following is the declaration in the header file

  ${TPUC_ROOT}/customlayer/include/backend_swapchannel.h:

  .. code-block:: c

    void tpu_impl_swapchannel_global(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int *shape,
        const int *order,
        data_type_t dtype);


  The code of ${TPUC_ROOT}/customlayer/src/tpu_impl_swapchannel.c:

  .. code-block:: c

    #include "tpu_impl_custom_ops.h"

    void tpu_impl_swapchannel_global(
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


3. Register the Custom Operator

  Add the code in ${TPUC_ROOT}/customlayer/register_ops.cmake:

  .. code-block:: c

    register_custom_op(swapchannel)

  After completing the implementation of the backend interface, you can run the command listed in step 5 in "Add TpuLang Custom Operator".


4. TpuLang Interface Invocation

  In file ${TPUC_ROOT}/customlayer/python/my_tpulang_layer.py, by using function `TpuLang.custom`, one could construct a custom operator named swapChannel, which has one input, one output, and an attribute whose value is an integer list of length 3:

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

  In file ${TPUC_ROOT}/customlayer/test_if/unittest/test_swapchannel.py, one could create a unittest on custom operator named "swapChannel":

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


Example of Caffe
~~~~~~~~~~~~~~~~~

This subsection provides application examples of custom operators absadd and ceiladd in Caffe.

1. Defining Caffe custom operators

  The definition of absadd and ceiladd in Caffe can be found in $TPUC_ROOT/customlayer/python/my_caffe_layer.py as follows:

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

2. Implement operator front-end conversion functions

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

        # The output shape compute based on reshape function in my_caffe_layer
        out_shape = self.getShape(layer.top[0])
        outs = top.CustomOp([self.mlir.get_tensor_type(out_shape)], [in_op],
                            attrs,
                            ip=self.mlir.insert_point).output
        # add the op result to self.operands
        self.addOperand(layer.top[0], outs[0])

3. Caffe front-end conversion

  Complete the conversion of Caffe model in the $TPUC_ROOT/customlayer/test directory (i.e., my_model.prototxt and my_model.caffemodel, which contain absadd and ceiladd operators) by calling the my_converter.py interface, the command is as follows:

  .. code-block:: shell

    my_converter.py \
    --model_name caffe_test_net \
    --model_def $TPUC_ROOT/customlayer/test/my_model.prototxt \
    --model_data $TPUC_ROOT/customlayer/test/my_model.caffemodel \
    --input_shapes [[1,3,14,14],[1,3,24,26]] \
    --mlir caffe_test_net.mlir

  So far, the Top MLIR file caffe_test_net.mlir has been obtained. For the subsequent model deployment process, please refer to the user interface chapter.

4. Backend operator and interface implementation

  The implementation of absadd and ceiladd is similar to the swapchannel operator and can be found in  $TPUC_ROOT/customlayer/include and $TPUC_ROOT/customlayer/src.
