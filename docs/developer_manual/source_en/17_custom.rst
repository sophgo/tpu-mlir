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

  b. The operator can optionally implement the local layer. The input and output data of the local layer are stored in local memory. It can be combined with other layers for LayerGroup optimization, avoiding the need to transfer data to and from global memory during the calculation of this layer. The advantage is that it saves GDMA transfers and achieves higher computational efficiency. However, it is more complex to implement. The local memory needs to be allocated in advance during model deployment, limiting its usage and making it impractical for certain operators.

  c. The operator also need to implement some additional functions for correctness verification, shape inference and more complex local layer during the compilation phase.

The frontend can build models containing custom operators using tpulang or Caffe, and finally deploy the models through the model conversion interface of TPU-MLIR. This chapter primarily introduces the process of using custom operators in the TPU-MLIR release package.


Custom Operator Addition Process
--------------------------------

Notice: in the following context, {op_name} represent the name of operator, whose length is limited to 20. {processor_arch} represents architecture of processor, whose optional values are `BM1684X` or `BM1688`.

Add TpuLang Custom Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load TPU-MLIR

.. include:: ./env_comm.rst

2. Define the structure of parameter and parse function

  a. Define the structure of the operator parameters in `$TPUC_ROOT/customlayer/include/backend_custom_param.h`. This structure will be used in functions in subsequent steps. An example of the structure is as follows:

  .. code-block:: c

    typedef struct {op_name}_param {
      ...
    } {op_name}_param_t;

  b. One should implement corresponding functions to parse the parameters passed from the frontend of toolchain based on the parameters required by the operator. Parameters are passed through a pointer to a `custom_param_t` array. Starting from the second element of the array, a `custom_param_t` structure contains information about a parameter, and the parameter value is stored in the corresponding member variables in `custom_param_t` (which includes integer, floating-point number, integer array, and floating-point array variables). The order of the parameters is the same as the order in which the user provides them when calling the TpuLang interface. The definition of the `custom_param_t` is as follows:

  .. code-block:: c

    typedef union {
      int int_t;
      float float_t;
      // max size of int and float array is set as 16
      int int_arr_t[16];
      float float_arr_t[16];
    } custom_param_t;

  An example of a parse function is as follows:

  .. code-block:: c

    static {op_name}_param_t {op_name}_parse_param(const void* param) {
      ...
    }

3. Plugins for compiler

  In some cases, small modifications are needed for controling the behaviour of compiler for different type of custom operator with different parameters. Recently some Plugins are provided to realize the aims (please define them in file in directory ./plugin):

  a. [Required] Inference function. This plugin is used for comparation of output data between TOP and TPU dialect. The form of plugin function is as follows:

  .. code-block:: c

    void inference_absadd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
      const int* input_dims, const float** inputs, float** outputs);

  where input_shapes and input_dims is array of input shapes and dims respectively. inputs and output is pointer of data of inputs and outputs data outputs.

  b. [Optional] Shape infer function. This plugin is used for shape inference in TOP dialect. If not implmemnt, the default is that there is one input and one output while output shape is equal to input shape. The form of plugin function is as follows:

  .. code-block:: c

    void shape_inference_absadd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
      const int* input_dims, int (*output_shapes)[MAX_SHAPE_DIMS], int* output_dims);

  c. [Optional] Force dynamic run. The shape of some operators changes dynamically (e.g., NonZero) and needs to be forced to run dynamically even under static compilation. The form of plugin function is as follows:

  .. code-block:: c

    bool force_dynamic_run_{op_name}(void* param, int param_size);

  Not provided this plugin, that custom operator if static by default.

  d. [Optional] Try to group with other operators. The form of plugin function is as follows:

  .. code-block:: c

    bool local_gen_support_{op_name}(void* param, int param_size);

  Not provided this plugin, the default is that custom operator corresponding to {op_name} is not supported to group with other operators. Otherwise, one should implement the backend apis like `api_xxx_local` and `api_xxx_local_bfsz`(optional) in cases when this plugin function returns true.

  e. [Optional] Try to split `axis` when group with other operators. The form of plugin function is as follows:

  .. code-block:: c

    bool allow_data_split_{op_name}(void* param, int param_size, int axis, group_type_t group_type);

  Not provided this plugin, the default is that when fusing with other operators, custom operator corresponding to {op_name} is supported to try to split `axis`.

  f. [Optional] Backward slice derivation. Also applies to local layers (see the LayerGroup chapter for details). The form of plugin function is as follows:

  .. code-block:: c

    bool backwardh_{op_name}(void* param, int param_size, int* in_idx, int* in_slice, int out_idx, int out_slice);

    bool backwardw_{op_name}(void* param, int param_size, int* in_idx, int* in_slice, int out_idx, int out_slice);

  where respectively `in_idx` and `in_slice` are pointers to index and size of slice of input tensor while `out_idx` and `out_slice` are index and size of slice of output tensor. Not provided this plugin, the default is that when fusing with other operators, `out_idx` is equal to the value that `in_idx` is pointing to and `out_slice` is equal to the value that `in_slice` is pointing to.


4. Develop backend operators

  We can develop backend operators based on TPU-Kernel(4.1) or based on ppl(4.2)

4.1 Based on TPU-Kernel

  Assuming the current path is $TPUC_ROOT/customlayer

  a. Declare the custom operator functions for the global layer and local layer in ./include/tpu_impl_custom_ops.h

  .. code-block:: c

    void tpu_impl_{op_name}_global // Required

    void tpu_impl_{op_name}_local  // Optional

  b. Add the tpu_impl_{op_name}.c file in the ./src directory and invoke the TPU-Kernel interfaces to implement the corresponding functions.

  c. Add the interface_{op_name}.c file in the ./src directory and implement the backend api.

  .. code-block:: c

    void api_{op_name}_global // Required. Calling void tpu_impl_{op_name}_global

    void api_{op_name}_local  // Optional. Calling void tpu_impl_{op_name}_local

4.1 Based on ppl

  Assuming the current path is $TPUC_ROOT/customlayer

  a. Add the {op_name}.pl file in the ./PplBackend/src directory, where .pl is an implementation of the kernerl function using ppl syntax.

  b. Add the {op_name}_tile.cpp file in the ./PplBackend/src directory and implement the tiling func and specifies the kernel implementation corresponding to the dtype.

  .. code-block:: c

    // The kernelFunc definition is the same as the function name {op_name}.pl
    using KernelFunc = int (*)(global_addr_t, global_addr_t,
                              float, int, int, int, int, int, bool);

    int {op_name}_tiling/{op_name}(...) { // Required.
      KernelFunc func;
      if (dtype == SG_DTYPE_FP32) {
        func = {op_name}_f32;
      } else if (dtype == SG_DTYPE_FP16) {
        func = {op_name}_f16;
      } else if (dtype == SG_DTYPE_BFP16) {
        func = {op_name}_bf16;
      ....
      } else {
        assert(0 && "unsupported dtype");
      }
      // Optional. Tiling func
    ...
    }

  c. Add the {op_name}_api.c file in the ./PplBackend/src directory and implement the backend api.

  .. code-block:: c

    extern int {op_name}_tiling/{op_name} (...);            // Required.

    void api_addconst_global/local(..., onst void *param) { // Required.
      PARSE_PARAM({op_name}, {op_name}_param, param);
      {op_name}_tiling/{op_name}(...);
    }

5. Define the operator's parameter structure and write the operator's general interface

  Add the interface_{op_name}.c file in the ./src directory and implement the corresponding interfaces:

    int64_t api_{op_name}_global_bfsz (Optional. Calculate global buffer size)

    int api_{op_name}_local_bfsz (Optional. Calculate local buffer size)

    void type_infer_{op_name} (Optional. For dynamic run. Infer shape and dtype of outputs from those of inputs)

    void slice_infer_{op_name} (Optional. For dynamic run. Infer the output slice from the input slice, if not implemented, there is only one input and one output by default, and the output slice is the same as the input slice)

6. Register the operator

  In file register_ops.cmake, add op name for registering your operator:

  .. code-block:: shell

    register_custom_op({op_name})     // 4.1 Based on TPU-Kernel

    // OR

    register_custom_ppl_op({op_name}) // 4.2 Based on ppl

  Once local layer could be implemented, register it:

  .. code-block:: shell

    register_custom_local_op({op_name})     // 4.1 Based on TPU-Kernel

    // OR

    register_custom_ppl_local_op({op_name}) // 4.2 Based on ppl

  Once global layer needs buffer, register it:

  .. code-block:: shell

    register_custom_global_bfsz({op_name})

  Once local layer needs buffer, register it:

  .. code-block:: shell

    register_custom_local_bfsz({op_name})


7. Compile and install the dynamic library

  Firstly, initialize your environment by running the shell command:

  .. code-block:: shell

    source $TPUC_ROOT/customlayer/envsetup.sh

  Then compile the plugin for custom operators:

  .. code-block:: shell

    rebuild_custom_plugin

  and compile the backend apis (target: `libbackend_custom.so`):

  .. code-block:: shell

    rebuild_custom_backend

  After that, compile the corresponding firmware according to the actual usage scenario (For dynamic run):

  a. CMODEL mode (target: `libfirmware_custom_xxx.so`):

  .. code-block:: shell

    rebuild_custom_firmware_cmodel {processor_arch}

  b. SoC mode (target: `libxxx_kernel_module_custom_soc.so`):

  .. code-block:: shell

    rebuild_custom_firmware_soc {processor_arch}

  c. PCIe mode (target: `libxxx_kernel_module_custom_pcie.so`, note that BM1688 does not support PCIe mode):

  .. code-block:: shell

    rebuild_custom_firmware_pcie {processor_arch}

  At this point we have completed the work on the backend part of the custom operator.

8. Invoke TpuLang to build the model

  Refer to the TPULang Interface section for instructions on how to use TpuLang.

  TpuLang provides the `TpuLang.custom` interface to build custom operators in the frontend of toolchain (please ensure that the `op_name` matches the name of the backend operator): Note that, `params` should be dictionary in python, whose key should be a string representing the name of parameter and value should be a integer or floating-point number, or a list of integer or floating-point number (the length of list should be no greater than 16). When building the neural network, the number and order of keys should keep the same for the same custom operator and for the same key, if its value is a list, the length should keep the same.

  .. code-block:: python

    def custom(tensors_in: List[TpuLang.Tensor],
                   op_name: str,
                   out_dtypes: List[str],
                   out_names: List[str] = None,
                   params: dict = None)
                   -> List[TpuLang.Tensor]
    '''
        The custom op
        Arguments:
            tensors_in: list of input tensors (including weight tensors).
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

    import transform.TpuLang as tpul

    class xxx:
      @staticmethod
      def native(...):
          ...
          return ...
      @staticmethod
      def tpulang(inputs, ...):
          params = dict(...)
          outputs = tpul.custom(
              tensors_in=inputs,
              op_name={op_name},
              params=params,
              out_dtypes=...)
          return outputs

  where `native` function is used to calculate the reference output data of custom layer. `tpulang` function constructs the custom layer using `TpuLang.custom` function.

  b. Unit test

    After defining the custom operator, one should test whether this inferface is reliable. In the directory `$TPUC_ROOT/customlayer/test_if/unittest`, create a python file named "test_{op_name}.py". In this file, create a class, which is derived from class `TestTPULangCustom` and create test functions.

    The shell command below would tries to automatically perform the unit tests:

  .. code-block:: shell

    run_custom_unittest {processor_arch}

9. On-Processor test
  When at least a dynamic subnet exists in the network, the firmware containing in bmodel might be not useful since shell command `bmrt_test` does not work. In this case, one might need the following shell command to replace the old firmware with new one:

  .. code-block:: shell

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_soc.so # SoC mode

    tpu_model --kernel_update xxx.bmodel libxxx_kernel_module_custom_pcie.so #PCIe mode

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

The next steps are the same as compile and install steps in "Add TpuLang Custom Operator" section.

Custom Operator Example
-----------------------

This section assumes that the tpu-mlir release package has been loaded.

Example of TpuLang
~~~~~~~~~~~~~~~~~~~

This subsection provides a sample of swapchanel operator implementation and application through TpuLang interface.

1. Parameter Parser

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

2. Plugin Functions

  In source file ${TPUC_ROOT}/customlayer/plugin/plugin_swapchannel.c:

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

3. Backend Operator Implementation

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

4. Backend Interface

  In file ${TPUC_ROOT}/customlayer/src/interface_swapchannel.c, one should define two functions: `void type_infer_swapchannel` and `void api_swapchannel_global`:

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


5. Register the Custom Operator

  Add the code in ${TPUC_ROOT}/customlayer/register_ops.cmake:

  .. code-block:: c

    register_custom_op(swapchannel)

  After completion, you can refer to the section "Add TpuLang Custom Operator" to compile and install dynamic libraries.

6. TpuLang Interface Invocation

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

Custom AP(Application Processor) Operator Adding Process
-------------------------------------

TpuLang Custom AP Operator Adding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load TPU-MLIR

   The process is consistent with when loading tpu-mlir for TPU custom operators.

2. Write Processor Operator Implementation

  Assuming you are currently in the $TPUC_ROOT/customlayer path, declare a custom derived class
  layer that inherits from the ap_layer class in the header file
  ./include/custom_ap/ap_impl_{op_name}.h (where "forward()" declares the specific
  implementation method, "shape_infer()" declares the method for inferring tensor shape
  changes before and after, "dtype_infer()" declares the method for inferring data type
  changes before and after, "get_param()" declares the parameter parsing method). Also,
  add ap_impl_{op_name}.cpp in the ./ap_src directory, where you implement the corresponding
  functions, define new member variables, and override the member functions.

3. Register Custom Operator

  a. Add the operator's name in ap_impl_{op_name}.cpp to register the custom operator:

  .. code-block:: c++

    REGISTER_APLAYER_CLASS(AP_CUSTOM, {op_name});

  b. And define the member AP_CUSTOM_{OP_NAME} in the enumeration type `AP_CUSTOM_LAYER_TYPE_T`
    in ./customlayer/include/customap_common.h, where OP_NAME is uppercase.

  .. code-block:: c++

    typedef enum {
      AP_CUSTOM                                 = 10001,
      AP_CUSTOM_TOPK                            = 10002,
      AP_CUSTOM_XXXX                            = 10003,
      AP_CUSTOM_LAYER_NUM                       ,
      AP_CUSTOM_LAYER_UNKNOW = AP_CUSTOM_LAYER_NUM,
    } AP_CUSTOM_LAYER_TYPE_T;

  c. Define the instantiation method in customlayer/ap_src/ap_layer.cpp

  .. code-block:: c++

    bmap::ap_layer* create{OP_NAME}Layer() {
      return new bmap::ap_{op_name}layer();
    }

    void registerFactoryFunctions() {
      getFactoryMap()[std::string("{OP_NAME}")] = createTopkLayer;
      // Register other class creators
      // ...
    }

4. Compiler Patch

  Sometimes, it is necessary to modify the compiler to control the compilation behavior of
  different custom operators under different parameters, and this requires adding some patches.
  The following patch functions are currently supported (defined in the ./plugin folder):

  a. [Required] You need to implement the operator parameter parsing function yourself,
    which is used to obtain the key parameters required by the operator, and override the
    get_param() method of the custom layer:

    .. code-block:: c++

      int ap_mylayer::get_param(void *param, int param_size);


  b. [Required] Inference function, i.e., the C++ implementation of the operator. Override
    the custom layer's forward() method:

    .. code-block:: c++

      int ap_mylayer::forward(void *raw_param, int param_size);

  c. [Optional] Shape inference function. This patch function is used for compiler shape
    inference. If not implemented, by default, there is only one input and one output, and
    the output shape is the same as the input shape. The patch function is as follows:

    .. code-block:: c++

      int ap_mylayer::shepe_infer(void *param, int param_size,
                                    const vector<vector<int>> &input_shapes,
                                    vector<vector<int>> &output_shapes);

  Where input_shapes/output_shapes are arrays of input/output tensor shapes, and
  input_dims/output_dims are arrays of input/output tensor dimensions.

5. Compile and Install the Dynamic Library

  First, initialize the environment:

  .. code-block:: shell

    source $TPUC_ROOT/customlayer/envsetup.sh

  Then, complete the compilation of the patch (to obtain `libplugin_custom.so`):

  .. code-block:: shell

    rebuild_custom_plugin

  Compile the custom operator library file according to the processor architecture
  (to obtain `libcustomapop.so`). It is important to note that the environment for
  compiling the custom processor operator must be compatible with the glibc version in the bmodel
  runtime environment. The commands are as follows:

  a. x86 architecture

  .. code-block:: shell

    rebuild_custom_apop_x86

  b. ARM architecture


  .. code-block:: shell

    rebuild_custom_apop_aarch64

  With this, we have completed the backend part of the custom processor operator.

6. Build Custom AP Operators with TpuLang

  For how to use TpuLang, please refer to the TpuLang interface section.

  TpuLang provides the `TpuLang.custom` interface which can also be used for custom processor operators.
  The method of use is basically the same as that for custom TPU operators. The difference is that
  when defining the "TpuLang.custom" object, the "op_name" parameter must start with the "ap."
  prefix to distinguish it, for example, "ap.topk":

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
              op_name="ap.topk",
              params=params,
              out_dtypes=...)
          return outputs

7. On-Processor Testing

  When the network contains custom processor operators, the bmodel needs to include operator information.
  Use the command to write libcustomapop.so into the bmodel file, which is used for all host
  processor architectures:

  .. code-block:: shell

    tpu_model --custom_ap_update xxx.bmodel libcustomapop.so

  Note: It is especially important that the environment for compiling the custom processor operator
  is compatible with the glibc version in the bmodel runtime environment.

Custom AP(Application Processor) Operator Example
----------------------------

This section assumes that the tpu-mlir release package has been loaded.

TpuLang Example
~~~~~~~~~~~~~~~~

This subsection provides an example of a swapchannel operator implementation and its application
through the TpuLang interface.

1. Custom Operator Derived Class

  Here, the field "order" corresponds to the "order" attribute on the frontend.

  Define member variables in the custom class in {TPUC_ROOT}/customlayer/ap_src/ap_impl_{op_name}.cpp:

  .. code-block:: c++

    private:
      int axis_;
      int K_;


  Override the `get_param()` interface in the custom class in
  {TPUC_ROOT}/customlayer/ap_src/ap_impl_{op_name}.cpp. It is worth noting that what is
  passed from the compiler to the backend is an array A of custom_param_t, the first
  element of which is reserved, and from the second element onwards, each element
  corresponds to an attribute on the frontend:

  .. code-block:: c++

    int ap_topklayer::get_param(void *param, int param_size) {
      axis_ = ((custom_param_t *)param)[1].int_t;
      K_ = ((custom_param_t *)param)[2].int_t;
      return 0;
    }

  Override the `shape_infer()` interface in the custom class in
  {TPUC_ROOT}/customlayer/ap_src/ap_impl_{op_name}.cpp:

  .. code-block:: c++

    int ap_topklayer::shepe_infer(const vector<vector<int> > &input_shapes,
                                      vector<vector<int> > &output_shapes) {
      get_param(param, param_size);
      for (const auto& array : input_shapes) {
        output_shapes.emplace_back(array);
      }
      output_shapes[0][axis_] = std::min(K_, input_shapes[0][axis_]);
      return 0;
    }

2. Processor Operator Implementation

  Override the `forward()` interface in the custom class in
  {TPUC_ROOT}/customlayer/ap_src/ap_impl_{op_name}.cpp:

  .. code-block:: c++

    int ap_topklayer::forward(void *raw_param, int param_size) {
      // implementation code right here
      return 0;
    }

3. Processor Operator Registration

  a. Add the operator's name in ap_impl_{op_name}.cpp to register the custom operator:

  .. code-block:: c++

    REGISTER_APLAYER_CLASS(AP_CUSTOM_TOPK, ap_topk);

  b. And define the member AP_CUSTOM_TOPK in the enumeration type `AP_CUSTOM_LAYER_TYPE_T`
    in ./customlayer/include/customap_common.h.

  .. code-block:: c++

    typedef enum {
      AP_CUSTOM                                 = 10001,
      AP_CUSTOM_TOPK                            = 10002,
      AP_CUSTOM_LAYER_NUM                          ,
      AP_CUSTOM_LAYER_UNKNOW = AP_CUSTOM_LAYER_NUM,
    } AP_CUSTOM_LAYER_TYPE_T;

  c. Define the instantiation method in customlayer/ap_src/ap_layer.cpp

  .. code-block:: c++

    bmap::ap_layer* createTopkLayer() {
      return new bmap::ap_topklayer();
    }

    void registerFactoryFunctions() {
      getFactoryMap()[std::string("TOPK")] = createTopkLayer;
      // Register other class creators
      // ...
    }

4. Frontend Preparation

  The process of building a custom Processor operator using the TpuLang interface is basically
  the same as for a TPU custom operator. The difference is that when defining the
  "TpuLang.custom" object, the "op_name" parameter must start with the "ap." prefix to
  distinguish it, for example, "ap.topk"
