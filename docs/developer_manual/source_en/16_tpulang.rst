TpuLang Interface
==================

This chapter mainly introduces the process of converting models using TpuLang.

Main Work
-----------

TpuLang provides mlir external interface functions. Users can directly build their own network through Tpulang, and convert the model to the Top layer (hardware-independent layer) mlir model (the Canonicalize part is not included, so the generated file name is "\*_origin.mlir"). This process will create and add operators (Op) one by one according to the input interface functions. Finally, a mlir model file and a corresponding weight npz file will be generated.


Work Process
--------------------

1. Initialization: Set up the platform and create the graph.

2. Add OPs: cyclically add OPs of the model

    * The input parameters are converted to dict format;

    * Create output tensor;

    * Set the quantization parameters of the tensor (scale, zero_point);

    * Create op(op_type, inputs, outputs, params) and insert it into the graph.


3. Set the input and output tensor of the model. Get all model information.

4. Initialize TpuLangConverter (initMLIRImporter)

5. generate_mlir

    * Create the input op, the nodes op in the middle of the model and the return op in turn, and add them to the mlir text (if the op has weight, an additional weight op will be created)

6. Output

    * Convert the generated text to str and save it as ".mlir" file

    * Save model weights (tensors) as ".npz" files

7. End: Release the graph.


The workflow of TpuLang conversion is shown in the figure (:ref:`tpulang_convert`)。

.. _tpulang_convert:
.. figure:: ../assets/tpulang_convert.png
   :align: center

   TpuLang conversion process


Supplementary Note:
   * The op interface requires:

      - The input tensor of the op (i.e., the output tensor of the previous operator or the graph input tensor and coeff);

      - attrs extracted from the interface. Attrs will be set by MLIRImporter as attributes corresponding to the ones defined in TopOps.td;

      - If the interface includes quantization parameters (i.e., scale and zero_point), the tensor corresponding to this parameter needs to set (or check) the quantization parameters.

      - Return the output tensor(tensors) of the op.

   * After all operators are inserted into the graph and the input/output tensors of the graph are set, the conversion to mlir text will start. This part is implemented by TpuLangConverter.

   * The conversion process of TpuLang Converter is the same as onnx front-end part. Please refer to (:doc:`../05_frontend`).


Operator Conversion Example
---------------------------

This section takes the Conv operator as an example to convert a single Conv operator model to Top mlir

   .. code-block:: python

      import numpy as np

      def model_def(in_shape):
         tpul.init("BM1684X")
         in_shape = [1,3,173,141]
         k_shape =[64,1,7,7]
         x = tpul.Tensor(dtype='float32', shape=in_shape)
         weight_data = np.random.random(k_shape).astype(np.float32)
         weight = tpul.Tensor(dtype='float32', shape=k_shape, data=weight_data, is_const=True)
         bias_data = np.random.random(k_shape[0]).astype(np.float32)
         bias = tpul.Tensor(dtype='float32', shape=k_shape[0], data=bias_data, is_const=True)
         conv = tpul.conv(x, weight, bias=bias, stride=[2,2], pad=[0,0,1,1], out_dtype="float32")
         tpul.compile("model_def", inputs=[x],outputs=[conv], cmp=True)
         tpul.deinit()

   Single Conv Model


The conversion process:

1. Interface definition

   The conv interface is defined as follows:

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
            # pass


   Parameter Description

   * input: Tensor type, indicating the input Tensor with 4-dimensional NCHW format.
   * weight: Tensor type, representing the convolution kernel Tensor with 4-dimensional [oc, ic, kh, kw] format. oc indicates the number of output channels, ic indicates the number of input channels, kh is kernel_h, and kw is kernel_w.
   * bias: Tensor type, indicating the bias Tensor. There is no bias when it is None. Otherwise, the shape is required to be [1, oc, 1, 1].
   * dilation: List[int], indicating the size of holes. None means dilation equals [1,1]. Otherwise, the length is required to be 2 and the order of List is [length, width].
   * pad: List[int], indicating the padding size, if it is None, no padding is applied. Otherwise, the length is required to be 4. The order in the List is [Up, Down, Left, Right].
   * stride: List[int], indicating the step size, [1,1] when it is None. Otherwise, the length is required to be 2 and the order in the List is [length, width].
   * groups: int type, indicating the number of groups in the convolutional layer. If ic=oc=groups, the convolution is depthwise conv
   * out_dtype: string type or None, indicating the type of the output Tensor. When the input tensor type is float16/float32, None indicates that the output tensor type is consistent with the input. Otherwise,  None means int32. Value range: /int32/uint32/float32/float16.
   * out_name: string type or None, indicating the name of the output Tensor. When it is None, the name will be automatically generated.


  Define the Top.Conv operator in TopOps.td, the operator definition is as shown in the figure (:ref:`conv_top_def`)

.. _conv_top_def:
.. figure:: ../assets/convop_def.png
   :align: center
   :height: 15cm

   Conv Operator Definition


2. Build Graph

  * Initialize the model: create an empty Graph.

  * Model input: Create input tensor x given shape and data type. A tensor name can also be specified here.

  * conv interface:

      - Call the conv interface with specified input tensor and input parameters.

      - attributes, pack the input parameters into attributes defined by (:ref:`conv_top_def`)

         .. code-block:: python

            attr = {
               "kernel_shape": ArrayAttr(weight.shape[2:]),
               "strides": ArrayAttr(stride),
               "dilations": ArrayAttr(dilation),
               "pads": ArrayAttr(pad),
               "do_relu": Attr(False, "bool"),
               "group": Attr(group)
            }

      - Define output tensor

      - Insert conv op. Insert Top.ConvOp into Graph.

      - return the output tensor

  * Set the input of Graph and output tensors.

3. init_MLIRImporter:

  Get the corresponding input_shape and output_shape from shapes according to input_names and output_names. Add model_name, and generate the initial mlir text MLIRImporter.mlir_module, as shown in the figure (:ref:`origin_mlir`).

.. _origin_top_mlir:
.. figure:: ../assets/origin_mlir.png
   :align: center

   Initial Mlir Text


4. generate_mlir

   * Build input op, the generated Top.inputOp will be inserted into MLIRImporter.mlir_module.

   * Call Operation.create to create Top.ConvOp, and the parameters required by the create function are:

      - Input op: According to the interface definition, the inputs of the Conv operator include input, weight and bias. The inputOp has been created, and the op of weight and bias is created through getWeightOp().

      - output_shape: get output shape from the output tensor stored in the Operator.

      - Attributes: Get attributes from Operator, and convert attributes to Attributes that can be recognized by MLIRImporter

      After Top.ConvOp is created, it will be inserted into the mlir text

   * Get the corresponding op from operands according to output_names, create return_op and insert it into the mlir text. By this point, the generated mlir text is as shown (:ref:`tpulang_mlir_txt`).

.. _tpulang_mlir_txt:
.. figure:: ../assets/tpulang_mlir_txt.jpeg
   :align: center

   Full Mlir Text


5. Output

  Save the mlir text as Conv_origin.mlir and the weights in tensors as Conv_TOP_F32_all_weight.npz.

Tpulang API usage method
---------------------------

TpuLang is currently only applicable to the inference portion of inference frameworks. For static graph frameworks like TensorFlow,
when integrating the network with TpuLang, users need to first initialize with tpul.init('processor') (where 'processor' can be BM1684X or BM1688).
Next, prepare the tensors, use operators to build the network, and finally, call the tpul.compile interface to compile and generate bmodel.
The detailed steps for each of these processes are explained below.
You can find detailed information on various interfaces used (such as tpul.init, deinit, Tensor, and operator interfaces) in appx02 (:ref:`Appendix 02: Basic Elements of TpuLang`).

The following steps assume that the loading of the tpu-mlir release package has been completed.

Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The specific definition of the initialization function can be found in the documentation under the section titled (:ref:`Initialization Function <init>`).

   .. code-block:: python

      import transform.TpuLang as tpul
      import numpy as np

      tpul.init('BM1684X')

Prepare the tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The specific definition of the initialization function can be found in the documentation under the section titled (:ref:`tensor <tensor>`)

   .. code-block:: python

      shape = [1, 1, 28, 28]
      x_data = np.random.randn(*shape).astype(np.float32)
      x = tpul.Tensor(dtype='float32', shape=shape, data=x_data)


Build the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Continuing with the utilization of existing operators (:ref:`operator`) and the tensors prepared earlier, here is a simple model construction example:

   .. code-block:: python

      def conv_op(x,
                  kshape,
                  stride,
                  pad=None,
                  group=1,
                  dilation=[1, 1],
                  bias=False,
                  dtype="float32"):
         oc = kshape[0]
         weight_data = np.random.randn(*kshape).astype(np.float32)
         weight = tpul.Tensor(dtype=dtype, shape=kshape, data=weight_data, ttype="coeff")
         bias_data = np.random.randn(oc).astype(np.float32)
         bias = tpul.Tensor(dtype=dtype, shape=[oc], data=bias_data, ttype="coeff")
         conv = tpul.conv(x,
                     weight,
                     bias=bias,
                     stride=stride,
                     pad=pad,
                     dilation=dilation,
                     group=group)
         return conv

      def model_def(x):
         conv0 = conv_op(x, kshape=[32, 1, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
         relu1 = tpul.relu(conv0)
         maxpool2 = tpul.maxpool(relu1, kernel=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
         conv3 = conv_op(maxpool2, kshape=[64, 32, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
         relu4 =  tpul.relu(conv3)
         maxpool5 = tpul.maxpool(relu4, kernel=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
         conv6 = conv_op(maxpool5, kshape=[1024, 64, 7, 7], stride=[1,1], dtype='float32')
         relu7 =  tpul.relu(conv6)
         softmax8 = tpul.softmax(relu7, axis=1)
         return softmax8

      y = model_def(x)

compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Call the tpul.compile function (:ref:`compile`). After compilation, you will get `example_f32.bmodel`:

   .. code-block:: python

      tpul.compile("example", [x], [y], mode="f32")

deinit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The specific definition can be found in the documentation under the section titled (:ref:`Deinitialization Function <deinit>`)

   .. code-block:: python

      tpul.deinit()

deploy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Finally, use model_deploy.py to complete the model deployment. Refer to the documentation for specific usage instructions.(:ref:`model_deploy <model_deploy>`)。
