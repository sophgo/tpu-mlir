Compile the ONNX model
======================

This chapter takes ``yolov5s.onnx`` as an example to introduce how to compile and transfer an onnx model to run on the BM1684X TPU platform.

The model is from the official website of yolov5: https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx

This chapter requires the following files (where xxxx corresponds to the actual version information):


**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

.. list-table::
   :widths: 35 20 30
   :header-rows: 1

   * - platform
     - file name
     - info
   * - cv183x/cv182x/cv181x/cv180x
     - xxx.cvimodel
     - please refer to the :ref:`CV18xx Guidance <onnx to cvimodel>`
   * - 其它
     - xxx.bmodel
     - please refer to the :ref:`following <onnx to bmodel>`

.. _onnx to bmodel:

Load tpu-mlir
------------------

.. include:: env_var.rst


Prepare working directory
-------------------------

Create a ``model_yolov5s`` directory, note that it is the same level directory as tpu-mlir; and put both model files and image files
into the ``model_yolov5s`` directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s && cd model_yolov5s
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.


ONNX to MLIR
------------------

If the input is image, we need to know the preprocessing of the model before transferring it. If the model uses preprocessed npz files as input, no preprocessing needs to be considered.
The preprocessing process is formulated as follows ( :math:`x` represents the input):

.. math::

   y = (x - mean) \times scale


The image of the official yolov5 is rgb. Each value will be multiplied by ``1/255``, respectively corresponding to
``0.0,0.0,0.0`` and ``0.0039216,0.0039216,0.0039216`` when it is converted into mean and scale.

The model conversion command is as follows:


.. code-block:: shell

   $ model_transform.py \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 350,498,646 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

.. _model_transform param:

The main parameters of ``model_transform.py`` are described as follows (for a complete introduction, please refer to the user interface chapter of the TPU-MLIR Technical Reference Manual):


.. list-table:: Function of model_transform parameters
   :widths: 20 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - model_name
     - Y
     - Model name
   * - model_def
     - Y
     - Model definition file (e.g., '.onnx', '.tflite' or '.prototxt' files)
   * - input_shapes
     - N
     - Shape of the inputs, such as [[1,3,640,640]] (a two-dimensional array), which can support multiple inputs
   * - input_types
     - N
     - Type of the inputs, such int32; separate by ',' for multi inputs; float32 as default
   * - resize_dims
     - N
     - The size of the original image to be adjusted to. If not specified, it will be resized to the input size of the model
   * - keep_aspect_ratio
     - N
     - Whether to maintain the aspect ratio when resize. False by default. It will pad 0 to the insufficient part when setting
   * - mean
     - N
     - The mean of each channel of the image. The default is 0.0,0.0,0.0
   * - scale
     - N
     - The scale of each channel of the image. The default is 1.0,1.0,1.0
   * - pixel_format
     - N
     - Image type, can be rgb, bgr, gray or rgbd. The default is bgr
   * - channel_format
     - N
     - Channel type, can be nhwc or nchw for image input, otherwise it is none. The default is nchw
   * - output_names
     - N
     - The names of the output. Use the output of the model if not specified, otherwise use the specified names as the output
   * - test_input
     - N
     - The input file for validation, which can be an image, npy or npz. No validation will be carried out if it is not specified
   * - test_result
     - N
     - Output file to save validation result
   * - excepts
     - N
     - Names of network layers that need to be excluded from validation. Separated by comma
   * - mlir
     - Y
     - The output mlir file name (including path)


After converting to an mlir file, a ``${model_name}_in_f32.npz`` file will be generated, which is the input file for the subsequent models.


MLIR to F16 bmodel
------------------

To convert the mlir file to the f16 bmodel, we need to run:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --chip bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model yolov5s_1684x_f16.bmodel

.. _model_deploy param:

The main parameters of ``model_deploy.py`` are as follows (for a complete introduction, please refer to the user interface chapter of the TPU-MLIR Technical Reference Manual):


.. list-table:: Function of model_deploy parameters
   :widths: 18 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - mlir
     - Y
     - Mlir file
   * - quantize
     - Y
     - Quantization type (F32/F16/BF16/INT8)
   * - chip
     - Y
     - The platform that the model will use. Support bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x.
   * - calibration_table
     - N
     - The calibration table path. Required when it is INT8 quantization
   * - tolerance
     - N
     - Tolerance for the minimum similarity between MLIR quantized and MLIR fp32 inference results
   * - test_input
     - N
     - The input file for validation, which can be an image, npy or npz. No validation will be carried out if it is not specified
   * - test_reference
     - N
     - Reference data for validating mlir tolerance (in npz format). It is the result of each operator
   * - compare_all
     - N
     - Compare all tensors, if set.
   * - excepts
     - N
     - Names of network layers that need to be excluded from validation. Separated by comma
   * - op_divide
     - N
     - cv183x/cv182x/cv181x/cv180x only, Try to split the larger op into multiple smaller op to achieve the purpose of ion memory saving, suitable for a few specific models
   * - model
     - Y
     - Name of output model file (including path)
   * - core
     - N
     - When the target is selected as bm1686 or cv186x, it is used to select the number of tpu cores for parallel computing, and the default setting is 1 tpu core


After compilation, a file named ``yolov5s_1684x_f16.bmodel`` is generated.


MLIR to INT8 bmodel
-------------------

Calibration table generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before converting to the INT8 model, you need to run calibration to get the calibration table. The number of input data is about 100 to 1000 according to the situation.

Then use the calibration table to generate a symmetric or asymmetric bmodel. It is generally not recommended to use the asymmetric one if the symmetric one already meets the requirements, because
the performance of the asymmetric model will be slightly worse than the symmetric model.

Here is an example of the existing 100 images from COCO2017 to perform calibration:


.. code-block:: shell

   $ run_calibration.py yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

After running the command above, a file named ``yolov5s_cali_table`` will be generated, which is used as the input file for subsequent compilation of the INT8 model.


Compile to INT8 symmetric quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the following command to convert to the INT8 symmetric quantized model:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --chip bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_1684x_int8_sym.bmodel

After compilation, a file named ``yolov5s_1684x_int8_sym.bmodel`` is generated.


Effect comparison
----------------------

There is a yolov5 use case written in python in this release package for object detection on images. The source code path is ``$TPUC_ROOT/python/samples/detect_yolov5.py``. It can be learned how the model is used by reading the code. Firstly, preprocess to get the model's input, then do inference to get the output, and finally do post-processing.
Use the following codes to validate the inference results of onnx/f16/int8 respectively.


The onnx model is run as follows to get ``dog_onnx.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model ../yolov5s.onnx \
       --output dog_onnx.jpg


The f16 bmodel is run as follows to get ``dog_f16.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_1684x_f16.bmodel \
       --output dog_f16.jpg



The int8 symmetric bmodel is run as follows to get ``dog_int8_sym.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_1684x_int8_sym.bmodel \
       --output dog_int8_sym.jpg


The result images are compared as shown in the figure (:ref:`yolov5s_result`).

.. _yolov5s_result:
.. figure:: ../assets/yolov5s.png
   :height: 13cm
   :align: center

   Comparison of TPU-MLIR for YOLOv5s' compilation effect

Due to different operating environments, the final performance will be somewhat different from :numref:`yolov5s_result`.


Model performance test
----------------------

The following operations need to be performed outside of Docker,

Install the ``libsophon``
~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the ``libsophon`` manual to install ``libsophon``.


Check the performance of ``BModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installing ``libsophon``, you can use ``bmrt_test`` to test the accuracy and performance of the ``bmodel``. You can choose a suitable model by estimating the maximum fps of the model based on the output of ``bmrt_test``.

.. code-block:: shell

   # Test the bmodel compiled above
   # --bmodel parameter followed by bmodel file,

   $ cd $TPUC_ROOT/../model_yolov5s/workspace
   $ bmrt_test --bmodel yolov5s_1684x_f16.bmodel
   $ bmrt_test --bmodel yolov5s_1684x_int8_sym.bmodel


Take the output of the last command as an example (the log is partially truncated here):

.. code-block:: shell
   :linenos:

   [BMRT][load_bmodel:983] INFO:pre net num: 0, load net num: 1
   [BMRT][show_net_info:1358] INFO: ########################
   [BMRT][show_net_info:1359] INFO: NetName: yolov5s, Index=0
   [BMRT][show_net_info:1361] INFO: ---- stage 0 ----
   [BMRT][show_net_info:1369] INFO:   Input 0) 'images' shape=[ 1 3 640 640 ] dtype=FLOAT32
   [BMRT][show_net_info:1378] INFO:   Output 0) '350_Transpose_f32' shape=[ 1 3 80 80 85 ] ...
   [BMRT][show_net_info:1378] INFO:   Output 1) '498_Transpose_f32' shape=[ 1 3 40 40 85 ] ...
   [BMRT][show_net_info:1378] INFO:   Output 2) '646_Transpose_f32' shape=[ 1 3 20 20 85 ] ...
   [BMRT][show_net_info:1381] INFO: ########################
   [BMRT][bmrt_test:770] INFO:==> running network #0, name: yolov5s, loop: 0
   [BMRT][bmrt_test:834] INFO:reading input #0, bytesize=4915200
   [BMRT][print_array:702] INFO:  --> input_data: < 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
   [BMRT][bmrt_test:982] INFO:reading output #0, bytesize=6528000
   [BMRT][print_array:702] INFO:  --> output ref_data: < 0 0 0 0 0 0 0 0 0 0 0 0 0 0...
   [BMRT][bmrt_test:982] INFO:reading output #1, bytesize=1632000
   [BMRT][print_array:702] INFO:  --> output ref_data: < 0 0 0 0 0 0 0 0 0 0 0 0 0 0...
   [BMRT][bmrt_test:982] INFO:reading output #2, bytesize=408000
   [BMRT][print_array:702] INFO:  --> output ref_data: < 0 0 0 0 0 0 0 0 0 0 0 0 0 0...
   [BMRT][bmrt_test:1014] INFO:net[yolov5s] stage[0], launch total time is 4122 us (npu 4009 us, cpu 113 us)
   [BMRT][bmrt_test:1017] INFO:+++ The network[yolov5s] stage[0] output_data +++
   [BMRT][print_array:702] INFO:output data #0 shape: [1 3 80 80 85 ] < 0.301003    ...
   [BMRT][print_array:702] INFO:output data #1 shape: [1 3 40 40 85 ] < 0 0.228689  ...
   [BMRT][print_array:702] INFO:output data #2 shape: [1 3 20 20 85 ] < 1.00135     ...
   [BMRT][bmrt_test:1058] INFO:load input time(s): 0.008914
   [BMRT][bmrt_test:1059] INFO:calculate  time(s): 0.004132
   [BMRT][bmrt_test:1060] INFO:get output time(s): 0.012603
   [BMRT][bmrt_test:1061] INFO:compare    time(s): 0.006514


The following information can be learned from the output above:

1. Lines 05-08: the input and output information of bmodel
2. Line 19: running time on the TPU, of which the TPU takes 4009us and the CPU takes 113us. The CPU time here mainly refers to the waiting time of calling at HOST
3. Line 24: the time to load data into the NPU's DDR
4. Line 25: the total time of Line 19
5. Line 26: the output data retrieval time
