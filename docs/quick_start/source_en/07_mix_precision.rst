Mix Precision
==================

This chapter takes ``yolov3 tiny`` as examples to introduce how to use mix precision。
This model is from <https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3>。

This chapter requires the following files (where xxxx corresponds to the actual version information):

**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

Load tpu-mlir
------------------

.. include:: env_var.rst

Prepare working directory
---------------------------

Create a ``yolov3_tiny`` directory, note that it is the same level as tpu-mlir, and put both model files and image files into the ``yolov3_tiny`` directory.

The operation is as follows:

.. code-block:: shell
  :linenos:

   $ mkdir yolov3_tiny && cd yolov3_tiny
   $ wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ mkdir workspace && cd workspace

``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.


Sample for onnx
-------------------

``detect_yolov3.py`` is a python program, to run ``yolov3_tiny`` model.

The operation is as follows:

.. code-block:: shell

   $ detect_yolov3.py \
        --model ../tiny-yolov3-11.onnx \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_onnx.jpg

The print result as follows:

.. code-block:: shell

    person:60.7%
    orange:77.5%

And get result image ``yolov3_onnx.jpg``, as below( :ref:`yolov3_onnx_result` ):

.. _yolov3_onnx_result:
.. figure:: ../assets/yolov3_onnx.jpg
   :height: 13cm
   :align: center

   yolov3_tiny ONNX


To INT8 symmetric model
-------------------------

Step 1: To F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_transform.py \
       --model_name yolov3_tiny \
       --model_def ../tiny-yolov3-11.onnx \
       --input_shapes [[1,3,416,416]] \
       --scale 0.0039216,0.0039216,0.0039216 \
       --pixel_format rgb \
       --keep_aspect_ratio \
       --pad_value 128 \
       --output_names=convolution_output1,convolution_output \
       --mlir yolov3_tiny.mlir

Step 2: Gen calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table

Step 3: To model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_int8.bmodel

Step 4: Run model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3.py \
        --model yolov3_int8.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_int8.jpg

The print result as follows, indicates that one target is detected:

.. code-block:: shell

    orange:73.0%

And get image ``yolov3_int8.jpg``, as below( :ref:`yolov3_int8_result` ):

.. _yolov3_int8_result:
.. figure:: ../assets/yolov3_int8.jpg
   :height: 13cm
   :align: center

   yolov3_tiny int8 symmetric

It can be seen that the int8 symmetric quantization model performs poorly compared to the original model on this image and only detects one target.

To Mix Precision Model
-----------------------

After int8 conversion, do these commands as beflow.

Step 1: Gen quantization table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``run_qtable.py`` to gen qtable, parameters as below:

.. list-table:: run_qtable.py parameters
   :widths: 18 10 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - mlir file
   * - dataset
     - N
     - Directory of input samples. Images, npz or npy files are placed in this directory
   * - data_list
     - N
     - The sample list (cannot be used together with "dataset")
   * - calibration_table
     - Y
     - Name of calibration table file
   * - chip
     - Y
     - The platform that the model will use. Support bm1684x/bm1684/cv183x/cv182x/cv181x/cv180x.
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - input_num
     - N
     - The number of sample, default 10
   * - expected_cos
     - N
     - Specify the minimum cos value for the expected final output layer of the network. The default is 0.99. The smaller the value, the more layers may be set to floating-point
   * - min_layer_cos
     - N
     - Specify the minimum cos expected per layer, below which an attempt is made to set the fp32 calculation. The default is 0.99
   * - debug_cmd
     - N
     - Specifies a debug command string for development. It is empty by default
   * - o
     - Y
     - output quantization table

The operation is as follows:

.. code-block:: shell

   $ run_qtable.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --min_layer_cos 0.999 \ #If the default 0.99 is used here, the program detects that the original int8 model already meets the cos of 0.99 and simply stops searching
       --expected_cos 0.9999 \
       --chip bm1684x \
       -o yolov3_qtable

The final output after execution is printed as follows:

.. code-block:: shell

    int8 outputs_cos:0.999317
    mix model outputs_cos:0.999739
    Output mix quantization table to yolov3_qtable
    total time:44 second

Above, int8 outputs_cos represents the cos similarity between original network output of int8 model and fp32; mix model outputs_cos represents the cos similarity of network output after mixing precision is used in some layers; total time represents the search time of 44 seconds.
In addition，get quantization table ``yolov3_qtable``, context as below:

.. code-block:: shell

    # op_name   quantize_mode
    convolution_output11_Conv F16
    model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool F16
    convolution_output10_Conv F16
    convolution_output9_Conv F16
    model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_5/LeakyRelu:0_pooling0_MaxPool F16
    model_1/concatenate_1/concat:0_Concat F16


In the table, first col is layer name, second is quantization type.
Also ``full_loss_table.txt`` is generated, context as blow:

.. code-block:: shell
    :linenos:

    # chip: bm1684x  mix_mode: F16
    ###
    No.0   : Layer: convolution_output11_Conv                            Cos: 0.984398
    No.1   : Layer: model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu          Cos: 0.998341
    No.2   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool   Cos: 0.998500
    No.3   : Layer: convolution_output9_Conv                             Cos: 0.998926
    No.4   : Layer: convolution_output8_Conv                             Cos: 0.999249
    No.5   : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_pooling0_MaxPool   Cos: 0.999284
    No.6   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_LeakyRelu          Cos: 0.999368
    No.7   : Layer: model_1/leaky_re_lu_3/LeakyRelu:0_LeakyRelu          Cos: 0.999554
    No.8   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_pooling0_MaxPool   Cos: 0.999576
    No.9   : Layer: model_1/leaky_re_lu_3/LeakyRelu:0_pooling0_MaxPool   Cos: 0.999723
    No.10  : Layer: convolution_output12_Conv                            Cos: 0.999810


This table is arranged smoothly according to the cos from small to large, indicating the cos calculated
by this Layer after the precursor layer of this layer has been changed to the corresponding floating-point mode.
If the cos is still smaller than the previous parameter min_layer_cos, this layer and its immediate successor
layer will be set to floating-point calculation。
``run_qtable.py`` calculates the output cos of the whole network every time the neighboring two layers are set
to floating point. If the cos is larger than the specified expected_cos, the search is withdrawn. Therefore,
if you set a larger expected_cos value, you will try to set more layers to floating point。


Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel

Step 3: run mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3.py \
        --model yolov3_mix.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_mix.jpg

The print result as follows:

.. code-block:: shell

    person:63.9%
    orange:73.0%

And get image ``yolov3_mix.jpg`` , as below( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny mix

It can be seen that targets that cannot be detected in int8 model can be detected again with the use of mixing precision.
