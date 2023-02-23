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
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_onnx.jpg

The print result as follows:

.. code-block:: shell

    car:81.7%
    car:72.6%
    car:71.1%
    car:66.0%
    bus:69.5%

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
       --output_names=transpose_output1,transpose_output \
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
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_int8.jpg

The print result as follows:

.. code-block:: shell

  car:79.0%
  car:72.4%
  bus:65.8%

And get image ``yolov3_int8.jpg``, as below( :ref:`yolov3_int8_result` ):

.. _yolov3_int8_result:
.. figure:: ../assets/yolov3_int8.jpg
   :height: 13cm
   :align: center

   yolov3_tiny int8 symmetric

Compared with onnx result, int8 model has large loss.

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

And get quantization table ``yolov3_qtable``, context as below:

.. code-block:: shell

  # op_name   quantize_mode
  convolution_output11_Conv F32
  model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu F32
  model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool F32
  convolution_output10_Conv F32
  model_1/leaky_re_lu_6/LeakyRelu:0_LeakyRelu F32
  model_1/leaky_re_lu_6/LeakyRelu:0_pooling0_MaxPool F32
  model_1/leaky_re_lu_7/LeakyRelu:0_LeakyRelu F32
  convolution_output5_Conv F32
  model_1/leaky_re_lu_8/LeakyRelu:0_LeakyRelu F32
  convolution_output4_Conv F32
  convolution_output3_Conv F32


In the table, first col is layer name, second is quantization type.
Also ``full_loss_table.txt`` is generated, context as blow:

.. code-block:: shell
    :linenos:

    # chip: bm1684x  mix_mode: F32
    ###
    No.0   : Layer: convolution_output11_Conv                                               Cos: 0.9923188653689166
    No.1   : Layer: model_1/leaky_re_lu_8/LeakyRelu:0_LeakyRelu                             Cos: 0.9982724675923477
    No.2   : Layer: model_1/leaky_re_lu_7/LeakyRelu:0_LeakyRelu                             Cos: 0.9984222695482265
    No.3   : Layer: model_1/leaky_re_lu_6/LeakyRelu:0_LeakyRelu                             Cos: 0.998515580396405
    No.4   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool                      Cos: 0.9987678931990402
    No.5   : Layer: model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu                             Cos: 0.9990712074303405
    No.6   : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu                             Cos: 0.999284826478191
    No.7   : Layer: model_1/leaky_re_lu_5/LeakyRelu:0_pooling0_MaxPool                      Cos: 0.9993153210002395
    No.8   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_LeakyRelu                             Cos: 0.9993530523531371
    No.9   : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_pooling0_MaxPool                      Cos: 0.9995473722523207
    No.10  : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_pooling0_MaxPool                      Cos: 0.999551823932271
    No.11  : Layer: convolution_output9_Conv                                                Cos: 0.9995627192000597
    No.12  : Layer: convolution_output6_Conv                                                Cos: 0.999667275119983
    No.13  : Layer: model_1/leaky_re_lu_3/LeakyRelu:0_LeakyRelu                             Cos: 0.9996674835174093
	....
	
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
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_mix.jpg

The print result as follows:

.. code-block:: shell

    car:78.7%
    car:68.8%
    car:63.1%
    bus:65.3%

And get image ``yolov3_mix.jpg`` , as below( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny mix

Compared to int8 model, mix model is better.
