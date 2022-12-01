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
   * - o
     - Y
     - output quantization table

The operation is as follows:

.. code-block:: shell

   $ run_qtable.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       -o yolov3_qtable

And get quantization table ``yolov3_qtable``, context as below:

.. code-block:: shell

  # op_name   quantize_mode
  convolution_output11_Conv F32


In the table, first col is layer name, second is quantization type.
Also ``full_loss_table.txt`` is generated, context as blow:

.. code-block:: shell
    :linenos:

    # all int8 loss: -17.297552609443663
    # chip: bm1684x  mix_mode: F32
    No.0 : Layer: convolution_output11_Conv                    Loss: -15.913658332824706
    No.1 : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu  Loss: -17.148419880867003
    No.2 : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu  Loss: -17.241489434242247
    No.3 : Layer: model_1/concatenate_1/concat:0_Concat        Loss: -17.263980317115784
    No.4 : Layer: model_1/leaky_re_lu_10/LeakyRelu:0_LeakyRelu Loss: -17.275933575630187
    No.5 : Layer: convolution_output4_Conv                     Loss: -17.288181042671205
    No.6 : Layer: model_1/leaky_re_lu_9/LeakyRelu:0_LeakyRelu  Loss: -17.289376521110533
    No.7 : Layer: model_1/leaky_re_lu_11/LeakyRelu:0_LeakyRelu Loss: -17.295218110084534
    ......

This table is ordered by loss from small to large. Smaller is better, if layer convert to float type.
``run_qtable.py`` use layers that have 5% improvement.
If application performs not good, you can also add more layers to quantization table.


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
