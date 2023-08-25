.. _quantization:

=================================
Quantization and optimization
=================================

In deploying neuron network, the accuracy and throughput (inference speed) are critical targets. To achieve high accuracy and high speed, for some networks, mix precision 
inference is essential. In this chapter, with yolo as examples, method of setting mix precision inference is demonstrated, and two useful 
tools 'sensitive layer search' and 'local non-quantization' are illustrated.

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

    orange:72.9.0%

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
   :widths: 23 8 50
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
     - The platform that the model will use. Support bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x.
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
   * - global_compare_layers
     - N
     - global compare layers, for example:\'layer1,layer2\' or \'layer1:0.3,layer2:0.7\'
   * - fp_type
     - N
     - float type of mix precision
   * - loss_table
     - N
     - output all loss of layers if each layer is quantized to f16

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

    int8 outputs_cos:0.999115 old
    mix model outputs_cos:0.999517
    Output mix quantization table to yolov3_qtable
    total time:44 second

Above, int8 outputs_cos represents the cos similarity between original network output of int8 model and fp32; mix model outputs_cos represents the cos similarity of network output after mixing precision is used in some layers; total time represents the search time of 44 seconds.
In addition，get quantization table ``yolov3_qtable``, context as below:

.. code-block:: shell

    # op_name   quantize_mode
    model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool F16
    convolution_output10_Conv F16
    model_1/leaky_re_lu_3/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_3/LeakyRelu:0_pooling0_MaxPool F16
    model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_4/LeakyRelu:0_pooling0_MaxPool F16
    model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu F16
    model_1/leaky_re_lu_5/LeakyRelu:0_pooling0_MaxPool F16
    model_1/concatenate_1/concat:0_Concat F16


In the table, first col is layer name, second is quantization type.
Also ``full_loss_table.txt`` is generated, context as blow:

.. code-block:: shell
    :linenos:

    # chip: bm1684x  mix_mode: F16
    ###
    No.0   : Layer: model_1/leaky_re_lu_3/LeakyRelu:0_LeakyRelu                Cos: 0.994063
    No.1   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu                Cos: 0.997447
    No.2   : Layer: model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu                Cos: 0.997450
    No.3   : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu                Cos: 0.997982
    No.4   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool         Cos: 0.998163
    No.5   : Layer: convolution_output11_Conv                                  Cos: 0.998300
    No.6   : Layer: convolution_output9_Conv                                   Cos: 0.999302
    No.7   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_LeakyRelu                Cos: 0.999371
    No.8   : Layer: convolution_output8_Conv                                   Cos: 0.999424
    No.9   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_pooling0_MaxPool         Cos: 0.999574
    No.10  : Layer: convolution_output12_Conv                                  Cos: 0.999784


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
    orange:72.9%

And get image ``yolov3_mix.jpg`` , as below( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny mix

It can be seen that targets that cannot be detected in int8 model can be detected again with the use of mixing precision.


Sensitive Layer Search
========================

This chapter takes ``mobilenet-v2`` as example to introduce how to use sensitive layer search.
This model is from <nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt>.

This chapter requires the following files (where xxxx corresponds to the actual version information):

**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

Load tpu-mlir
------------------

.. include:: env_var.rst

Prepare working directory
---------------------------

Create a ``mobilenet-v2`` directory, note that it is the same level as tpu-mlir, and put both model files and image files into the ``mobilenet-v2`` directory.

The operation is as follows:

.. code-block:: shell
  :linenos:

   $ mkdir mobilenet-v2 && cd mobilenet-v2
   $ cp -rf $TPUC_ROOT/regression/dataset/ILSVRC2012 .
   $ mkdir workspace && cd workspace

``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.
Note that mobilenet-v2.pt needs to be downloaded from nnmodels and then be placed in the mobilenet-v2 directory.

Accuracy test of float anf int8 models
---------------------------------------

Step 1: To F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_transform.py \
       --model_name mobilenet_v2 \
       --model_def ../mobilenet_v2.pt \
       --input_shapes [[1,3,224,224]] \
       --resize_dims 256,256 \
       --mean 123.675,116.28,103.53 \
       --scale 0.0171,0.0175,0.0174 \
       --pixel_format rgb \
       --mlir mobilenet_v2.mlir

Step 2: Gen calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

Step 3: To F32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --chip bm1684 \
       --model mobilenet_v2_1684_f32.bmodel

Step 4: To INT8 model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table mobilenet_v2_cali_table \
       --model mobilenet_v2_bm1684_int8_sym.bmodel

Step 5: Accuracy test
~~~~~~~~~~~~~~~~~~~~~~

``classify_mobilenet_v2.py`` is a python program, to run ``mobilenet-v2`` model.

Test the fp32 model:

.. code-block:: shell

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_f32.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_fp32_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The classification information is displayed on the output image. The right label ``tench, Tinca tinca`` ranks first.

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
    n02422106 hartebeest
    n02749479 assault rifle, assault gun
    n02916936 bulletproof vest

Test the INT8 model:

.. code-block:: shell

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_int8_sym.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The right label ``tench, Tinca tinca`` ranks second.

.. code-block:: shell

    Top-5
    n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis
    n01440764 tench, Tinca tinca
    n01871265 tusker
    n02396427 wild boar, boar, Sus scrofa
    n02074367 dugong, Dugong dugon

To Mix Precision Model
-----------------------

After int8 conversion, do these commands as beflow.

Step 1: Search sensitive layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``run_sensitive_layer.py`` and bad cases to search sensitive layers, parameters as below:

.. list-table:: run_sensitive_layer.py parameters
   :widths: 23 8 50
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
     - The platform that the model will use. Support bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x.
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - input_num
     - N
     - The number of samples used for calibration, default 10
   * - inference_num
     - N
     - The number of samples used for inference, default 10
   * - max_float_layers
     - N
     - The number of layers set to float, default 5
   * - tune_list
     - N
     - The sample list for tune threshold
   * - tune_num
     - N
     - The number of samples for tune threshold, default 5
   * - histogram_bin_num
     - N
     - The number of bins used in kld calibration, default 2048
   * - post_process
     - N
     - The user defined prost process program path, default None
   * - expected_cos
     - N
     - Specify the minimum cos value for the expected final output layer of the network. The default is 0.99. The smaller the value, the more layers may be set to floating-point
   * - debug_cmd
     - N
     - Specifies a debug command string for development. It is empty by default
   * - o
     - Y
     - output quantization table
   * - global_compare_layers
     - N
     - global compare layers, for example:\'layer1,layer2\' or \'layer1:0.3,layer2:0.7\'
   * - fp_type
     - N
     - float type of mix precision

In this example, 100 images are used for calibration and 30 images are used for inference, and the command is as follows (for the chip of CV18xx series, set the chip to the corresponding chip name) :

The operation is as follows:

.. code-block:: shell

   $ run_sensitive_layer.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --calibration_table mobilenet_v2_cali_table \
       --chip bm1684 \
       --post_process post_process_func.py \
       -o mobilenet_v2_qtable

Sensitive layer program supports user defined post process programs ``post_process_func.py``. The post process function must be named ``PostProcess``.

.. code-block:: shell

   $ def PostProcess(data):
       print("in post process")
       return data

The final output after execution is printed as follows:

.. code-block:: shell

    the layer input3.1 is 0 sensitive layer, loss is 0.008808857469573828, type is top.Conv
    the layer input11.1 is 1 sensitive layer, loss is 0.0016958347875666302, type is top.Conv
    the layer input128.1 is 2 sensitive layer, loss is 0.0015641432811860367, type is top.Conv
    the layer input130.1 is 3 sensitive layer, loss is 0.0014325751094084183, type is top.Scale
    the layer input127.1 is 4 sensitive layer, loss is 0.0011817314259702227, type is top.Add
    the layer input13.1 is 5 sensitive layer, loss is 0.001018420214596527, type is top.Scale
    the layer 787 is 6 sensitive layer, loss is 0.0008603856180608993, type is top.Scale
    the layer input2.1 is 7 sensitive layer, loss is 0.0007558935451825732, type is top.Scale
    the layer input119.1 is 8 sensitive layer, loss is 0.000727441637624282, type is top.Add
    the layer input0.1 is 9 sensitive layer, loss is 0.0007138056757098887, type is top.Conv
    the layer input110.1 is 10 sensitive layer, loss is 0.000662179506136229, type is top.Conv
    ......
    run result:
    int8 outputs_cos:0.978847 old
    mix model outputs_cos:0.989741
    Output mix quantization table to mobilenet_v2_qtable
    total time:402.15848112106323
    success sensitive layer search

Above, int8 outputs_cos represents the cosine similarity between network outputs of int8 model and float model; mix model outputs_cos represents the cosine similarity between network outputs of mix model and float model; total time represents the search time is 402 seconds.
In addition，this program generates a quantization table ``mobilenet_v2_qtable``, the context is as below:

.. code-block:: shell

    # op_name   quantize_mode
    input3.1 F32
    input11.1 F32
    input128.1 F32
    input130.1 F32
    input127.1 F32

The first column in the table is layer name, and the second one is quantization type.
Also a log file named ``SensitiveLayerSearch`` is generated, its context is as blow:

.. code-block:: shell
    :linenos:

    INFO:root:start to handle layer: input3.1, type: top.Conv
    INFO:root:adjust layer input3.1 th, with method MAX, and threshlod 5.5119305
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:outputs_cos_los = 0.014830573787862011
    INFO:root:adjust layer input3.1 th, with method Percentile9999, and threshlod 4.1202815
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:outputs_cos_los = 0.011843443367980822
    INFO:root:adjust layer input3.1 th, with method KL, and threshlod 2.6186381997094728
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:outputs_cos_los = 0.008808857469573828
    INFO:root:layer input3.1, layer type is top.Conv, best_th = 2.6186381997094728, best_method = KL, best_cos_loss = 0.008808857469573828

This log file records the cosine losses between the outputs of mix model and float model when setting each op to int8 with different quantize methods(MAX/Percentile9999/KL).
It also contains the loss information printed in the screen and the cosine similarity of mix model and float model.
The qtable generated by this program can be modified according to the loss information.
The best thresholds of each op are recorded in a new cali table named new_cali_table. This table is restored in current workspace and need to be used when generating mix model.
In this example, the loss of input3.1 is larger than other ops, thus you can only set input3.1 as float in qtable.

Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table new_cali_table \
       --quantize_table mobilenet_v2_qtable \
       --model mobilenet_v2_bm1684_int8_mix.bmodel

Step 3: Test accuracy of mix model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_mix.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The classification results are as follows. The right label ``tench, Tinca tinca`` ranks first again.

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02749479 assault rifle, assault gun
    n02916936 bulletproof vest
    n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
    n04090263 rifle


Local Non-Quantization
==============================


For specific neural networks, some layers may not be suitable for quantization due to significant differences in data distribution. The "Local Non-Quantization" allows you to add certain layers before, after, or between other layers to a mixed-precision table. These layers will not be quantized when generating a mixed-precision model.

In this chapter, we will continue using the example of the YOLOv5s network mentioned in Chapter 3 and demonstrate how to use the Local Non-Quantization to quickly generate a mix-precision model.

The process of generating FP32 and INT8 models is the same as in Chapter 3. Here, we focus on generating mix-precision model and the accuracy testing.

For YOLO series models, the last three convolutional layers often have significantly different data distributions, and adding them manually to the mixed-precision table can improve accuracy. With the Local Non-Quantization feature, you can search for the corresponding layers from the FP32 MLIR file and quickly add them to the mixed-precision table using the following command:

.. code-block:: shell

   $ fp_forward.py \
       yolov5s.mlir \
       --quantize INT8 \
       --chip bm1684x \
       --fpfwd_outputs 474_Conv,326_Conv,622_Conv\
       --chip bm1684x \
       -o yolov5s_qtable

Opening the file "yolov5s_qtable" will reveal that the relevant layers have been added to the qtable.

Generating the Mixed-Precision Model

.. code-block:: shell

  $ model_deploy.py \
      --mlir yolov5s.mlir \
      --quantize INT8 \
      --calibration_table yolov5s_cali_table \
      --quantize_table yolov5s_qtable \
      --chip bm1684x \
      --test_input yolov5s_in_f32.npz \
      --test_reference yolov5s_top_outputs.npz \
      --tolerance 0.85,0.45 \
      --model yolov5s_1684x_mix.bmodel

Validating the Accuracy of FP32 and Mixed-Precision Models
In the model-zoo, there is a program called "yolo" used for accuracy validation of object detection models. You can use the "harness" field in the mlir.config.yaml file to invoke "yolo" as follows:

Modify the relevant fields as follows:

.. code-block:: shell

  $ dataset:
      imagedir: $(coco2017_val_set)
      anno: $(coco2017_anno)/instances_val2017.json

  harness:
      type: yolo
      args:
          - name: FP32
          bmodel: $(workdir)/$(name)_bm1684_f32.bmodel
          - name: INT8
          bmodel: $(workdir)/$(name)_bm1684_int8_sym.bmodel
          - name: mix
          bmodel: $(workdir)/$(name)_bm1684_mix.bmodel

Switch to the top-level directory of model-zoo and use tpu_perf.precision_benchmark for accuracy testing, as shown in the following command:
.. code-block:: shell

  $ python3 -m tpu_perf.precision_benchmark yolov5s_path --mlir --target BM1684X --devices 0

The accuracy test results will be stored in output/yolo.csv:

mAP for the FP32 model:
mAP for the mixed-precision model using the default mixed-precision table:

Performance Testing

mAP for the mixed-precision model using the manually added mixed-precision table:

Parameter Description


.. list-table:: fp_forward.py parameters
   :widths: 23 8 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - mlir file
   * - chip
     - Y
     - The platform that the model will use. Support bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x.
   * - fpfwd_inputs
     - N
     - Specify layers (including this layer) to skip quantization before them. Multiple inputs are separated by commas.
   * - fpfwd_outputs
     - N
     - Specify layers (including this layer) to skip quantization after them. Multiple inputs are separated by commas.
   * - fpfwd_blocks
     - N
     - Specify the start and end layers between which quantization will be skipped. Start and end layers are separated by space, and multiple blocks are separated by spaces.
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - o
     - Y
     - output quantization table
