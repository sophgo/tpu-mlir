.. _quantization:

=================================
Quantization and optimization
=================================

In deploying neuron network, the accuracy and throughput (inference speed) are critical targets. To achieve high accuracy and high speed, for some networks, mix precision inference is essential.

The mixed-precision method of TPU-MLIR is searching layers in neural network that are not suitable for low-bit quantization to generate a quantize table, which is used to specify these layers to use higher-bit quantization in the model_deploy stage.
This chapter will first introduce the current full int8 symmetric quantization of TPU-MLIR, and then explain how to use the existing quantize table automatic generation tools in TPU-MLIR.

TPU-MLIR Full Int8 Symmetric Quantization
===========================================

TPU-MLIR adopts full int8 symmetric quantization by default, where full int8 means that all operators, except for those that the compiler defaults to floating-point operations (such as ``layernorm``), are quantized to int8. This section introduces how to use the TPU-MLIR full int8 symmetric quantization tool.

After generating the corresponding MLIR file for the model using the ``model_transform`` command as instructed in the previous tutorial, if you want to perform int8 symmetric quantization on the model,
you also need to generate a calibration table ``cali_table`` using the ``run_calibration`` command. How to use the parameters of the ``run_calibration`` command for different types of models to ensure
the generated quantized model has good accuracy will be provided in detailed guidance below.

run_calibration Process Introduction
---------------------------------------

The quantization part of the following figure (:ref:calibration) shows the overall process of the current ``run_calibration`` , which includes the automatic mixed-precision module ``search_qtable`` , the automatic calibration method selection module ``search_threshold`` , cross-layer weight equalization module ``weight_equalization``,
and bias correction module ``bias_correction``, etc. In the following sections, we will provide the usage details of the above methods based on actual situations.

.. _calibration:
.. figure:: ../assets/quant.png
   :height: 20cm
   :align: center

   run_calibration process

run_calibration Parameter Introduction
----------------------------------------

The table below provides an introduction to the parameters of the ``run_calibration`` command.

.. list-table:: run_calibration.py parameter
   :widths: 25 60
   :header-rows: 1

   * - parameter
     - description
   * - mlir_file
     - mlir file
   * - we
     - open weight_equalization
   * - bc
     - open bias_correction
   * - dataset
     - calibration dataset
   * - data_list
     - sample list
   * - input_num
     - number of calibration sample
   * - inference_num
     - the number of images required for the inference process with search_qtable and search_threshold is set to 30 by default
   * - bc_inference_num
     - the number of images required for the inference process with bias correction is set to 30 by default
   * - tune_list
     - the list of sample used for tuning
   * - tune_num
     - the number of images for tuning
   * - histogram_bin_num
     - specify the number of histogram bins for KLD calculation, default is 2048
   * - expected_cos
     - expect the similarity between the mixed-precision model output of search_qtable and the floating-point model output, with a value range of [0,1], default is 0.99
   * - min_layer_cos
     - the lower bound of similarity between the quantized output and the floating-point output of the layer in bias_correction; compensation is required for the layer when it falls below this threshold, with a value range of [0,1], default is 0.99
   * - max_float_layers
     - set the number of floating-point layers for search_qtable, default is 5
   * - chip
     - chip type, default is bm1684x
   * - cali_method
     - select the calibration mode; if this parameter is not added, the default is KLD calibration. "use_percentile9999" uses the 99.99 percentile as the threshold. "use_max" uses the absolute maximum value as the threshold. "use_torch_observer_for_cali" uses torch's observer for calibration. "use_mse" uses octav for calibration.
   * - fp_type
     - search_qtable floating-point layer data type
   * - post_process
     - post-processing path
   * - global_compare_layers
     - specify the global contrastive layers, for example, layer1,layer2 or layer1:0.3,layer2:0.7.
   * - search
     - specify the search type, which includes search_qtable, search_threshold, and false. The default is false, which means search is not enabled
   * - transformer
     - whether it is a transformer model, in search_qtable, if it is a transformer model, a specific acceleration strategy can be assigned, default is False
   * - quantize_method_list
     - the calibration method used for searching in search_qtable, default is MSE, with selectable range being MSE, KL, MAX, Percentile9999
   * - benchmark_method
     - specify the similarity calculation method of search_threshold, with the default being cosine similarity (cos)
   * - quantize_table
     - the mixed-precision quantization table from search_qtable
   * - o
     - cali_table output path
   * - debug_cmd
     - debug command
   * - debug_log
     - log output level

The Use of run_calibration Parameters Introduction
--------------------------------------------------------

Based on the user's needs and their understanding of the model itself and quantization, we have provided targeted ways to use the ``run_calibration`` parameters in different situations.

.. list-table:: applicable scenarios for run_calibration parameter
   :widths: 8 25 8 8 25
   :header-rows: 1

   * - scenario
     - description
     - quantization speed
     - calibration method
     - recommended method
   * - case1
     - initial model quantization
     - insensitive
     - unclear
     - search_threshold
   * - case2
     - initial model quantization
     - /
     - clear
     - cali_method directly selects the corresponding calibration method
   * - case3
     - initial model quantization
     - sensitive
     - unclear
     - the cali_method selects a fixed calibration method; for details on choosing a specific calibration method, refer to the subsequent sections
   * - case4
     - after model quantization, the accuracy on the bm1684 chip does not meet the requirements
     - /
     - /
     - open we and bc methods

case 1: When you perform the initial quantization on your model, which is the first time you use the ``run_calibration`` command, you may not be clear
about the calibration method that is best suited for your current model and you may not be sensitive to the quantization speed.
In this case, it is recommended to use the ``search_threshold`` method. This method can automatically select the calibration method that is most suitable
for your current model and output the calibration table ``cali_table`` generated by this method to the output path you specify. It will also generate a log file ``Search_Threshold``,
which records the quantization information for different calibration methods. The specific operation is as follows:

.. code-block:: shell

   $ run_calibration mlir.file \
       --dataset data_path \
       --input_num 100 \
       --chip bm1684x \
       --search search_threshold \
       --inference_num 30 \
       -o cali_table

Notes:1.At this point, it is necessary to select the chip parameter, which corresponds to the chip platform on which the model is intended to be deployed. The current default is bm1684x.
2. ``inference_num`` corresponds to the number of inference data required for the ``search_threshold`` process (this data will be extracted from the dataset you provide).
The larger the ``inference_num``, the more accurate the ``search_threshold`` result, but the longer the quantization time required. Here, the default for ``inference_num`` is set to 30, which can be customized according to the actual situation.

case2: When quantizing your model for the first time, you already know which calibration method is suitable for the model. At this point, you can directly choose a fixed calibration method based on the ``cali_method`` parameter. The specific operation is as follows:

.. code-block:: shell

   $ run_calibration mlir.file \
       --dataset data_path \
       --input_num 100 \
       --cali_method use_mse \
       -o cali_table

Notes:1.when the ``cali_method`` parameter is not added, the default KLD calibration method will be used. 2.currently, the ``cali_method`` supports five options, including ``use_mse``, ``use_max``, ``use_percentile9999``, ``use_aciq_gauss`` and ``use_aciq_laplace``.

case3: When you are sensitive to quantization time and wish to generate the calibration table ``cali_table`` as quickly as possible, but you are unsure how to choose a calibration method, it is recommended to select a fixed calibration method based on the ``cali_method`` parameter.
In comparison to the quantization speed of TPU-MLIR V1.8, the V1.9 version shows a 100% speed improvement for individual calibration methods, resulting in an average time reduction of around 50%. The acceleration effect is significant.
In the V1.9 version, ``use_mse`` is the fastest calibration method on average. When selecting a calibration method, you can consider the following empirical conclusions:

1.For non-transformer models without attention structure, ``use_mse`` is a suitable calibration method. Here is a specific operation guide:

.. code-block:: shell

   $ run_calibration mlir.file \
       --dataset data_path \
       --input_num 100 \
       --cali_method use_mse \
       -o cali_table

You can also choose the default ``KLD`` calibration method. Here is a specific operation guide:

.. code-block:: shell

   $ run_calibration mlir.file \
       --dataset data_path \
       --input_num 100 \
       -o cali_table

If neither of the above two methods meets the accuracy requirements, you may need to consider adopting a mixed precision strategy or a hybrid threshold method. More detailed information on these approaches can be found in the subsequent section.

2.When your model is a transformer model that includes an attention structure, you can choose the ``use_mse`` calibration method. If the ``use_mse`` calibration method does not produce satisfactory results, you can then consider trying the ``use_max`` calibration method. Here is a specific operation guide:

.. code-block:: shell

   $ run_calibration mlir.file \
       --dataset data_path \
       --input_num 100 \
       --cali_method use_max \
       -o cali_table

If the ``use_max`` method also fails to meet the requirements, at this point, you may need to adopt a mixed precision strategy. You can then try the mixed precision methods that will be introduced later.

Apart from the overall selection rules mentioned above, here are some specific details for choosing calibration methods:1.If your model is a YOLO series object detection model, it is recommended to use the default KLD calibration method.2.If your model is a multi-output classification model,
it is also recommended to use the default KLD calibration method.

case4: When your model is deployed on the bm1684 chip and the full int8 quantized model obtained through the methods mentioned above has poor accuracy,
you can try enabling cross-layer weight equalization (``we``) and bias correction (``bc``). To do this, simply add the ``we`` and ``bc`` parameters to the original command.
If you have used ``search_threshold`` for searching, the operations for adding we and bc are as follows:

.. code-block:: shell

   $ run_calibration mlir.file \
       --we \
       --bc \
       --dataset data_path \
       --input_num 100 \
       --chip bm1684 \
       --search search_threshold \
       --inference_num 30 \
       --bc_inference_num 100 \
       -o cali_table

If you choose a fixed calibration method using ``cali_method`` , for example, using ``use_mse`` , to add the ``we`` and ``bc`` methods, the specific operation is as follows:

.. code-block:: shell

   $ run_calibration mlir.file \
       --we \
       --bc \
       --dataset data_path \
       --input_num 100 \
       --chip bm1684 \
       --cali_method use_mse \
       --bc_inference_num 100 \
       -o cali_table

If you are using the default KLD calibration method, simply remove the ``cali_method`` parameter.

Notes:1.Make sure to specify the chip parameter as bm1684. 2.The ``bc_inference_num`` parameter is the number of data samples required when using the ``bc`` quantization method (these samples will be extracted from the dataset you provide), so the number of images should not be too few.
3.The ``we`` and ``bc`` methods can be used independently. If you choose only the ``we`` method, simply omit the ``bc`` parameter in the operation.

Overview of TPU-MLIR Mixed Precision Quantization
==================================================

Currently, TPU-MLIR provides four mixed precision quantization methods: ``search_qtable`` , ``run_qtable`` , ``run_sensitive_layer`` and ``fp_forward``. Among these, ``search_qtable`` is an optimized version of ``run_qtable`` and ``run_sensitive_layer``.
Compared to ``run_qtable`` , ``search_qtable`` offers better effectiveness and typically achieves superior quantization results, albeit at a slightly slower speed. In contrast to ``run_sensitive_layer``, ``search_qtable`` is faster and supports more customizable parameters.
The following section will provide detailed introductions to these four mixed precision tools.

1. search_qtable
=====================

``search_qtable`` is a mixed precision feature integrated into ``run_calibration``. When full int8 quantization precision does not meet the requirements, mixed precision method are needed, meaning that some operators are set to perform floating-point operations.
This section takes ``mobilenet-v2`` as example to introduce how to use ``search_qtable``.

.. This model is from <nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt>.

This section requires the tpu_mlir python package.


Install tpu_mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[all]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[all]

Prepare working directory
---------------------------

.. include:: get_resource.rst

Create a ``mobilenet-v2`` directory, and put both model files and image files into the ``mobilenet-v2`` directory.

The operation is as follows:

.. code-block:: shell
  :linenos:

   $ mkdir mobilenet-v2 && cd mobilenet-v2
   $ wget https://github.com/sophgo/tpu-mlir/releases/download/v1.4-beta.0/mobilenet_v2.pt
   $ cp -rf tpu_mlir_resource/dataset/ILSVRC2012 .
   $ mkdir workspace && cd workspace

Accuracy test of float anf int8 models
---------------------------------------

Step 1: To F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_transform \
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

Here, we use the ``use_mse`` method for calibration.

.. code-block:: shell

   $ run_calibration.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --cali_method use_mse \
       -o mobilenet_v2_cali_table

Step 3: To F32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --processor bm1684 \
       --model mobilenet_v2_1684_f32.bmodel

Step 4: To INT8 model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --processor bm1684 \
       --calibration_table mobilenet_v2_cali_table \
       --model mobilenet_v2_bm1684_int8_sym.bmodel

Step 5: Accuracy test
~~~~~~~~~~~~~~~~~~~~~~

``classify_mobilenet_v2`` is a python program, to run ``mobilenet-v2`` model.

Test the fp32 model:

.. code-block:: shell

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_f32.bmodel \
       --input ../ILSVRC2012/n02090379_7346.JPEG \
       --output mobilenet_v2_fp32_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The classification information is displayed on the output image. The right label ``sleeping bag`` ranks first.

.. _mobilenet_v2_fp32_bmodel_1.JPEG:
.. figure:: ../assets/mobilenet_v2_fp32_bmodel_1.JPEG
   :align: center

   Execution Performance of classify_mobilenet_v2 in FP32

Test the INT8 model:

.. code-block:: shell

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_int8_sym.bmodel \
       --input ../ILSVRC2012/n02090379_7346.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The classification information is displayed on the output image. The right label ``sleeping bag`` ranks second.

.. _mobilenet_v2_INT8_sym_bmodel_1.JPEG:
.. figure:: ../assets/mobilenet_v2_INT8_sym_bmodel_1.JPEG
   :align: center

   Execution Performance of classify_mobilenet_v2 in INT8

To Mix Precision Model
-----------------------

After int8 conversion, do these commands as beflow.

Step 1: Execute the ``search_qtable`` command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``search_qtable`` feature is currently integrated into the ``run_calibration`` process. Therefore, to use it, you only need to add the relevant parameters to the ``run_calibration`` command.
The parameters related to ``search_qtable`` in ``run_calibration`` are explained as follows:

.. list-table:: search_qtable parameters
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
   * - chip
     - Y
     - The platform that the model will use. Support bm1690, bm1688, bm1684x, bm1684, cv186x, cv183x, cv182x, cv181x, cv180x
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - input_num
     - N
     - The number of samples used for calibration
   * - inference_num
     - N
     - The number of samples used for inference, default 30
   * - max_float_layers
     - N
     - The number of layers set to float, default 5
   * - tune_list
     - N
     - The sample list for tune threshold
   * - tune_num
     - N
     - The number of samples for tune threshold, default 5
   * - post_process
     - N
     - The user defined prost process program path, default None
   * - expected_cos
     - N
     - Specify the minimum cos value for the expected final output layer of the network. The default is 0.99
   * - debug_cmd
     - N
     - Specifies a debug command string for development. It is empty by default
   * - global_compare_layers
     - N
     - global compare layers, for example: ``layer1,layer2`` or ``layer1:0.3,layer2:0.7``
   * - search
     - Yes
     - Specify the search type, which includes ``search_qtable``, ``search_threshold``, or false. You need to select ``search_qtable``
   * - transformer
     - N
     - Is it a transformer model? In ``search_qtable``, if it is a transformer model, a specific acceleration strategy can be assigned. The default is False
   * - quantize_method_list
     - N
     - the calibration method used for searching in ``search_qtable``, default is MSE, with selectable range being MSE, KL, MAX, Percentile9999
   * - quantize_table
     - Yes
     - qtable output path
   * - calibration_table
     - Yes
     - cali_table output path

``search_qtable`` supports user defined post process programs ``post_process_func.py``. It can be placed in the current project directory or in another location, if it is placed in another location, you need to specify the full path of the file in the ``post_process`` . The post process function must be named ``PostProcess`` , the input data is the output of the network and the output data is the post-processing result. Create the ``post_process_func.py`` file with the following sample contents:

.. code-block:: python

   def PostProcess(data):
       print("in post process")
       return data

``search_qtable`` can customize the calibration method with mixed thresholds, controlled by the parameter ``quantize_method_list``. By default, only the ``MSE`` calibration method is used for the search. If you want to use a mixed search with KLD and MSE, set the parameter ``quantize_method_list`` to ``KL,MSE``.
``search_qtable`` has an acceleration strategy for transformer models. If the model is a transformer model with an attention structure, you can set the parameter ``transformer`` to True.
Use ``search_qtable`` to search for layers with significant loss. Note that it is recommended to use bad cases for the search.

In this example, 100 images are used for quantization, 30 images are used for inference, and a mixed search using KLD and MSE calibration methods is performed. Execute the command as follows:

.. code-block:: shell

   $ run_calibration.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --expected_cos 0.99 \
       --quantize_method_list KL,MSE \
       --search search_qtable \
       --transformer False \
       --chip bm1684 \
       --post_process post_process_func.py \
       --quantize_table mobilenet_v2_qtable \
       --calibration_table mobilenet_v2_cali_table \

The final output after execution is printed as follows:

.. code-block:: shell

    the layer input3.1 is 0 sensitive layer, loss is 0.004858517758037473, type is top.Conv
    the layer input5.1 is 1 sensitive layer, loss is 0.002798812150635266, type is top.Scale
    the layer input11.1 is 2 sensitive layer, loss is 0.0015642610676610547, type is top.Conv
    the layer input13.1 is 3 sensitive layer, loss is 0.0009357141882855302, type is top.Scale
    the layer input6.1 is 4 sensitive layer, loss is 0.0009211346574943269, type is top.Conv
    the layer input2.1 is 5 sensitive layer, loss is 0.0007767164275293004, type is top.Scale
    the layer input0.1 is 6 sensitive layer, loss is 0.0006842551513905892, type is top.Conv
    the layer input128.1 is 7 sensitive layer, loss is 0.0003780628201499603, type is top.Conv
    ......
    run result:
    int8 outputs_cos:0.986809 old
    mix model outputs_cos:0.993372
    Output mix quantization table to mobilenet_v2_qtable
    total time:667.644282579422
    success search qtable

Above, int8 outputs_cos represents the cosine similarity between network outputs of int8 model and float model; mix model outputs_cos represents the cosine similarity between network outputs of mix model and float model; total time represents the search time is 667 seconds.
In addition，this program generates a quantization table ``mobilenet_v2_qtable``, the context is as below:

.. code-block:: shell

    # op_name   quantize_mode
    input3.1 F32
    input5.1 F32
    input11.1 F32
    input13.1 F32
    input6.1 F32

In the table, the first column represents the corresponding layer, and the second column represents the type. Supported types are F32/F16/BF16/INT8. ``search_qtable`` will determine
the number of mixed precision layers in the qtable based on the user-defined ``expected_cos`` parameter value. For example, if the ``expected_cos`` parameter value is equal to 0.99,
the number of mixed precision layers in the qtable corresponds to the minimum number of mixed precision layers required to achieve that level of model output comparison.
Of course, the number of mixed precision layers in the table will be limitted based on the number of model operators. If the minimum number of mixed precision layers exceeds the limitation,
only the limited quantity of mixed precision layers will be taken. Additionally, a log file ``Search_Qtable`` will be generated with the following content:

.. code-block:: shell
    :linenos:

    INFO:root:quantize_method_list =['KL', 'MSE']
    INFO:root:run float mode: mobilenet_v2.mlir
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:all_int8_cos=0.9868090914371674
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:layer name check pass !
    INFO:root:all layer number: 117
    INFO:root:all layer number no float: 116
    INFO:root:transformer model: False, all search layer number: 116
    INFO:root:Global metrics layer is : None
    INFO:root:start to handle layer: input0.1, type: top.Conv
    INFO:root:adjust layer input0.1 th, with method KL, and threshlod 9.442267236793155
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:outputs_cos_los = 0.0006842551513905892
    INFO:root:adjust layer input0.1 th, with method MSE, and threshlod 9.7417731
    INFO:root:run int8 mode: mobilenet_v2.mlir
    INFO:root:outputs_cos_los = 0.0007242344141149548
    INFO:root:layer input0.1, layer type is top.Conv, best_th = 9.442267236793155, best_method = KL, best_cos_loss = 0.0006842551513905892
    .....

The log file first provides the custom parameters, including the calibration method
used for the mixed threshold ``quantize_method_list``, the number of ops to be searched
all search layer number and whether it is a transformer model or not.
Then, it records the threshold obtained for each op under the given calibration methods
(in this case, MSE and KL) and provides the loss of similarity
(1 - cosine similarity) between the mixed-precision model using only the
corresponding threshold for that operation in int8 computation and the original
float model. It also includes the loss information of each operation output on
the screen side and the cosine similarity between the final mixed-precision
model and the original float model. Users can use the qtable output by the
program, or modify the qtable based on the loss information, and then generate
the mixed-precision model. After ``Search_Qtable`` is finished,
the optimal threshold will be updated to a new quantization table
``new_cali_table.txt`` , stored in the current project directory, which needs to be
called when generating the mixed-precision model.

Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --processor bm1684 \
       --calibration_table new_cali_table.txt \
       --quantize_table mobilenet_v2_qtable \
       --model mobilenet_v2_bm1684_int8_mix.bmodel

Step 3: Test accuracy of mix model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ classify_mobilenet_v2 \
       --model_def mobilenet_v2_bm1684_int8_mix.bmodel \
       --input ../ILSVRC2012/n02090379_7346.JPEG \
       --output mobilenet_v2_INT8_mix_bmodel_1.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The classification information is displayed on the output image. The right label ``sleeping bag`` ranks first.

.. _mobilenet_v2_INT8_mix_bmodel_1.JPEG:
.. figure:: ../assets/mobilenet_v2_INT8_mix_bmodel_1.JPEG
   :align: center

   Execution Performance of classify_mobilenet_v2 in the Mixed Precision Model

2. run_qtable
==================

This section takes ``yolov3 tiny`` as examples to introduce how to use run_qtable for mix precision.

.. This model is from <https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3>。

This section requires the tpu_mlir python package.


Install tpu_mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[all]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[all]


Prepare working directory
---------------------------

.. include:: get_resource.rst

Create a ``yolov3_tiny`` directory, and put both model files and image files into the ``yolov3_tiny`` directory.

The operation is as follows:

.. code-block:: shell
  :linenos:

   $ mkdir yolov3_tiny && cd yolov3_tiny
   $ wget https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ mkdir workspace && cd workspace

Note that if ``tiny-yolov3-11.onnx`` fails to download with wget, please download it by other means and put it into ``yolov3_tiny`` directory.


Verify onnx
-------------------

``detect_yolov3`` is a python program, to run ``yolov3_tiny`` model.

The operation is as follows:

.. code-block:: shell

   $ detect_yolov3 \
        --model ../tiny-yolov3-11.onnx \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_onnx.jpg

The print result as follows:

.. code-block:: shell

    person:60.7%
    orange:77.5%

And get result image ``yolov3_onnx.jpg``, as below ( :ref:`yolov3_onnx_result` ):

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

   $ model_transform \
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

   $ run_calibration yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table

Step 3: To model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --calibration_table yolov3_cali_table \
       --processor bm1684x \
       --model yolov3_int8.bmodel

Step 4: Run model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3 \
        --model yolov3_int8.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_int8.jpg

The print result as follows, indicates that one target is detected:

.. code-block:: shell

    orange:72.9.0%

And get image ``yolov3_int8.jpg``, as below ( :ref:`yolov3_int8_result` ):

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

Use ``run_qtable`` to gen qtable, parameters as below:

.. list-table:: run_qtable parameters
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
   * - processor
     - Y
     - The platform that the model will use. Support bm1690, bm1688, bm1684x, bm1684, cv186x, cv183x, cv182x, cv181x, cv180x.
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
     - global compare layers, for example: ``layer1,layer2`` or ``layer1:0.3,layer2:0.7``
   * - loss_table
     - N
     - Specify the name of the file that holds the loss values for all layers quantized to floating point type, default is ``full_loss_table.txt``

In this example, the default calibration of 10 images is used, you need install Graphviz first:

.. code-block:: shell

   $ sudo apt-get install graphviz


Then use following command:

.. code-block:: shell

   $ run_qtable yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --min_layer_cos 0.999 \
       --expected_cos 0.9999 \
       --processor bm1684x \
       -o yolov3_qtable

If the default 0.99 is used in ``--min_layer_cos``, the program detects that the original int8 model already meets the cos of 0.99 and simply stops searching. The final output after execution is printed as follows:

.. code-block:: shell

    int8 outputs_cos:0.999115 old
    mix model outputs_cos:0.999517
    Output mix quantization table to yolov3_qtable
    total time:44 second

Above, int8 outputs_cos represents the cos similarity between original network output of int8 model and fp32; mix model outputs_cos represents the cos similarity of network output after mixing precision is used in some layers; total time represents the search time of 44 seconds.
In addition, get quantization table ``yolov3_qtable``, context as below:

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

    # platform: bm1684x  mix_mode: F16
    ###
    No.0   : Layer: model_1/leaky_re_lu_3/LeakyRelu:0_LeakyRelu             Cos: 0.994022
    No.1   : Layer: model_1/leaky_re_lu_5/LeakyRelu:0_LeakyRelu             Cos: 0.997445
    No.2   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu             Cos: 0.997487
    No.3   : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu             Cos: 0.997978
    No.4   : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_pooling0_MaxPool      Cos: 0.998159
    No.5   : Layer: convolution_output11_Conv                               Cos: 0.998307
    No.6   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_LeakyRelu             Cos: 0.999249
    No.7   : Layer: convolution_output9_Conv                                Cos: 0.999292
    No.8   : Layer: convolution_output8_Conv                                Cos: 0.999427
    No.9   : Layer: model_1/leaky_re_lu_1/LeakyRelu:0_pooling0_MaxPool      Cos: 0.999580
    No.10  : Layer: convolution_output12_Conv                               Cos: 1.000004


This table is arranged smoothly according to the cos from small to large, indicating the cos calculated
by this Layer after the precursor layer of this layer has been changed to the corresponding floating-point mode.
If the cos is still smaller than the previous parameter min_layer_cos, this layer and its immediate successor
layer will be set to floating-point calculation。
``run_qtable`` calculates the output cos of the whole network every time the neighboring two layers are set
to floating point. If the cos is larger than the specified expected_cos, the search is withdrawn. Therefore,
if you set a larger expected_cos value, you will try to set more layers to floating point。


Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --processor bm1684x \
       --model yolov3_mix.bmodel

Step 3: run mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3 \
        --model yolov3_mix.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_mix.jpg

The print result as follows:

.. code-block:: shell

    person:63.9%
    orange:72.9%

And get image ``yolov3_mix.jpg`` , as below ( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny mix

It can be seen that targets that cannot be detected in int8 model can be detected again with the use of mixing precision.


3. run_sensitive_layer
========================

This section takes ``mobilenet-v2`` as example to introduce how to use sensitive layer search.

.. This model is from <nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt>.

This section requires the tpu_mlir python package.


Install tpu_mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[all]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[all]

Prepare working directory
---------------------------

.. include:: get_resource.rst

Create a ``mobilenet-v2`` directory, and put both model files and image files into the ``mobilenet-v2`` directory.

The operation is as follows:

.. code-block:: shell
  :linenos:

   $ mkdir mobilenet-v2 && cd mobilenet-v2
   $ wget https://github.com/sophgo/tpu-mlir/releases/download/v1.4-beta.0/mobilenet_v2.pt
   $ cp -rf tpu_mlir_resource/dataset/ILSVRC2012 .
   $ mkdir workspace && cd workspace


Accuracy test of float anf int8 models
---------------------------------------

Step 1: To F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_transform \
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

   $ run_calibration mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

Step 3: To F32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --processor bm1684 \
       --model mobilenet_v2_1684_f32.bmodel

Step 4: To INT8 model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --processor bm1684 \
       --calibration_table mobilenet_v2_cali_table \
       --model mobilenet_v2_bm1684_int8_sym.bmodel

Step 5: Accuracy test
~~~~~~~~~~~~~~~~~~~~~~

``classify_mobilenet_v2`` is a python program, to run ``mobilenet-v2`` model.

Test the fp32 model:

.. code-block:: shell

   $ classify_mobilenet_v2 \
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

   $ classify_mobilenet_v2 \
       --model_def mobilenet_v2_bm1684_int8_sym.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

The right label ``tench, Tinca tinca`` ranks first.

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02749479 assault 日file, assau
    n02536864 coho, cohoe, coho
    n02916936 bulletproof vest
    n04336792 stretcher

To Mix Precision Model
-----------------------

After int8 conversion, do these commands as beflow.

Step 1: Search sensitive layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``run_sensitive_layer`` and bad cases to search sensitive layers, parameters as below:

.. list-table:: run_sensitive_layer parameters
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
   * - processor
     - Y
     - The platform that the model will use. Support bm1690, bm1688, bm1684x, bm1684, cv186x, cv183x, cv182x, cv181x, cv180x.
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
     - global compare layers, for example: ``layer1,layer2`` or ``layer1:0.3,layer2:0.7``
   * - fp_type
     - N
     - float type of mix precision

Sensitive layer program supports user defined post process programs ``post_process_func.py``. It can be placed in the current project directory or in another location, if it is placed in another location, you need to specify the full path of the file in the ``post_process`` . The post process function must be named ``PostProcess`` , the input data is the output of the network and the output data is the post-processing result. Create the ``post_process_func.py`` file with the following sample contents:

.. code-block:: python

   def PostProcess(data):
       print("in post process")
       return data

In this example, 100 images are used for calibration and 30 images are used for inference, and the command is as follows:

The operation is as follows:

.. code-block:: shell

   $ run_sensitive_layer mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --calibration_table mobilenet_v2_cali_table \
       --processor bm1684 \
       --post_process post_process_func.py \
       -o mobilenet_v2_qtable

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

The log file records the threshold obtained for each operation under different
quantization methods (MAX/Percentile9999/KL) and provides the loss of similarity
(1 - cosine similarity) between the mixed-precision model using only the
corresponding threshold for that operation in int8 computation and the original
float model. It also includes the loss information of each operation output on
the screen side and the cosine similarity between the final mixed-precision
model and the original float model. Users can use the qtable output by the
program, or modify the qtable based on the loss information, and then generate
the mixed-precision model. After the search for sensitive layers is finished,
the optimal threshold will be updated to a new quantization table
'new_cali_table.txt', stored in the current project directory, which needs to be
called when generating the mixed-precision model. In this case, based on the
output loss information, it was observed that the loss of input3.1 is much
higher than that of other operations, which can be set to FP32 only in the
qtable.


Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --processor bm1684 \
       --calibration_table new_cali_table.txt \
       --quantize_table mobilenet_v2_qtable \
       --model mobilenet_v2_bm1684_int8_mix.bmodel

Step 3: Test accuracy of mix model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ classify_mobilenet_v2 \
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


4. fp_forward
==============================


For specific neural networks, some layers may not be suitable for quantization due to significant differences in data distribution. The "Local Non-Quantization" allows you to add certain layers before, after, or between other layers to a mixed-precision table. These layers will not be quantized when generating a mixed-precision model.

In this section, we will continue using the example of the YOLOv5s network mentioned in Chapter 3 and demonstrate how to use the Local Non-Quantization to quickly generate a mix-precision model.

The process of generating FP32 and INT8 models is the same as in Chapter 3. Here, we focus on generating mix-precision model and the accuracy testing.

For YOLO series models, the last three convolutional layers often have significantly different data distributions, and adding them manually to the mixed-precision table can improve accuracy. With the Local Non-Quantization feature, you can search for the corresponding layers from the Top MLIR file generated by model_transform and quickly add them to the mixed-precision table using the following command:

.. code-block:: shell

   $ fp_forward \
       yolov5s.mlir \
       --quantize INT8 \
       --processor bm1684x \
       --fpfwd_outputs 474_Conv,326_Conv,622_Conv\
       -o yolov5s_qtable

Opening the file "yolov5s_qtable" will reveal that the relevant layers have been added to the qtable.

Generating the Mixed-Precision Model

.. code-block:: shell

  $ model_deploy \
      --mlir yolov5s.mlir \
      --quantize INT8 \
      --calibration_table yolov5s_cali_table \
      --quantize_table yolov5s_qtable \
      --processor bm1684x \
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


.. list-table:: fp_forward parameters
   :widths: 23 8 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - mlir file
   * - processor
     - Y
     - The platform that the model will use. Support bm1690, bm1688, bm1684x, bm1684, cv186x, cv183x, cv182x, cv181x, cv180x.
   * - fpfwd_inputs
     - N
     - Specify layers (including this layer) to skip quantization before them. Multiple inputs are separated by commas.
   * - fpfwd_outputs
     - N
     - Specify layers (including this layer) to skip quantization after them. Multiple inputs are separated by commas.
   * - fpfwd_blocks
     - N
     - Specify the start and end layers between which quantization will be skipped. Start and end layers are separated by colon, and multiple blocks are separated by space.
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - o
     - Y
     - output quantization table
