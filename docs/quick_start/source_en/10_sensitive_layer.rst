.. _sensitive layer:

Sensitive Layer Search
==================

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
