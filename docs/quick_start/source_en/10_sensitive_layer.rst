.. _sensitive layer:

Sensitive Layer Search
==================

This chapter takes ``mobilenet-v2`` as examples to introduce how to use sensitive layer search。
This model is from <nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt>。

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
   $ cp -rf $TPUC_ROOT/regression/dataset/ILSVCR2012 .
   $ mkdir workspace && cd workspace

``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.
Note that mobilenet-v2.pt needs to be downloaded from nnmodels and then be placed in the mobilenet-v2 directory.

Accuracy test of float anf int8 models
---------------------------------------

Step 1: To F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_transform.py \
       --model_name mobilenet \
       --model_def ../mobilenet_v2.pt \
       --input_shapes [[1,3,224,224]] \
       --resize_dims 256,256 \
       --mean 123.675,116.28,103.53 \
       --scale 0.0171,0.0175,0.0174 \
       --pixel_format rgb \
       --mlir mobilenet.mlir

Step 2: Gen calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration.py mobilenet.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_cali_table

Step 3: To F32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet.mlir \
       --quantize F32 \
       --chip bm1684 \
       --test_input mobilenet_in_f32.npz \
       --test_reference mobilenet_pt_top.npz \
       --model mobilenet_1684_f32.bmodel

Step 4: To INT8 model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table mobilenet_cali_table \
       --model mobilenet_bm1684_int8_sym.bmodel

Step 5: Accuracy test
~~~~~~~~~~~~~~~~~~~~~~

The topk program can be used to test the accuracy of mobilenet-v2. It is available in model-zoo, and can be called by adding harness to mlir.config.yaml.

.. code-block:: shell

   $ dataset:
        image_path: $(imagenet2012_val_set)
        image_label: $(imagenet2012_caffe_val_ground_truth)
        mean: [123.675, 116.28, 103.53]
        scale: [0.0171, 0.0175, 0.0174]
        resize_dims: 256
        size: 224
        trans: true
        bgr2rgb: true

      harness:
        type: topk
        args:
          - name: FP32
            bmodel: $(workdir)/$(name)_bm1684_f32.bmodel
          - name: INT8
            bmodel: $(workdir)/$(name)_bm1684_int8_sym.bmodel

Switch to the model-zoo directory and test accuracy using tpu_perf.precision_benchmark.

.. code-block:: shell

   $ python3 -m tpu_perf.precision_benchmark mobilenet_v2_path --mlir --target BM1684 --devices 0

The accuracy test results are stored in output/topk.csv.

.. code-block:: shell

    name,top1,top5
    mobilenet-v2-FP32,70.72%,89.81%
    mobilenet-v2-INT8,67.53%,87.84%

It can be seen that the int8 symmetric quantization model performs poorly compared to the float model. Top1 accuracy lost 3.2% and top5 accuracy lost 2%.

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
     - The platform that the model will use. Support bm1684x/bm1684/cv183x/cv182x/cv181x/cv180x.
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

In this example, the default 100 images are used for calibration and 30 images are used for inference, and the command is as follows (for the chip of CV18xx series, set the chip to the corresponding chip name) :

The operation is as follows:

.. code-block:: shell

   $ run_sensitive_layer.py mobilenet.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --calibration_table mobilenet_cali_table \
       --chip bm1684 \
       -o mobilenet_qtable

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
    Output mix quantization table to mobilenet_qtable
    total time:402.15848112106323
    success sensitive layer search

Above, int8 outputs_cos represents the cos similarity between original network output of int8 model and fp32; mix model outputs_cos represents the cos similarity of network output after mixing precision is used in some layers; total time represents the search time of 402 seconds.
In addition，get quantization table ``mobilenet_qtable``, context as below:

.. code-block:: shell

    # op_name   quantize_mode
    input3.1 F32
    input11.1 F32
    input128.1 F32
    input130.1 F32
    input127.1 F32

In the table, first col is layer name, second is quantization type.
Also a log file named``SensitiveLayerSearch`` is generated, context as blow:

.. code-block:: shell
    :linenos:

    INFO:root:start to handle layer: input3.1, type: top.Conv
    INFO:root:adjust layer input3.1 th, with method MAX, and threshlod 5.5119305
    INFO:root:run int8 mode: mobilenet.mlir
    INFO:root:outputs_cos_los = 0.014830573787862011
    INFO:root:adjust layer input3.1 th, with method Percentile9999, and threshlod 4.1202815
    INFO:root:run int8 mode: mobilenet.mlir
    INFO:root:outputs_cos_los = 0.011843443367980822
    INFO:root:adjust layer input3.1 th, with method KL, and threshlod 2.6186381997094728
    INFO:root:run int8 mode: mobilenet.mlir
    INFO:root:outputs_cos_los = 0.008808857469573828
    INFO:root:layer input3.1, layer type is top.Conv, best_th = 2.6186381997094728, best_method = KL, best_cos_loss = 0.008808857469573828

This log file records the cosine losses between the outputs of mix model and float model when setting each op to int8 with different quantize methods(MAX/Percentile9999/KL).
It also contaions the loss information printed in the screen and the cosine similarity of mix model and float model.
The qtable generated by this program can be modified according to the loss information.
The best thresholds of each op are recorded in a new cali table named new_cali_table. This table is restored in current workspace and need to be used when generating mix model.
In this example, the loss of input3.1 is larger than other ops, thus you can only set input3.1 as float in qtable.

Step 2: Gen mix precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table new_cali_table \
       --quantize_table mobilenet_qtable \
       --model mobilenet_bm1684_int8_mix.bmodel

Step 3: test accuracy of mix model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ harness:
        type: topk
        args:
          - name: INT8
            bmodel: $(workdir)/$(name)_bm1684_int8_mix.bmodel

The print result as follows:

.. code-block:: shell

    name,top1,top5
    mobilenet-v2-INT8,69.07%,88.73%

It can be seen that the top1 accuracy is improved by 1.5%.
