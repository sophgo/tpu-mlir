User Interface
==============

This chapter introduces the user interface.

Introduction
--------------------

The basic procedure is transforming the model into a mlir file with ``model_transform.py``, and then transforming the mlir into the corresponding model with ``model_deploy.py``.
Calibration is required if you need to get the INT8 model.
The general process is shown in the figure (:ref:`ui_0`).

Other complex cases such as image input with preprocessing and multiple inputs are also supported, as shown in the figure (:ref:`ui_1`).

TFLite model conversion is also supported, with the following command:

.. code-block:: shell

    # TFLite conversion example
    $ model_transform.py \
        --model_name resnet50_tf \
        --model_def  ../resnet50_int8.tflite \
        --input_shapes [[1,3,224,224]] \
        --mean 103.939,116.779,123.68 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input ../image/dog.jpg \
        --test_result resnet50_tf_top_outputs.npz \
        --mlir resnet50_tf.mlir
   $ model_deploy.py \
       --mlir resnet50_tf.mlir \
       --quantize INT8 \
       --processor bm1684x \
       --test_input resnet50_tf_in_f32.npz \
       --test_reference resnet50_tf_top_outputs.npz \
       --tolerance 0.95,0.85 \
       --model resnet50_tf_1684x.bmodel

Supporting the conversion of Caffe models, the commands are as follows:

.. code-block:: shell

    # Caffe conversion example
    $ model_transform.py \
        --model_name resnet18_cf \
        --model_def  ../resnet18.prototxt \
        --model_data ../resnet18.caffemodel \
        --input_shapes [[1,3,224,224]] \
        --mean 104,117,123 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input ../image/dog.jpg \
        --test_result resnet50_cf_top_outputs.npz \
        --mlir resnet50_cf.mlir
    # The call of model_deploy is consistent with onnx
    # ......

.. _ui_0:
.. figure:: ../assets/ui_0.png
   :height: 9.5cm
   :align: center

   User interface 1

.. _ui_1:
.. figure:: ../assets/ui_1.png
   :height: 9.5cm
   :align: center

   User interface 2

.. _model_transform:

model_transform.py
--------------------

Used to convert various neural network models into MLIR files, the supported parameters are shown below:


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
   * - model_data
     - N
     - Specify the model weight file, required when it is caffe model (corresponding to the '.caffemodel' file)
   * - input_shapes
     - N
     - The shape of the input, such as [[1,3,640,640]] (a two-dimensional array), which can support multiple inputs
   * - input_types
     - N
     - Type of the inputs, such int32; separate by ',' for multi inputs; float32 as default
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
   * - add_postprocess
     - N
     - add postprocess op into bmodel, set the type of post handle op such as yolov3/yolov3_tiny/yolov5/ssd
   * - test_input
     - N
     - The input file for verification, which can be an image, npy or npz. No verification will be carried out if it is not specified
   * - test_result
     - N
     - Output file to save verification result
   * - excepts
     - N
     - Names of network layers that need to be excluded from verification. Separated by comma
   * - onnx_sim
     - N
     - option for onnx-sim, currently only support 'skip_fuse_bn' args
   * - mlir
     - Y
     - The output mlir file name (including path)
   * - debug
     - N
     - If open debug, immediate model file will keep; or will remove after conversion done
   * - tolerance
     - N
     - Minimum similarity tolerance to model transform

After converting to an mlir file, a ``${model_name}_in_f32.npz`` file will be generated, which is the input file for the subsequent models.


.. _run_calibration:

run_calibration.py
--------------------

Use a small number of samples for calibration to get the quantization table of the network (i.e., the threshold/min/max of each layer of op).

Supported parameters:

.. list-table:: Function of run_calibration parameters
   :widths: 20 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - Mlir file
   * - dataset
     - N
     - Directory of input samples. Images, npz or npy files are placed in this directory
   * - data_list
     - N
     - The sample list (cannot be used together with "dataset")
   * - input_num
     - N
     - The number of input for calibration. Use all samples if it is 0
   * - tune_num
     - N
     - The number of fine-tuning samples. 10 by default
   * - tune_list
     - N
     - Tune list file contain all input for tune
   * - histogram_bin_num
     - N
     - The number of histogram bins. 2048 by default
   * - o
     - Y
     - Name of output calibration table file
   * - debug_cmd
     - N
     - debug cmd

A sample calibration table is as follows:

.. code-block:: shell

    # genetated time: 2022-08-11 10:00:59.743675
    # histogram number: 2048
    # sample number: 100
    # tune number: 5
    ###
    # op_name    threshold    min    max
    images 1.0000080 0.0000000 1.0000080
    122_Conv 56.4281803 -102.5830231 97.6811752
    124_Mul 38.1586478 -0.2784646 97.6811752
    125_Conv 56.1447888 -143.7053833 122.0844193
    127_Mul 116.7435987 -0.2784646 122.0844193
    128_Conv 16.4931355 -87.9204330 7.2770605
    130_Mul 7.2720342 -0.2784646 7.2720342
    ......

It is divided into 4 columns: the first column is the name of the Tensor; the second column is the threshold (for symmetric quantization);
The third and fourth columns are min/max, used for asymmetric quantization.


.. _run_qtable:

run_qtable.py
--------------------

Use ``run_qtable.py`` to generate a mixed precision quantization table. The relevant parameters are described as follows:

Supported parameters:

.. list-table:: Function of run_qtable.py parameters
   :widths: 20 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - Mlir file
   * - dataset
     - N
     - Directory of input samples. Images, npz or npy files are placed in this directory
   * - data_list
     - N
     - The sample list (cannot be used together with "dataset")
   * - calibration_table
     - N
     - The quantization table path
   * - processor
     - Y
     - The platform that the model will use. Support bm1688/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
   * - input_num
     - N
     - The number of input for calibration. Use all samples if it is 10
   * - expected_cos
     - N
     - Expected net output cos
   * - global_compare_layers
     - N
     - Global compare layers, for example: layer1,layer2 or layer1:0.3,layer2:0.7
   * - fp_type
     - N
     - The precision type, default auto
   * - base_quantize_table
     - N
     - Base quantize table
   * - loss_table
     - N
     - The output loss table, default full_loss_table.txt
   * - o
     - N
     - Output mixed precision quantization table

A sample mixed precision quantization table is as follows:

.. code-block:: shell

    # genetated time: 2022-11-09 21:35:47.981562
    # sample number: 3
    # all int8 loss: -39.03119206428528
    # processor: bm1684x  mix_mode: F32
    ###
    # op_name   quantize_mode
    conv2_1/linear/bn F32
    conv2_2/dwise/bn  F32
    conv6_1/linear/bn F32

It is divided into 2 columns: the first column corresponds to the name of the layer, and the second column corresponds to the quantization mode.

At the same time, a loss table will be generated, the default is ``full_loss_table.txt``, the sample is as follows:

.. code-block:: shell

    # genetated time: 2022-11-09 22:30:31.912270
    # sample number: 3
    # all int8 loss: -39.03119206428528
    # processor: bm1684x  mix_mode: F32
    ###
    No.0 : Layer: conv2_1/linear/bn Loss: -36.14866065979004
    No.1 : Layer: conv2_2/dwise/bn  Loss: -37.15774385134379
    No.2 : Layer: conv6_1/linear/bn Loss: -38.44639046986898
    No.3 : Layer: conv6_2/expand/bn Loss: -39.7430411974589
    No.4 : Layer: conv1/bn          Loss: -40.067259073257446
    No.5 : Layer: conv4_4/dwise/bn  Loss: -40.183939139048256
    No.6 : Layer: conv3_1/expand/bn Loss: -40.1949667930603
    No.7 : Layer: conv6_3/expand/bn Loss: -40.61786969502767
    No.8 : Layer: conv3_1/linear/bn Loss: -40.9286363919576
    No.9 : Layer: conv6_3/linear/bn Loss: -40.97952524820963
    No.10: Layer: block_6_1         Loss: -40.987406969070435
    No.11: Layer: conv4_3/dwise/bn  Loss: -41.18325670560201
    No.12: Layer: conv6_3/dwise/bn  Loss: -41.193763415018715
    No.13: Layer: conv4_2/dwise/bn  Loss: -41.2243926525116
    ......

It represents the loss of the output obtained after the corresponding Layer is changed to floating point calculation.


.. _model_deploy:

model_deploy.py
--------------------

Convert the mlir file into the corresponding model, the parameters are as follows:


.. list-table:: Function of model_deploy parameters
   :widths: 20 12 50
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
   * - quant_input
     - N
     - Strip input type cast in bmodel, need outside type conversion
   * - quant_input_list
     - N
     - choose index to strip cast, such as 1,3 means first & third input`s cast
   * - quant_output
     - N
     - Strip output type cast in bmodel, need outside type conversion
   * - quant_output_list
     - N
     - Choose index to strip cast, such as 1,3 means first & third output`s cast
   * - processor
     - Y
     - The platform that the model will use. Support bm1688/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x.
   * - calibration_table
     - N
     - The quantization table path. Required when it is INT8 quantization
   * - ignore_f16_overflow
     - N
     - Operators with F16 overflow risk are still implemented according to F16; otherwise, F32 will be implemented by default, such as LayerNorm
   * - tolerance
     - N
     - Tolerance for the minimum similarity between MLIR quantized and MLIR fp32 inference results
   * - test_input
     - N
     - The input file for verification, which can be an image, npy or npz. No verification will be carried out if it is not specified
   * - test_reference
     - N
     - Reference data for verifying mlir tolerance (in npz format). It is the result of each operator
   * - excepts
     - N
     - Names of network layers that need to be excluded from verification. Separated by comma
   * - op_divide
     - N
     - cv183x/cv182x/cv181x/cv180x only, Try to split the larger op into multiple smaller op to achieve the purpose of ion memory saving, suitable for a few specific models
   * - model
     - Y
     - Name of output model file (including path)
   * - debug
     - N
     - to keep all intermediate files for debug
   * - core
     - N
     - When the target is selected as bm1688, it is used to select the number of tpu cores for parallel computing, and the default setting is 1 tpu core
   * - asymmetric
     - N
     - Do INT8 asymmetric quantization
   * - dynamic
     - N
     - Do compile dynamic
   * - includeWeight
     - N
     - Include weight in tosa.mlir
   * - customization_format
     - N
     - Pixel format of input frame to the model
   * - compare_all
     - N
     - Decide if compare all tensors when lowering
   * - num_device
     - N
     - The number of devices to run for distributed computation
   * - num_core
     - N
     - The number of Tensor Computing Processor cores used for parallel computation
   * - skip_verification
     - N
     - Skip checking the correctness of bmodel
   * - merge_weight
     - N
     - Merge weights into one weight binary with previous generated cvimodel
   * - model_version
     - N
     - If need old version cvimodel, set the verion, such as 1.2
   * - q_group_size
     - N
     - Group size for per-group quant, only used in W4A16 quant mode

.. _tools:

Other Tools
--------------------

model_runner.py
~~~~~~~~~~~~~~~~

Model inference. mlir/pytorch/onnx/tflie/bmodel/prototxt supported.

Example:

.. code-block:: shell

   $ model_runner.py \
      --input sample_in_f32.npz \
      --model sample.bmodel \
      --output sample_output.npz

Supported parameters:

.. list-table:: Function of model_runner parameters
   :widths: 18 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - input
     - Y
     - Input npz file
   * - model
     - Y
     - Model file (mlir/pytorch/onnx/tflie/bmodel/prototxt)
   * - dump_all_tensors
     - N
     - Export all the results, including intermediate ones, when specified


npz_tool.py
~~~~~~~~~~~~~~~~

npz will be widely used in TPU-MLIR project for saving input and output results, etc. npz_tool.py is used to process npz files.

Example:

.. code-block:: shell

   # Check the output data in sample_out.npz
   $ npz_tool.py dump sample_out.npz output

Supported functions:

.. list-table:: npz_tool functions
   :widths: 18 60
   :header-rows: 1

   * - Function
     - Description
   * - dump
     - Get all tensor information of npz
   * - compare
     - Compare difference of two npz files
   * - to_dat
     - Export npz as dat file, contiguous binary storage


visual.py
~~~~~~~~~~~~~~~~

visual.py is an visualized network/tensor compare application with interface in web browser, if accuracy of quantized network is not
as good as expected, this tool can be used to investigate the accuracy in every layer.

Example:

.. code-block:: shell

   # use TCP port 9999 in this example
   $ visual.py \
     --f32_mlir netname.mlir \
     --quant_mlir netname_int8_sym_tpu.mlir \
     --input top_input_f32.npz --port 9999

Supported functions:

.. list-table:: visual functions
   :widths: 18 60
   :header-rows: 1

   * - Function
     - Description
   * - f32_mlir
     - fp32 mlir file
   * - quant_mlir
     - quantized mlir file
   * - input
     - test input data for networks, can be in jpeg or npz format.
   * - port
     - TCP port used for UI, default port is 10000，the port should be mapped when starting docker
   * - host
     - Host ip, default:0.0.0.0
   * - manual_run
     - if net will be automaticall inferenced when UI is opened, default is false for auto inference

Notice: ``--debug`` flag should be opened during model_deploy.py to save intermediate files for visual.py. More details refer to (:ref:`visual-usage`)

gen_rand_input.py
~~~~~~~~~~~~~~~~~~~~

During model transform, if you do not want to prepare additional test data (test_input), you can use this tool to generate random input data to facilitate model verification.

The basic procedure is transforming the model into a mlir file with ``model_transform.py``. This step does not perform model verification. And then use ``gen_rand_input.py``
to read the mlir file generated in the previous step and generate random test data for model verification. Finally, use ``model_transform.py`` again to do the complete model transformation and verification.

Example:

.. code-block:: shell

    # To MLIR
    $ model_transform.py \
        --model_name yolov5s  \
        --model_def ../regression/model/yolov5s.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio     --pixel_format rgb \
        --output_names 350,498,646 \
        --mlir yolov5s.mlir

    # Generate dummy input. Here is a pseudo test picture.
    $ python gen_rand_input.py
        --mlir yolov5s.mlir \
        --img --output yolov5s_fake_img.png

    # Verification
    $ model_transform.py \
        --model_name yolov5s  \
        --model_def ../regression/model/yolov5s.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input yolov5s_fake_img.png    \
        --test_result yolov5s_top_outputs.npz \
        --keep_aspect_ratio     --pixel_format rgb \
        --output_names 350,498,646 \
        --mlir yolov5s.mlir

For more detailed usage, please refer to the following:

.. code-block:: shell

    # Value ranges can be specified for multiple inputs.
    $ python gen_rand_input.py \
      --mlir ernie.mlir \
      --ranges [[0,300],[0,0]] \
      --output ern.npz

    # Type can be specified for the input.
    $ python gen_rand_input.py \
      --mlir resnet.mlir \
      --ranges [[0,300]] \
      --input_types si32 \
      --output resnet.npz

    # Generate random image
    $ python gen_rand_input.py
        --mlir yolov5s.mlir \
        --img --output yolov5s_fake_img.png

Supported functions:

.. list-table:: gen_rand_input functions
   :widths: 18 10 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - mlir
     - Y
     - The input mlir file name (including path)
   * - img
     - N
     - Used for CV tasks to generate random images, otherwise generate npz
       files. The default image value range is [0,255], the data type is
       'uint8', and cannot be changed.
   * - ranges
     - N
     - Set the value ranges of the model inputs, expressed in list form, such as
       [[0,300],[0,0]]. If you want to generate a picture, you do not need to
       specify the value range, the default is [0,255]. In other cases, value ranges need to be specified.
   * - input_types
     - N
     - Set the model input types, such as 'si32,f32'. 'si32' and 'f32' types are
       supported. False by default, and it will be read from mlir. If you
       generate an image, you do not need to specify the data type, the default
       is 'uint8'.
   * - output
     - Y
     - The names of the output.

Notice: CV-related models usually perform a series of preprocessing on the input image. To ensure that the model is verificated correctly, you need to use '--img' to generate a random image as input.
Random npz files cannot be used as input.
It is worth noting that random input may cause model correctness verification to fail, especially NLP-related models, so it is recommended to give priority to using real test data.
