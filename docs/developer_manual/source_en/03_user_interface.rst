User Interface
==============

This chapter introduces the user interface, including the basic process of converting models and the usage methods of various tools.

Model Conversion Process
--------------------------

The basic procedure is transforming the model into a mlir file with ``model_transform.py``, and then transforming the mlir into the corresponding model with ``model_deploy.py``. Take the ``somenet.onnx`` model as an example, the steps are as follows:

.. code-block:: shell

    # To MLIR
    $ model_transform.py \
        --model_name somenet \
        --model_def  somenet.onnx \
        --test_input somenet_in.npz \
        --test_result somenet_top_outputs.npz \
        --mlir somenet.mlir

    # To Float Model
    $ model_deploy.py \
       --mlir somenet.mlir \
       --quantize F32 \ # F16/BF16
       --processor BM1684X \
       --test_input somenet_in_f32.npz \
       --test_reference somenet_top_outputs.npz \
       --model somenet_f32.bmodel

Support for Image Input
~~~~~~~~~~~~~~~~~~~~~~~~

When using images as input, preprocessing information needs to be specified, as follows:

.. code-block:: shell

    $ model_transform.py \
        --model_name img_input_net \
        --model_def img_input_net.onnx \
        --input_shapes [[1,3,224,224]] \
        --mean 103.939,116.779,123.68 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input cat.jpg \
        --test_result img_input_net_top_outputs.npz \
        --mlir img_input_net.mlir

Support for Multiple Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the model has multiple inputs, you can pass in a single npz file or sequentially pass in multiple npy files separated by commas, as follows:

.. code-block:: shell

    $ model_transform.py \
        --model_name multi_input_net \
        --model_def  multi_input_net.onnx \
        --test_input multi_input_net_in.npz \ # a.npy,b.npy,c.npy
        --test_result multi_input_net_top_outputs.npz \
        --mlir multi_input_net.mlir

Support for INT8 Symmetric and Asymmetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration is required if you need to get the INT8 model.

.. code-block:: shell

  $ run_calibration.py somenet.mlir \
      --dataset dataset \
      --input_num 100 \
      -o somenet_cali_table

Generating Model with Calibration Table Input.

.. code-block:: shell

    $ model_deploy.py \
       --mlir somenet.mlir \
       --quantize INT8 \
       --calibration_table somenet_cali_table \
       --processor BM1684X \
       --test_input somenet_in_f32.npz \
       --test_reference somenet_top_outputs.npz \
       --tolerance 0.9,0.7 \
       --model somenet_int8.bmodel

Support for Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the precision of the INT8 model does not meet business requirements, you can try using mixed precision. First, generate the quantization table, as follows:

.. code-block:: shell

   $ run_calibration.py somenet.mlir \
       --dataset dataset \
       --input_num 100 \
       --inference_num 30 \
       --expected_cos 0.99 \
       --calibration_table somenet_cali_table \
       --processor BM1684X \
       --search search_qtable \
       --quantize_method_list KL,MSE\
       --quantize_table somenet_qtable

Then pass the quantization table to generate the model

.. code-block:: shell

    $ model_deploy.py \
       --mlir somenet.mlir \
       --quantize INT8 \
       --calibration_table somenet_cali_table \
       --quantize_table somenet_qtable \
       --processor BM1684X \
       --model somenet_mix.bmodel

Support for Quantized TFLite Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       --processor BM1684X \
       --test_input resnet50_tf_in_f32.npz \
       --test_reference resnet50_tf_top_outputs.npz \
       --tolerance 0.95,0.85 \
       --model resnet50_tf_1684x.bmodel

Support for Caffe Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Support LLM models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ llm_convert.py \
        -m /workspace/Qwen2.5-VL-3B-Instruct-AWQ \
        -s 2048 \
        -q w4bf16 \
        -c bm1684x \
        --max_pixels 672,896 \
        -o qwen2.5vl_3b


Introduction to Tool Parameters
---------------------------------

model_transform.py
~~~~~~~~~~~~~~~~~~~~~~~~

Used to convert various neural network models into MLIR files (with ``.mlir`` suffix) and corresponding weight files (``${model_name}_top_${quantize}_all_weight.npz``). The supported parameters are as follows:


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
   * - mlir
     - Y
     - Specify the output mlir file name and path, with the suffix ``.mlir``
   * - input_shapes
     - N
     - The shape of the input, such as ``[[1,3,640,640]]`` (a two-dimensional array), which can support multiple inputs
   * - model_extern
     - N
     - Extra multi model definition files (currently mainly used for MaskRCNN). None by default. separate by ','
   * - model_data
     - N
     - Specify the model weight file, required when it is caffe model (corresponding to the '.caffemodel' file)
   * - input_types
     - N
     - When the model is a ``.pt`` file, specify the input type, such as int32; separate multiple inputs with ``,``. If not specified, it will be treated as float32 by default.
   * - keep_aspect_ratio
     - N
     - When the size of test_input is different from input_shapes, whether to keep the aspect ratio when resizing, the default is false; when set, the insufficient part will be padded with 0
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
     - The names of the output. Use the output of the model if not specified, otherwise output in the order of the specified names
   * - add_postprocess
     - N
     - add postprocess op into bmodel, set the type of post handle op such as yolov3/yolov3_tiny/yolov5/yolov8/yolov11/ssd/yolov8_seg
   * - test_input
     - N
     - The input file for verification, which can be an jpg, npy or npz file. No verification will be carried out if it is not specified
   * - test_result
     - N
     - Output file to save verification result with suffix .npz
   * - excepts
     - N
     - Names of network layers that need to be excluded from verification. Separated by comma
   * - onnx_sim
     - N
     - option for onnx-sim, currently only support 'skip_fuse_bn' args
   * - debug
     - N
     - If open debug, immediate model file will keep; or will remove after conversion done
   * - tolerance
     - N
     - Minimum Cosine and Euclidean similarity tolerance to model transform. 0.99,0.99 by default.
   * - cache_skip
     - N
     - skip checking the correctness when generate same mlir and bmodel
   * - dynamic_shape_input_names
     - N
     - Name list of inputs with dynamic shape, like:input1,input2. If set, '--dynamic' is required during model_deploy.
   * - shape_influencing_input_names
     - N
     - Name list of inputs which influencing other tensors\' shape during inference, like:input1,input2. If set, test_input is required and '--dynamic' is required during model_deploy.
   * - dynamic
     - N
     - Only valid for onnx model. If set, will automatically set inputs with dyanmic axis as dynamic_shape_input_names and set 1-d inputs as shape_influencing_input_names and '--dynamic' is required during model_deploy.
   * - resize_dims
     - N
     - The original image size 'h,w', default is same as net input dims
   * - pad_value
     - N
     - pad value when resize
   * - pad_type
     - N
     - type of pad when resize, such as normal/center
   * - preprocess_list
     - N
     - choose which input need preprocess, like:'1,3' means input 1&3 need preprocess, default all inputs
   * - path_yaml
     - N
     - the path for one single yaml file (currently mainly used for MaskRCNN)
   * - enable_maskrcnn
     - N
     - if enable MaskRCNN transformation
   * - yuv_type
     - N
     - Specify its type when using the '.yuv' file as input

After converting to an mlir file, a ``${model_name}_in_f32.npz`` file will be generated, which is the input file for the subsequent models.

run_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Use a small number of samples for calibration to get the quantization table of the network (i.e., the threshold/min/max of each layer of op).

Supported parameters:

.. list-table:: Function of run_calibration parameters
   :widths: 25 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - Mlir file
   * - sq
     - N
     - Open SmoothQuant
   * - we
     - N
     - Open weight_equalization
   * - bc
     - N
     - Open bias_correction
   * - dataset
     - N
     - Directory of input samples. Images, npz or npy files are placed in this directory
   * - data_list
     - N
     - The sample list (cannot be used together with "dataset")
   * - input_num
     - N
     - The number of input for calibration. Use all samples if it is 0
   * - inference_num
     - N
     - The number of images required for the inference process of search_qtable and search_threshold
   * - bc_inference_num
     - N
     - The number of images required for the inference process of bias_correction
   * - tune_num
     - N
     - The number of fine-tuning samples. 10 by default
   * - tune_list
     - N
     - Tune list file contain all input for tune
   * - histogram_bin_num
     - N
     - The number of histogram bins. 2048 by default
   * - expected_cos
     - N
     - The expected similarity between the mixed-precision model output and the floating-point model output in search_qtable, with a value range of [0,1]
   * - min_layer_cos
     - N
     - The minimum similarity between the quantized output and the floating-point output of a layer in bias_correction. Compensation is required for the layer when the similarity is below this threshold, with a value range of [0,1]
   * - max_float_layers
     - N
     - The number of floating-point layers in search_qtable
   * - processor
     - N
     - Processor type
   * - cali_method
     - N
     - Choose quantization threshold calculation method
   * - fp_type
     - N
     - The data type of floating-point layers in search_qtable
   * - post_process
     - N
     - The path for post-processing
   * - global_compare_layers
     - N
     - Specifies the global comparison layers, for example, layer1,layer2 or layer1:0.3,layer2:0.7
   * - search
     - N
     - Specifies the type of search, including search_qtable, search_threshold, false. The default is false, which means search is not enabled
   * - transformer
     - N
     - Whether it is a transformer model, if it is, search_qtable can allocate specific acceleration strategies
   * - quantize_method_list
     - N
     - The threshold methods used for searching in search_qtable
   * - benchmark_method
     - N
     - Specifies the method for calculating similarity in search_threshold
   * - kurtosis_analysis
     - N
     - Specify the generation of the kurtosis of the activation values for each layer
   * - part_quantize
     - N
     - Specify partial quantization of the model. The calibration table (cali_table) will be automatically generated alongside the quantization table (qtable). Available modes include N_mode, H_mode, or custom_mode, with H_mode generally delivering higher accuracy
   * - custom_operator
     - N
     - Specify the operators to be quantized, which should be used in conjunction with the aforementioned custom_mode
   * - part_asymmetric
     - N
     - When symmetric quantization is enabled, if specific subnets in the model match a defined pattern, the corresponding operators will automatically switch to asymmetric quantization
   * - mix_mode
     - N
     - Specify the mixed-precision types for the search_qtable. Currently supported options are 8_16 and 4_8
   * - cluster
     - N
     - pecify that a clustering algorithm is used to detect sensitive layers during the search_qtable process
   * - quantize_table
     - N
     - The mixed-precision quantization table output by search_qtable
   * - o
     - Y
     - Name of output calibration table file
   * - debug_cmd
     - N
     - debug cmd
   * - debug_log
     - N
     - Log output level

A sample calibration table is as follows:

.. code-block:: shell

    # generated time: 2022-08-11 10:00:59.743675
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

.. _model_deploy:

model_deploy.py
~~~~~~~~~~~~~~~~~

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
   * - processor
     - Y
     - The platform that the model will use. Support BM1684, BM1684X, BM1688, BM1690, CV186X, CV183X, CV182X, CV181X, CV180X
   * - quantize
     - Y
     - Quantization type (e.g., F32/F16/BF16/INT8), the quantization types supported by different processors are shown in the table below.
   * - quant_input
     - N
     - Strip input type cast in bmodel, need outside type conversion
   * - quant_output
     - N
     - Strip output type cast in bmodel, need outside type conversion
   * - quant_input_list
     - N
     - choose index to strip cast, such as 1,3 means first & third input`s cast
   * - quant_output_list
     - N
     - Choose index to strip cast, such as 1,3 means first & third output`s cast
   * - quantize_table
     - N
     - Specify the path to the mixed precision quantization table. If not specified, quantization is performed according to the quantize type; otherwise, quantization is prioritized according to the quantization table
   * - fuse_preprocess
     - N
     - Specify whether to fuse preprocessing into the model. If this parameter is specified, the model input will be of type uint8, and the resized original image can be directly input
   * - calibration_table
     - N
     - The quantization table path. Required when it is INT8/F8E4M3 quantization
   * - high_precision
     - N
     - Some ops will force to be float32
   * - tolerance
     - N
     - Tolerance for the minimum Cosine and Euclidean similarity between MLIR quantized and MLIR fp32 inference results. 0.8,0.5 by default.
   * - test_input
     - N
     - The input file for verification, which can be an jpg, npy or npz. No verification will be carried out if it is not specified
   * - test_reference
     - N
     - Reference data for verifying mlir tolerance (in npz format). It is the result of each operator
   * - excepts
     - N
     - Names of network layers that need to be excluded from verification. Separated by comma
   * - op_divide
     - N
     - CV183x/CV182x/CV181x/CV180x only, Try to split the larger op into multiple smaller op to achieve the purpose of ion memory saving, suitable for a few specific models
   * - model
     - Y
     - Name of output model file (including path)
   * - debug
     - N
     - to keep all intermediate files for debug
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
     - Group size for per-group quant, only used in W4A16/W8A16 quant mode
   * - q_symmetric
     - N
     - Do symmetric W4A16/W8A16 quant
   * - compress_mode
     - N
     - Specify the compression mode of the model: "none", "weight", "activation", "all". Supported on BM1688. Default is "none", no compression
   * - opt_post_processor
     - N
     - Specify whether to further optimize the results of LayerGroup. Supported on MARS3. Default is "none", no opt
   * - lgcache
     - N
     - Specifies whether to cache the partitioning results of LayerGroup: "true", "false". The default is "true", which saves the partitioning results of each subnet to the working directory as "cut_result_{subnet_name}.mlircache".
   * - cache_skip
     - N
     - skip checking the correctness when generate same mlir and bmodel
   * - aligned_input
     - N
     - if the input frame is width/channel aligned. VPSS input alignment for CV series processors only
   * - group_by_cores
     - N
     - whether layer groups force group by cores, auto/true/false, default is auto
   * - opt
     - N
     - Optimization type of LayerGroup, 1/2/3, default is 2. 1: Simple LayerGroup mode, all operators will be grouped as much as possible, and the compilation speed is faster; 2: Dynamic compilation calculates the global cycle optimal Group grouping, suitable for inference graphs; 3: Linear programming LayerGroup mode, suitable for training graphs.
   * - addr_mode
     - N
     - set address assign mode ['auto', 'basic', 'io_alone', 'io_tag', 'io_tag_fuse', 'io_reloc'], if not set, auto as default
   * - disable_layer_group
     - N
     - Whether to disable LayerGroup pass
   * - disable_gdma_check
     - N
     - Whether to disable gdma address check
   * - do_winograd
     - N
     - if do WinoGrad convolution, only for BM1684
   * - time_fixed_subnet
     - N
     - Split the model by fixed-duration intervals, supporting ['normal', 'limit', 'custom'] modes. Currently compatible with BM1684X and BM1688 processors. Enabling this feature may impact model performance
   * - subnet_params
     - N
     - When time_fixed_subnet is set to "custom", it allows manual configuration of the subnet frequency (MHz) and execution time (ms)
   * - matmul_perchannel
     - N
     - if matmul is quantized in per-channel mode, for BM1684X and BM1688, the performance may be decreased if enable
   * - enable_maskrcnn
     - N
     - if enable comparison for MaskRCNN.

The following table shows the correspondence between different processors and the supported quantize types:

.. list-table:: Quantization types supported by different processors
   :widths: 18 30
   :header-rows: 1

   * - Processor
     - Supported quantize
   * - BM1684
     - F32, INT8
   * - BM1684X
     - F32, F16, BF16, INT8, W4F16, W8F16, W4BF16, W8BF16
   * - BM1688
     - F32, F16, BF16, INT8, INT4, W4F16, W8F16, W4BF16, W8BF16
   * - BM1690
     - F32, F16, BF16, INT8, F8E4M3, F8E5M2, W4F16, W8F16, W4BF16, W8BF16
   * - CV186X
     - F32, F16, BF16, INT8, INT4
   * - CV183X, CV182X, CV181X, CV180X
     - BF16, INT8

The ``Weight-only`` quantization mode of ``W4A16`` and ``W8A16`` only applies to the MatMul operation, and other operators will still perform ``F16`` or ``BF16`` quantization.


llm_convert.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert the LLM model into bmodel, the parameters are as follows:

.. list-table:: llm_convert Parameter Functions
   :widths: 18 10 50
   :header-rows: 1

   * - Parameter
     - Required?
     - Description
   * - model_path
     - Yes
     - Specifies the path to the model
   * - seq_length
     - Yes
     - Specifies the maximum sequence length
   * - quantize
     - Yes
     - Specifies the quantization type, e.g., w4bf16/w4f16/bf16/f16
   * - q_group_size
     - No
     - Specifies the group size for quantization
   * - chip
     - Yes
     - Specifies the processor type; supports bm1684x/bm1688/cv186ah
   * - max_pixels
     - No
     - Multimodal parameter; specifies the maximum dimensions, either “672,896” or “602112”
   * - num_device
     - No
     - Specifies the number of devices for bmodel deployment
   * - num_core
     - No
     - Specifies the number of cores for bmodel deployment; 0 means use the maximum available cores
   * - max_input_length
     - No
     - Specifies the maximum input length, default is seq_length
   * - embedding_disk
     - No
     - If set, exports the word embeddings to a binary file and runs inference on the CPU
   * - out_dir
     - Yes
     - Specifies the output directory for the bmodel file


model_runner.py
~~~~~~~~~~~~~~~~

Model inference. mlir/pytorch/onnx/tflite/bmodel/prototxt supported.

Example:

.. code-block:: shell

   $ model_runner.py \
      --input sample_in_f32.npz \
      --model sample.bmodel \
      --output sample_output.npz \
      --out_fixed

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
     - Model file (mlir/pytorch/onnx/tflite/bmodel/prototxt)
   * - dump_all_tensors
     - N
     - Export all the results, including intermediate ones, when specified
   * - out_fixed
     - N
     - Remain integer output when the dtype of output is int8, instead of transforming to float32 automaticall


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


mlir2graph.py
~~~~~~~~~~~~~~~~

Visualizes MLIR files based on dot, supporting MLIR files from all stages. After execution, corresponding .dot and .svg files will be generated in the MLIR directory. The .dot file can be rendered into other formats using the dot command. .svg is the default output rendering format and can be directly opened in a browser.

Execution command example:

.. code-block:: shell

   $ mlir2graph.py \
     --mlir netname.mlir

For large MLIR files, the original rendering algorithm for dot files may take a long time. You can add the --is_big parameter to reduce the iteration time of the algorithm and generate the graph faster:

.. code-block:: shell

   $ mlir2graph.py \
     --mlir netname.mlir --is_big

Supported functions:

.. list-table:: mlir2graph functions
   :widths: 18 60
   :header-rows: 1

   * - Function
     - Description
   * - mlir
     - Any MLIR file
   * - is_big
     - Indicates whether the MLIR file is relatively large; there is no specific criterion, usually judged based on rendering time
   * - failed_keys
     - List of failed node names for comparison, separated by ",", nodes corresponding to these keys will be rendered in red after rendering
   * - bmodel_checker_data
     - Path to the failed.npz file generated by bmodel_checker.py; when this path is specified, it will automatically parse the error nodes and render them in red
   * - output
     - Path to the output file, default is the path of --mlir with the corresponding format suffix, such as netname.mlir.dot/netname.mlir.svg


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
    $ gen_rand_input.py \
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
    $ gen_rand_input.py \
      --mlir ernie.mlir \
      --ranges [[0,300],[0,0]] \
      --output ern.npz

    # Type can be specified for the input.
    $ gen_rand_input.py \
      --mlir resnet.mlir \
      --ranges [[0,300]] \
      --input_types si32 \
      --output resnet.npz

    # Generate random image
    $ gen_rand_input.py \
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
     - Specify the output mlir file name and path, with the suffix ``.mlir``
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


model_tool
~~~~~~~~~~~~~~~~~~~~


This tool is used to process the final model file "bmodel" or "cvimodel". All arguments and corresponding function descriptions can be viewed by executing the following command:

.. code-block:: shell

   $ model_too

The following uses "xxx.bmodel" as an example to introduce the main functions of this tool.

1) show basic info of bmodel

Example:

.. code-block:: shell

   $ model_tool --info xxx.bmodel

Displays the basic information of the model, including the compiled version of the model, the compilation date, the name of the network in the model, input and output parameters, etc.
The display effect is as follows:

.. code-block:: text

  bmodel version: B.2.2+v1.7.beta.134-ge26380a85-20240430
  processor: BM1684X
  create time: Tue Apr 30 18:04:06 2024

  kernel_module name: libbm1684x_kernel_module.so
  kernel_module size: 3136888
  ==========================================
  net 0: [block_0]  static
  ------------
  stage 0:
  input: input_states, [1, 512, 2048], bfloat16, scale: 1, zero_point: 0
  input: position_ids, [1, 512], int32, scale: 1, zero_point: 0
  input: attention_mask, [1, 1, 512, 512], bfloat16, scale: 1, zero_point: 0
  output: /layer/Add_1_output_0_Add, [1, 512, 2048], bfloat16, scale: 1, zero_point: 0
  output: /layer/self_attn/Add_1_output_0_Add, [1, 1, 512, 256], bfloat16, scale: 1, zero_point: 0
  output: /layer/self_attn/Transpose_2_output_0_Transpose, [1, 1, 512, 256], bfloat16, scale: 1, zero_point: 0
  ==========================================
  net 1: [block_1]  static
  ------------
  stage 0:
  input: input_states, [1, 512, 2048], bfloat16, scale: 1, zero_point: 0
  input: position_ids, [1, 512], int32, scale: 1, zero_point: 0
  input: attention_mask, [1, 1, 512, 512], bfloat16, scale: 1, zero_point: 0
  output: /layer/Add_1_output_0_Add, [1, 512, 2048], bfloat16, scale: 1, zero_point: 0
  output: /layer/self_attn/Add_1_output_0_Add, [1, 1, 512, 256], bfloat16, scale: 1, zero_point: 0
  output: /layer/self_attn/Transpose_2_output_0_Transpose, [1, 1, 512, 256], bfloat16, scale: 1, zero_point: 0

  device mem size: 181645312 (weight: 121487360, instruct: 385024, runtime: 59772928)
  host mem size: 0 (weight: 0, runtime: 0)

2) combine multi bmodels

Example:

.. code-block:: shell

   $ model_tool --combine a.bmodel b.bmodel c.bmodel -o abc.bmodel


Merge multiple bmodels into one bmodel. If there is a network with the same name in the bmodel, it will be divided into different stages.

3) extract model to multi bmodels

Example:

.. code-block:: shell

   $ model_tool --extract abc.bmodel


Decomposing a bmodel into multiple bmodels is the opposite operation to the combine command. It will be divided into different stages.

4) show weight info

Example:

.. code-block:: shell

   $ model_tool --weight xxx.bmodel

Display the weight range information of each operator in different networks. The display effect is as follows:

.. code-block:: text

  net 0 : "block_0", stage:0
  -------------------------------
  tpu.Gather : [0x0, 0x40000)
  tpu.Gather : [0x40000, 0x80000)
  tpu.RMSNorm : [0x80000, 0x81000)
  tpu.A16MatMul : [0x81000, 0x2b1000)
  tpu.A16MatMul : [0x2b1000, 0x2f7000)
  tpu.A16MatMul : [0x2f7000, 0x33d000)
  tpu.A16MatMul : [0x33d000, 0x56d000)
  tpu.RMSNorm : [0x56d000, 0x56e000)
  tpu.A16MatMul : [0x56e000, 0x16ee000)
  tpu.A16MatMul : [0x16ee000, 0x286e000)
  tpu.A16MatMul : [0x286e000, 0x39ee000)
  ==========================================
  net 1 : "block_1", stage:0
  -------------------------------
  tpu.Gather : [0x0, 0x40000)
  tpu.Gather : [0x40000, 0x80000)
  tpu.RMSNorm : [0x80000, 0x81000)
  tpu.A16MatMul : [0x81000, 0x2b1000)
  tpu.A16MatMul : [0x2b1000, 0x2f7000)
  tpu.A16MatMul : [0x2f7000, 0x33d000)
  tpu.A16MatMul : [0x33d000, 0x56d000)
  tpu.RMSNorm : [0x56d000, 0x56e000)
  tpu.A16MatMul : [0x56e000, 0x16ee000)
  tpu.A16MatMul : [0x16ee000, 0x286e000)
  tpu.A16MatMul : [0x286e000, 0x39ee000)
  ==========================================

5) update weight from one bmodel to dst bmodel

Example:

.. code-block:: shell

   # Update the weight of the network named src_net in src.bmodel at the 0x2000 position to the 0x1000 position of dst_net in dst.bmodel
   $ model_tool --update_weight dst.bmodel dst_net 0x1000 src.bmodel src_net 0x2000


The model weights can be updated. For example, if the weight of an operator of a certain model needs to be updated, compile the operator separately into bmodel, and then update its weight to the original model.


6) model encryption and decryption

Example:

.. code-block:: shell

   # -model specifies the combined or regular bmodel, -net specifies the network to be encrypted, -lib specifies the library implementing the encryption algorithm, -o specifies the name of the encrypted model output
   $ model_tool --encrypt -model combine.bmodel -net block_0 -lib libcipher.so -o encrypted.bmodel
   $ model_tool --decrypt -model encrypted.bmodel -lib libcipher.so -o decrypted.bmodel

This can achieve the encryption of model weights, flatbuffer structured data, and headers.
The encryption and decryption interfaces must be implemented in C style, not using C++. The interface specifications are as follows:

.. code-block:: text

  extern "C" uint8_t* encrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
  extern "C" uint8_t* decrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
