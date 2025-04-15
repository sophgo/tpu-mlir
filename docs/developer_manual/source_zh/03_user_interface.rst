用户界面
========

本章介绍用户的使用界面, 包括转换模型的基本过程, 和各类工具的使用方法。

模型转换过程
--------------------

基本操作过程是用 ``model_transform.py`` 将模型转成mlir文件，然后用 ``model_deploy.py`` 将mlir转成对应的model。以 ``somenet.onnx`` 模型为例，操作步骤如下：

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

支持图片输入
~~~~~~~~~~~~~~~

当用图片作为输入的时候, 需要指定预处理信息, 如下:

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

支持多输入
~~~~~~~~~~~~~~~~

当模型有多输入的时候, 可以传入1个npz文件, 或者按顺序传入多个npy文件, 用逗号隔开。如下:

.. code-block:: shell

    $ model_transform.py \
        --model_name multi_input_net \
        --model_def  multi_input_net.onnx \
        --test_input multi_input_net_in.npz \ # a.npy,b.npy,c.npy
        --test_result multi_input_net_top_outputs.npz \
        --mlir multi_input_net.mlir

支持INT8对称和非对称
~~~~~~~~~~~~~~~~~~~~

如果需要转INT8模型, 则需要进行calibration。如下:

.. code-block:: shell

  $ run_calibration.py somenet.mlir \
      --dataset dataset \
      --input_num 100 \
      -o somenet_cali_table

传入校准表生成模型, 如下:

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

支持混精度
~~~~~~~~~~~~~~

当INT8模型精度不满足业务要求时, 可以尝试使用混精度, 先生成量化表, 如下:

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

然后将量化表传入生成模型, 如下:

.. code-block:: shell

    $ model_deploy.py \
       --mlir somenet.mlir \
       --quantize INT8 \
       --calibration_table somenet_cali_table \
       --quantize_table somenet_qtable \
       --processor BM1684X \
       --model somenet_mix.bmodel

支持量化模型TFLite
~~~~~~~~~~~~~~~~~~~

支持TFLite模型的转换, 命令参考如下:

.. code-block:: shell

    # TFLite转模型举例
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

支持Caffe模型
~~~~~~~~~~~~~~~~

.. code-block:: shell

    # Caffe转模型示例
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

支持LLM模型
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ llm_convert.py \
        -m /workspace/Qwen2.5-VL-3B-Instruct-AWQ \
        -s 2048 \
        -q w4bf16 \
        -c bm1684x \
        --max_pixels 672,896 \
        -o qwen2.5vl_3b


工具参数介绍
-------------

model_transform.py
~~~~~~~~~~~~~~~~~~~~~~~~

用于将各种神经网络模型转换成MLIR文件（``.mlir``后缀）以及配套的权重文件（ ``${model_name}_top_${quantize}_all_weight.npz`` ），支持的参数如下:

.. list-table:: model_transform 参数功能
   :widths: 20 12 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - model_name
     - 是
     - 指定模型名称
   * - model_def
     - 是
     - 指定模型定义文件, 比如 ``.onnx`` 或 ``.tflite`` 或 ``.prototxt`` 文件
   * - mlir
     - 是
     - 指定输出的mlir文件名称和路径, ``.mlir`` 后缀
   * - input_shapes
     - 否
     - 指定输入的shape, 例如 ``[[1,3,640,640]]`` ; 二维数组, 可以支持多输入情况
   * - model_extern
     - 否
     - 其他模型定义文件, 用于与model_def模型合并（目前主要用于MaskRCNN功能），默认处理为None; 多个输入模型文件时，用 ``,`` 隔开
   * - model_data
     - 否
     - 指定模型权重文件, caffe模型需要, 对应 ``.caffemodel`` 文件
   * - input_types
     - 否
     - 当模型为 ``.pt`` 文件时指定输入的类型, 例如int32; 多输入用 ``,`` 隔开; 不指定情况下默认处理为float32。
   * - keep_aspect_ratio
     - 否
     - 当test_input与input_shapes不同时，在resize时是否保持长宽比, 默认为false; 设置时会对不足部分补0
   * - mean
     - 否
     - 图像每个通道的均值, 默认为 0.0,0.0,0.0
   * - scale
     - 否
     - 图片每个通道的比值, 默认为 1.0,1.0,1.0
   * - pixel_format
     - 否
     - 图片类型, 可以是rgb、bgr、gray、rgbd四种情况, 默认为bgr
   * - channel_format
     - 否
     - 通道类型, 对于图片输入可以是nhwc或nchw, 非图片输入则为none, 默认是nchw
   * - output_names
     - 否
     - 指定输出的名称, 如果不指定, 则用模型的输出; 指定后按照该指定名称的顺序做输出
   * - add_postprocess
     - 否
     - 将后处理融合到模型中, 指定后处理类型, 目前支持yolov3、yolov3_tiny、yolov5、yolov8、yolov11、ssd、yolov8_seg后处理
   * - test_input
     - 否
     - 指定输入文件用于验证, 可以是jpg或npy或npz; 可以不指定, 则不会正确性验证
   * - test_result
     - 否
     - 指定验证后的输出文件, ``.npz`` 格式
   * - excepts
     - 否
     - 指定需要排除验证的网络层的名称, 多个用 ``,`` 隔开
   * - onnx_sim
     - 否
     - onnx-sim 的可选项参数，目前仅支持 skip_fuse_bn 选项，用于关闭 batch_norm 和 Conv 层的合并
   * - debug
     - 否
     - 保存可用于debug的模型
   * - tolerance
     - 否
     - 模型转换的余弦与欧式相似度的误差容忍度，默认为0.99,0.99
   * - cache_skip
     - 否
     - 是否在生成相同mlir/bmodel时跳过正确性的检查
   * - dynamic_shape_input_names
     - 否
     - 具有动态shape的输入的名称列表, 例如input1,input2. 如果设置了, model_deploy需要设置参数'dynamic'.
   * - shape_influencing_input_names
     - 否
     - 在推理过程中会影响其他张量形状的输入的名称列表, 例如input1,input2. 如果设置了, 则必须指定test_input, 且model_deploy需要设置参数'dynamic'。
   * - dynamic
     - 否
     - 该参数只对onnx模型有效. 如果设置了, 工具链会自动将模型带有dynamic_axis的输入加入dynamic_shape_input_names列表中, 将模型中1维的输入加入shape_influencing_input_names列表中, 且model_deploy需要设置参数'dynamic'.
   * - resize_dims
     - 否
     - 预处理前的原始输入图像尺寸h,w，默认为模型原始输入尺寸
   * - pad_value
     - 否
     - 图片缩放时边框填充大小
   * - pad_type
     - 否
     - 图片缩放时的填充类型，有normal, center
   * - preprocess_list
     - 否
     - 输入是否需要做预处理的选项,例如:'1,3' 表示输入1&3需要进行预处理, 缺省代表所有输入要做预处理
   * - path_yaml
     - 否
     - 单个 yaml文件 的路径（当前主要用于MaskRCNN参数配置）
   * - enable_maskrcnn
     - 否
     - 是否启用 MaskRCNN大算子.
   * - yuv_type
     - 否
     - 采用'.yuv'文件作为输入时指定其类型


转成mlir文件后, 会生成一个 ``${model_name}_in_f32.npz`` 文件, 该文件是后续模型的输入文件。

注意：

1. `model_transform.py` 阶段输入的预处理参数会用于对 `test_input` 进行预处理，并且会记录到 `mlir` 文件中，后续执行 `run_calibration.py` 时会读取该预处理参数对校准数据集进行预处理。若 `model_transform.py` 阶段没有对应参数输入将可能影响到模型的实际量化效果。

2. 预处理参数计算方式：

.. math::

    input &= scale \times (input - mean) \\
    scale &= \frac{1}{255 \times std}


run_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~

用少量的样本做calibration, 得到网络的校准表, 即每一层op的threshold/min/max。

支持的参数如下:

.. list-table:: run_calibration 参数功能
   :widths: 25 12 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - 无
     - 是
     - 指定mlir文件
   * - sq
     - 否
     - SmoothQuant
   * - we
     - 否
     - 跨层权重均衡
   * - bc
     - 否
     - 偏差校正
   * - dataset
     - 否
     - 指定输入样本的目录, 该路径放对应的图片, 或npz, 或npy
   * - data_list
     - 否
     - 指定样本列表, 与dataset必须二选一
   * - input_num
     - 否
     - 指定校准数量, 如果为0, 则使用全部样本
   * - inference_num
     - 否
     - search_threshold 和 search_qtable 过程中所需推理图片数量，通常小于input_num
   * - bc_inference_num
     - 否
     - 偏差校正过程中所需推理图片数量
   * - tune_num
     - 否
     - 指定微调样本数量, 默认为10
   * - tune_list
     - 否
     - 指定微调样本文件
   * - histogram_bin_num
     - 否
     - 直方图bin数量, 默认2048
   * - expected_cos
     - 否
     - 期望search_qtable混精模型输出与浮点模型输出的相似度,取值范围[0,1]
   * - min_layer_cos
     - 否
     - bias_correction中该层量化输出与浮点输出的相似度下限,当低于该下限时需要对该层进行补偿,取值范围[0,1]
   * - max_float_layers
     - 否
     - search_qtable 浮点层数量
   * - processor
     - 否
     - 处理器类型
   * - cali_method
     - 否
     - 选择量化门限计算方法
   * - fp_type
     - 否
     - search_qtable浮点层数据类型
   * - post_process
     - 否
     - 后处理路径
   * - global_compare_layers
     - 否
     - 指定全局对比层，例如 layer1,layer2 或 layer1:0.3,layer2:0.7
   * - search
     - 否
     - 指定搜索类型,其中包括search_qtable,search_threshold,false。其中默认为false,不开启搜索
   * - transformer
     - 否
     - 是否是transformer模型,search_qtable中如果是transformer模型可分配指定加速策略
   * - quantize_method_list
     - 否
     - search_qtable用来搜索的门限方法
   * - benchmark_method
     - 否
     - 指定search_threshold中相似度计算方法
   * - kurtosis_analysis
     - 否
     - 指定生成各层激活值的kurtosis
   * - part_quantize
     - 否
     - 指定模型部分量化,获得cali_table同时会自动生成qtable。可选择N_mode,H_mode,custom_mode,H_mode通常精度较好
   * - custom_operator
     - 否
     - 指定需要量化的算子,配合开启上述custom_mode后使用
   * - part_asymmetric
     - 否
     - 指定当开启对称量化后,模型某些子网符合特定pattern时,将对应位置算子改为非对称量化
   * - mix_mode
     - 否
     - 指定search_qtable特定的混精类型,目前支持8_16和4_8两种
   * - cluster
     - 否
     - 指定search_qtable寻找敏感层时采用聚类算法
   * - quantize_table
     - 否
     - search_qtable输出的混精度量化表
   * - o
     - 是
     - 输出calibration table文件
   * - debug_cmd
     - 否
     - debug cmd
   * - debug_log
     - 否
     - 日志输出级别

校准表的样板如下:

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

它分为4列: 第一列是Tensor的名字; 第二列是阈值(用于对称量化);
第三列第四列是min/max, 用于非对称量化。

.. _model_deploy:

model_deploy.py
~~~~~~~~~~~~~~~~~

将mlir文件转换成相应的model, 参数说明如下:


.. list-table:: model_deploy 参数功能
   :widths: 18 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - mlir
     - 是
     - 指定mlir文件
   * - processor
     - 是
     - 指定模型将要用到的平台, 支持BM1684，BM1684X，BM1688，BM1690，CV186X，CV183X，CV182X，CV181X，CV180X
   * - quantize
     - 是
     - 指定默认量化类型, 支持F32/F16/BF16/INT8等, 不同处理器支持的量化类型如下表所示。
   * - quant_input
     - 否
     - 指定输入数据类型是否与量化类型一致，例如int8模型指定quant_input，那么输入数据类型也为int8，若不指定则为F32
   * - quant_output
     - 否
     - 指定输出数据类型是否与量化类型一致，例如int8模型指定quant_input，那么输出数据类型也为int8，若不指定则为F32
   * - quant_input_list
     - 否
     - 选择要转换的索引，例如 1,3 表示第一个和第三个输入的强制转换
   * - quant_output_list
     - 否
     - 选择要转换的索引，例如 1,3 表示第一个和第三个输出的强制转换
   * - quantize_table
     - 否
     - 指定混精度量化表路径, 如果没有指定则按quantize类型量化; 否则优先按量化表量化
   * - fuse_preprocess
     - 否
     - 指定是否将预处理融合到模型中，如果指定了此参数，则模型输入为uint8类型，直接输入resize后的原图即可
   * - calibration_table
     - 否
     - 指定校准表路径, 当存在INT8/F8E4M3量化的时候需要校准表
   * - high_precision
     - 否
     - 打开时一部分算子会固定用float32
   * - tolerance
     - 否
     - 表示 MLIR 量化后的结果与 MLIR fp32推理结果余弦与欧式相似度的误差容忍度，默认为0.8,0.5
   * - test_input
     - 否
     - 指定输入文件用于验证, 可以是jpg或npy或npz; 可以不指定, 则不会正确性验证
   * - test_reference
     - 否
     - 用于验证模型正确性的参考数据(使用npz格式)。其为各算子的计算结果
   * - excepts
     - 否
     - 指定需要排除验证的网络层的名称, 多个用,隔开
   * - op_divide
     - 否
     - CV183x/CV182x/CV181x/CV180x only, 尝试将较大的op拆分为多个小op以达到节省ion内存的目的, 适用少数特定模型
   * - model
     - 是
     - 指定输出的model文件名称和路径
   * - debug
     - 否
     - 是否保留中间文件
   * - asymmetric
     - 否
     - 指定做int8非对称量化
   * - dynamic
     - 否
     - 动态编译
   * - includeWeight
     - 否
     - tosa.mlir 的 includeWeight
   * - customization_format
     - 否
     - 指定模型输入帧的像素格式
   * - compare_all
     - 否
     - 指定对比模型所有的张量
   * - num_device
     - 否
     - 用于并行计算的设备数量,默认1
   * - num_core
     - 否
     - 用于并行计算的智能视觉深度学习处理器核心数量,默认1
   * - skip_validation
     - 否
     - 跳过检查 bmodel 的正确性
   * - merge_weight
     - 否
     - 将权重与之前生成的 cvimodel 合并为一个权重二进制文件，默认否
   * - model_version
     - 否
     - 如果需要旧版本的cvimodel,请设置版本,例如1.2，默认latest
   * - q_group_size
     - 否
     - 每组定量的组大小，仅用于 W4A16/W8A16 定量模式，默认0
   * - q_symmetric
     - 否
     - 指定做W4A16对称量化，仅用于 W4A16/W8A16 定量模式
   * - compress_mode
     - 否
     - 指定模型的压缩模式："none","weight","activation","all"。支持bm1688, 默认为"none",不进行压缩
   * - opt_post_processor
     - 否
     - 是否对LayerGroup的结果继续图优化, 支持mars3, 默认为"none",不进行
   * - lgcache
     - 否
     - 指定是否暂存 LayerGroup 的切分结果： "true", "false"。默认为"true", 将每个子网的切分结果保存到工作目录 "cut_result_{subnet_name}.mlircache"
   * - cache_skip
     - 否
     - 是否在生成相同mlir/bmodel时跳过正确性的检查
   * - aligned_input
     - 否
     - 是否输入图像的宽/通道是对齐的，仅用于CV系列处理器的VPSS输入对齐
   * - group_by_cores
     - 否
     - layer groups是否根据core数目进行强制分组, 可选auto/true/false, 默认为auto
   * - opt
     - 否
     - LayerGroup优化类型，可选1/2/3, 默认为2。1：简单LayerGroup模式，所有算子会尽可能做Group，编译速度较快；2：通过动态编译计算全局cycle最优的Group分组，适用于推理图编译；3：线性规划LayerGroup模式，适用于模型训练图编译。
   * - addr_mode
     - 否
     - 设置地址分配模式['auto', 'basic', 'io_alone', 'io_tag', 'io_tag_fuse', 'io_reloc'], 默认为auto
   * - disable_layer_group
     - 否
     - 是否关闭LayerGroup
   * - disable_gdma_check
     - 否
     - 是否关闭gdma地址检查
   * - do_winograd
     - 否
     - 是否使用WinoGrad卷积, 仅用于BM1684平台
   * - time_fixed_subnet
     - 否
     - 将模型按固定时长间隔分割，支持['normal', 'limit', 'custom']，目前支持BM1684X和BM1688处理器，打开可能影响模型性能
   * - subnet_params
     - 否
     - 当time_fixed_subnet为custom时，用于设定子网的频率(MHZ)和耗时(ms)
   * - matmul_perchannel
     - 否
     - MatMul是否使用per-channel量化模式，目前支持BM1684X和BM1688处理器，打开可能影响运行时间
   * - enable_maskrcnn
     - 否
     - 是否启用 MaskRCNN大算子.

对于不同处理器和支持的量化类型对应关系如下表所示：

.. list-table:: 不同处理器支持的 quantize 量化类型
   :widths: 15 30
   :header-rows: 1

   * - 处理器
     - 支持的quantize
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

其中， ``W4A16`` 与 ``W8A16`` 的 ``Weight-only`` 量化模式仅作用于 MatMul 运算，其余算子根据实际情况仍会进行 ``F16`` 或 ``BF16`` 量化。


llm_convert.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于将HuggingFace LLM模型转换成bmodel, 支持的参数如下:

.. list-table:: llm_convert 参数功能
   :widths: 18 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - model_path
     - 是
     - 指定模型路径
   * - seq_length
     - 是
     - 指定序列最大长度
   * - quantize
     - 是
     - 指定量化类型, 如w4bf16/w4f16/bf16/f16
   * - q_group_size
     - 否
     - 每组量化的组大小
   * - chip
     - 是
     - 指定处理器类型, 支持bm1684x/bm1688/cv186ah
   * - max_pixels
     - 否
     - 多模态参数, 指定最大尺寸, 可以是 ``672,896``, 也可以是 ``602112``
   * - num_device
     - 否
     - 指定 bmodel 部署的设备数
   * - num_core
     - 否
     - 指定 bmodel 部署使用的核数, 0表示采用最大核数
   * - max_input_length
     - 否
     - 指定最大输入长度, 默认为seq_length
   * - embedding_disk
     - 否
     - 如果设置该标志, 则将word_embedding导出为二进制文件, 并通过 CPU 进行推理
   * - out_dir
     - 是
     - 指定输出的 bmodel 文件保存路径

model_runner.py
~~~~~~~~~~~~~~~~~~

对模型进行推理, 支持mlir/pytorch/onnx/tflite/bmodel/prototxt。

执行参考如下:

.. code-block:: shell

   $ model_runner.py \
      --input sample_in_f32.npz \
      --model sample.bmodel \
      --output sample_output.npz \
      --out_fixed

支持的参数如下:

.. list-table:: model_runner 参数功能
   :widths: 18 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - input
     - 是
     - 指定模型输入, npz文件
   * - model
     - 是
     - 指定模型文件, 支持mlir/pytorch/onnx/tflite/bmodel/prototxt
   * - dump_all_tensors
     - 否
     - 开启后对导出所有的结果, 包括中间tensor的结果
   * - out_fixed
     - 否
     - 开启后当出现int8类型定点数时不再自动转成float32类型进行打印


npz_tool.py
~~~~~~~~~~~~~~~~

npz在TPU-MLIR工程中会大量用到, 包括输入输出的结果等等。npz_tool.py用于处理npz文件。

执行参考如下:

.. code-block:: shell

   # 查看sample_out.npz中output的数据
   $ npz_tool.py dump sample_out.npz output

支持的功能如下:

.. list-table:: npz_tool 功能
   :widths: 18 60
   :header-rows: 1

   * - 功能
     - 描述
   * - dump
     - 得到npz的所有tensor信息
   * - compare
     - 比较2个npz文件的差异
   * - to_dat
     - 将npz导出为dat文件, 连续的二进制存储

visual.py
~~~~~~~~~~~~~~~~

量化网络如果遇到精度对比不过或者比较差, 可以使用此工具逐层可视化对比浮点网络和量化后网络的不同, 方便进行定位和手动调整。

执行命令可参考如下：

.. code-block:: shell

   # 以使用9999端口为例
   $ visual.py \
     --f32_mlir netname.mlir \
     --quant_mlir netname_int8_sym_tpu.mlir \
     --input top_input_f32.npz --port 9999

支持的功能如下:

.. list-table:: visual 功能
   :widths: 18 60
   :header-rows: 1

   * - 功能
     - 描述
   * - f32_mlir
     - fp32网络mlir文件
   * - quant_mlir
     - 量化后网络mlir文件
   * - input
     - 测试输入数据, 可以是图像文件或者npz文件
   * - port
     - 使用的TCP端口, 默认10000, 需要在启动docker时映射至系统端口
   * - host
     - 使用的host ip地址, 默认0.0.0.0
   * - manual_run
     - 启动后是否自动进行网络推理比较, 默认False, 会自动推理比较

注意：需要在model_deploy.py阶段打开 ``--debug`` 选项保留中间文件供visual.py使用，工具的详细使用说明见(:ref:`visual-usage`)。

mlir2graph.py
~~~~~~~~~~~~~~~~

基于 dot 对 mlir 文件可视化，支持所有阶段的 mlir 文件。执行后会在 mlir 对应目录生成对应的 .dot 文件和 .svg 文件。其中 .dot 文件可以基于 dot 渲染成其他格式的命令。.svg 是默认输出的渲染格式。可以直接在浏览器打开。

执行命令可参考如下：

.. code-block:: shell

   $ mlir2graph.py \
     --mlir netname.mlir

对较大的 mlir 文件，dot 文件使用原始的渲染算法可能会消耗较长的时间，可以添加 --is_big 参数，会减少算法的迭代时间，出图更快：

.. code-block:: shell

   $ mlir2graph.py \
     --mlir netname.mlir --is_big

支持的功能如下:

.. list-table:: mlir2graph 功能
   :widths: 18 60
   :header-rows: 1

   * - 功能
     - 描述
   * - mlir
     - 任意 mlir 文件
   * - is_big
     - 是否是比较大的 mlir 文件，没有明确指标，一般靠人为根据渲染用时判断
   * - failed_keys
     - 对比失败的节点名列表，多个用 "," 隔开，在渲染后对应节点会渲染为红色
   * - bmodel_checker_data
     - 使用 bmodel_checker.py 生成的 failed.npz 文件路径，当指定该路径时，会自动解析错误节点，并将对应节点渲染为红色
   * - output
     - 输出文件的路径，默认为 --mlir 的路径加格式后缀，如 netname.mlir.dot/netname.mlir.svg


gen_rand_input.py
~~~~~~~~~~~~~~~~~~~~

在模型转换时，如果不想额外准备测试数据(test_input)，可以使用此工具生成随机的输入数据，方便模型验证工作。

基本操作过程是用 ``model_transform.py`` 将模型转成mlir文件, 此步骤不进行模型验证；接下来，用 ``gen_rand_input.py`` 读取上一步生成的mlir文件，生成用于模型验证的随机测试数据；
最后，再次使用 ``model_transform.py`` 进行完整的模型转换和验证工作。

执行的命令可参考如下：

.. code-block:: shell

    # 模型初步转换为mlir文件
    $ model_transform.py \
        --model_name yolov5s  \
        --model_def ../regression/model/yolov5s.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names 350,498,646 \
        --mlir yolov5s.mlir

    # 生成随机测试数据，这里生成的是伪测试图片
    $ gen_rand_input.py \
        --mlir yolov5s.mlir \
        --img --output yolov5s_fake_img.png

    # 完整的模型转换和验证
    $ model_transform.py \
        --model_name yolov5s  \
        --model_def ../regression/model/yolov5s.onnx \
        --input_shapes [[1,3,640,640]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --test_input yolov5s_fake_img.png    \
        --test_result yolov5s_top_outputs.npz \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names 350,498,646 \
        --mlir yolov5s.mlir

更详细的使用方法可参考如下：

.. code-block:: shell

    # 可为多个输入分别指定取值范围
    $ gen_rand_input.py \
      --mlir ernie.mlir \
      --ranges [[0,300],[0,0]] \
      --output ern.npz

    # 可为输入指定type类型，如不指定，默认从mlir文件中读取
    $ gen_rand_input.py \
      --mlir resnet.mlir \
      --ranges [[0,300]] \
      --input_types si32 \
      --output resnet.npz

    # 指定生成随机图片，不指定取值范围和数据类型
    $ gen_rand_input.py \
        --mlir yolov5s.mlir \
        --img --output yolov5s_fake_img.png

支持的功能如下:

.. list-table:: gen_rand_input 功能
   :widths: 18 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - mlir
     - 是
     - 指定输出的mlir文件名称和路径, ``.mlir`` 后缀
   * - img
     - 否
     - 用于CV任务生成随机图片，否则生成npz文件。默认图片的取值范围为[0,255]，数据类型为'uint8'，不通过'ranges'或'input_types'更改。
   * - ranges
     - 否
     - 指定模型输入的取值范围，以列表形式表现，如[[0,300],[0,0]]。如果指定生成图片，则不需要指定取值范围，默认[0,255]。其他情况下，需要指定取值范围。
   * - input_types
     - 否
     - 指定模型输入的数据类型，如si32,f32。目前仅支持 'si32' 和 'f32' 类型。如果不填，默认从mlir中读取。如果指定生成图片，则不需要指定数据类型，默认'uint8'。
   * - output
     - 是
     - 指定输出的名称

注意：CV相关模型通常会对输入图片进行一系列预处理，为保证模型正确性验证通过，需要用'--img'生成随机图片作为输入，不能使用随机npz文件作为输入。
值得关注的是，随机输入可能会引起模型正确性验证对比不通过，特别是NLP相关模型，因此建议优先使用真实的测试数据。


model_tool
~~~~~~~~~~~~~~~~~~~~

该工具用于处理最终的模型文件"bmodel"或者"cvimodel"，所有参数及对应功能描述可通过执行以下命令查看：

.. code-block:: shell

   $ model_too

以下均以"xxx.bmodel"为例，介绍该工具的主要功能。

1) 查看bmodel的基本信息

执行参考如下:

.. code-block:: shell

   $ model_tool --info xxx.bmodel

显示模型的基本信息，包括模型的编译版本，编译日期，模型中网络名称，输入和输出参数等等。
显示效果如下：

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

2) 合并多个bmodel

执行参考如下:

.. code-block:: shell

   $ model_tool --combine a.bmodel b.bmodel c.bmodel -o abc.bmodel

将多个bmodel合并成一个bmodel，如果bmodel中存在同名的网络，则会分不同的stage。

3) 分解成多个bmodel

执行参考如下:

.. code-block:: shell

   $ model_tool --extract abc.bmodel

将一个bmodel分解成多个bmodel，与combine命令是相反的操作。

4) 显示权重信息

执行参考如下:

.. code-block:: shell

   $ model_tool --weight xxx.bmodel

显示不同网络的各个算子的权重范围信息，显示效果如下：

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

5) 更新权重

执行参考如下:

.. code-block:: shell

   # 将src.bmodel中网络名为src_net的网络，在0x2000位置的权重，更新到dst.bmodel的dst_net的0x1000位置
   $ model_tool --update_weight dst.bmodel dst_net 0x1000 src.bmodel src_net 0x2000

可以实现将模型权重进行更新。比如某个模型的某个算子权重需要更新，则将该算子单独编译成bmodel，然后将其权重更新到原始的模型中。

6) 模型加密与解密

执行参考如下:

.. code-block:: shell

   # -model输入combine后的模型或正常bmodel，-net输入要加密的网络，-lib实现具体的加密算法，-o输出加密后模型的名称
   $ model_tool --encrypt -model combine.bmodel -net block_0 -lib libcipher.so -o encrypted.bmodel
   $ model_tool --decrypt -model encrypted.bmodel -lib libcipher.so -o decrypted.bmodel

可以实现将模型的权重、flatbuffer结构化数据、header都进行加密。
加解密接口必须按照C风格来实现，不能使用C++，接口规定如下：

.. code-block:: text

  extern "C" uint8_t* encrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
  extern "C" uint8_t* decrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
