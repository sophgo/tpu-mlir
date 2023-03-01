用户界面
========

本章介绍用户的使用界面, 包括转换模型的基本过程, 和各类工具的使用方法。

模型转换过程
--------------------

基本操作过程是用 ``model_transform.py`` 将模型转成mlir文件, 然后用
``model_deploy.py`` 将mlir转成对应的model。如下:

.. code-block:: shell

    # To MLIR
    $ model_transform.py \
        --model_name resnet \
        --model_def  resnet.onnx \
        --test_input resnet_in.npz \
        --test_result resnet_top_outputs.npz \
        --mlir resnet.mlir

    # To Float Model
    $ model_deploy.py \
       --mlir resnet.mlir \
       --quantize F32 \ # F16/BF16
       --chip bm1684x \
       --test_input resnet_in_f32.npz \
       --test_reference resnet_top_outputs.npz \
       --model resnet50_f32.bmodel

支持图片输入
~~~~~~~~~~~~~~~

当用图片做为输入的时候, 需要指定预处理信息, 如下:

.. code-block:: shell

    $ model_transform.py \
        --model_name resnet \
        --model_def resnet.onnx \
        --input_shapes [[1,3,224,224]] \
        --mean 103.939,116.779,123.68 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input cat.jpg \
        --test_result resnet_top_outputs.npz \
        --mlir resnet.mlir

支持多输入
~~~~~~~~~~~~~~~~

当模型有多输入的时候, 可以传入1个npz文件, 或者按顺序传入多个npz文件, 用逗号隔开。如下:

.. code-block:: shell

    $ model_transform.py \
        --model_name somenet \
        --model_def  somenet.onnx \
        --test_input somenet_in.npz \ # a.npy,b.npy,c.npy
        --test_result somenet_top_outputs.npz \
        --mlir somenet.mlir

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
       --mlir resnet.mlir \
       --quantize INT8 \
       # --asymmetric \
       --calibration_table somenet_cali_table \
       --chip bm1684x \
       --test_input somenet_in_f32.npz \
       --test_reference somenet_top_outputs.npz \
       --tolerance 0.9,0.7 \
       --model somenet_int8.bmodel

支持混精度
~~~~~~~~~~~~~~

当INT8模型精度不满足业务要求时, 可以尝试使用混精度, 先生成量化表, 如下:

.. code-block:: shell

   $ run_qtable.py somenet.mlir \
       --dataset dataset \
       --calibration_table somenet_cali_table \
       --chip bm1684x \
       -o somenet_qtable

然后将量化表传入生成模型, 如下:

.. code-block:: shell

    $ model_deploy.py \
       --mlir resnet.mlir \
       --quantize INT8 \
       --calibration_table somenet_cali_table \
       --quantize_table somenet_qtable \
       --chip bm1684x \
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
       --asymmetric \
       --chip bm1684x \
       --test_input resnet50_tf_in_f32.npz \
       --test_reference resnet50_tf_top_outputs.npz \
       --tolerance 0.95,0.85 \
       --model resnet50_tf_1684x.bmodel


支持Caffe模型
~~~~~~~~~~~~~~~~

.. code-block:: shell

    # Caffe转模型举例
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


工具参数介绍
-------------

model_transform.py
~~~~~~~~~~~~~~~~~~~~~~~~

用于将各种神经网络模型转换成MLIR文件, 支持的参数如下:


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
     - 指定模型定义文件, 比如`.onnx`或`.tflite`或`.prototxt`文件
   * - model_data
     - 否
     - 指定模型权重文件, caffe模型需要, 对应`.caffemodel`文件
   * - input_shapes
     - 否
     - 指定输入的shape, 例如[[1,3,640,640]]; 二维数组, 可以支持多输入情况
   * - resize_dims
     - 否
     - 原始图片需要resize之后的尺寸; 如果不指定, 则resize成模型的输入尺寸
   * - keep_aspect_ratio
     - 否
     - 在Resize时是否保持长宽比, 默认为false; 设置时会对不足部分补0
   * - mean
     - 否
     - 图像每个通道的均值, 默认为0.0,0.0,0.0
   * - scale
     - 否
     - 图片每个通道的比值, 默认为1.0,1.0,1.0
   * - pixel_format
     - 否
     - 图片类型, 可以是rgb、bgr、gray、rgbd四种情况
   * - output_names
     - 否
     - 指定输出的名称, 如果不指定, 则用模型的输出; 指定后用该指定名称做输出
   * - test_input
     - 否
     - 指定输入文件用于验证, 可以是图片或npy或npz; 可以不指定, 则不会正确性验证
   * - test_result
     - 否
     - 指定验证后的输出文件
   * - excepts
     - 否
     - 指定需要排除验证的网络层的名称, 多个用,隔开
   * - mlir
     - 是
     - 指定输出的mlir文件名称和路径


转成mlir文件后, 会生成一个 ``${model_name}_in_f32.npz`` 文件, 该文件是后续模型的输入文件。


run_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~

用少量的样本做calibration, 得到网络的校准表, 即每一层op的threshold/min/max。

支持的参数如下:

.. list-table:: run_calibration 参数功能
   :widths: 20 12 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - 无
     - 是
     - 指定mlir文件
   * - dataset
     - 否
     - 指定输入样本的目录, 该路径放对应的图片, 或npz, 或npy
   * - data_list
     - 否
     - 指定样本列表, 与dataset必须二选一
   * - input_num
     - 否
     - 指定校准数量, 如果为0, 则使用全部样本
   * - tune_num
     - 否
     - 指定微调样本数量, 默认为10
   * - histogram_bin_num
     - 否
     - 直方图bin数量, 默认2048
   * - o
     - 是
     - 输出calibration table文件

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

run_qtable.py
~~~~~~~~~~~~~~~~

使用 ``run_qtable.py`` 生成混精度量化表, 相关参数说明如下:

.. list-table:: run_qtable.py 参数功能
   :widths: 18 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - 无
     - 是
     - 指定mlir文件
   * - dataset
     - 否
     - 指定输入样本的目录, 该路径放对应的图片, 或npz, 或npy
   * - data_list
     - 否
     - 指定样本列表, 与dataset必须二选一
   * - calibration_table
     - 是
     - 输入校准表
   * - chip
     - 是
     - 指定模型将要用到的平台, 支持bm1684x/bm1684/cv183x/cv182x/cv181x/cv180x
   * - input_num
     - 否
     - 指定输入样本数量, 默认用10个
   * - loss_table
     - 否
     - 输出Loss表, 默认为full_loss_table.txt
   * - o
     - 是
     - 输出混精度量化表

混精度量化表的样板如下:

.. code-block:: shell

    # genetated time: 2022-11-09 21:35:47.981562
    # sample number: 3
    # all int8 loss: -39.03119206428528
    # chip: bm1684x  mix_mode: F32
    ###
    # op_name   quantize_mode
    conv2_1/linear/bn F32
    conv2_2/dwise/bn  F32
    conv6_1/linear/bn F32

它分为2列: 第一列对应layer的名称, 第二列对应量化模式。

同时会生成loss表, 默认为 ``full_loss_table.txt``, 样板如下:

.. code-block:: shell

    # genetated time: 2022-11-09 22:30:31.912270
    # sample number: 3
    # all int8 loss: -39.03119206428528
    # chip: bm1684x  mix_mode: F32
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

它代表对应的Layer改成浮点计算后, 得到的输出的Loss。

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
   * - chip
     - 是
     - 指定模型将要用到的平台, 支持bm1684x/bm1684/cv183x/cv182x/cv181x/cv180x
   * - quantize
     - 是
     - 指定默认量化类型, 支持F32/F16/BF16/INT8
   * - quantize_table
     - 否
     - 指定混精度量化表路径, 如果没有指定则按quantize类型量化; 否则优先按量化表量化
   * - calibration_table
     - 否
     - 指定校准表路径, 当存在INT8量化的时候需要校准表
   * - tolerance
     - 否
     - 表示 MLIR 量化后的结果与 MLIR fp32推理结果相似度的误差容忍度
   * - test_input
     - 否
     - 指定输入文件用于验证, 可以是图片或npy或npz; 可以不指定, 则不会正确性验证
   * - test_reference
     - 否
     - 用于验证模型正确性的参考数据(使用npz格式)。其为各算子的计算结果
   * - excepts
     - 否
     - 指定需要排除验证的网络层的名称, 多个用,隔开
   * - model
     - 是
     - 指定输出的model文件名称和路径

model_runner.py
~~~~~~~~~~~~~~~~~~

对模型进行推理, 支持bmodel/mlir/onnx/tflite。

执行参考如下:

.. code-block:: shell

   $ model_runner.py \
      --input sample_in_f32.npz \
      --model sample.bmodel \
      --output sample_output.npz

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
     - 指定模型文件, 支持bmodel/mlir/onnx/tflite
   * - dump_all_tensors
     - 否
     - 开启后对导出所有的结果, 包括中间tensor的结果


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

量化网络如果遇到精度对比不过或者比较差，可以使用此工具逐层可视化对比浮点网络和量化后网络的不同，方便进行定位和手动调整。

执行命令可参考如下：

.. code-block:: shell

   # 以使用9999端口为例
   $ visual.py --fp32_mlir f32.mlir --quant_mlir quant.mlir --input top_input_f32.npz --port 9999

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
     - 测试输入数据，可以是图像文件或者npz文件
   * - port
     - 使用的TCP端口，默认10000，需要在启动docker时映射至系统端口
   * - manual_run
     - 启动后是否自动进行网络推理比较，默认False，会自动推理比较

