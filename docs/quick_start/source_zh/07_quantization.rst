.. _quantization:

===================
量化与量化调优
===================

神经网络在大规模部署时候，往往对吞吐量也就是推理时间有较高要求，芯片也专门对低比特计算进行了优化，其算力更加突出。所以以尽量高的精度进行低比特量化就显得尤为重要。
但是要保持高精度和高吞吐率，网络往往需要以混合精度方式运行，即大部分算子以低比特定点计算，少部分以浮点进行计算。如何决定哪些算子使用浮点往往与网络和网络权重有直接关系，
需要根据网络特点来选择。本章节主要以yolo系列的两个模型为例，说明了混合精度网络的工作原理和设置方法，并介绍了敏感层搜索和局部不量化两个工具的使用方法。


混精度使用方法
==================

本章以检测网络 ``yolov3 tiny`` 网络模型为例, 介绍如何使用混精度。
该模型来自https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3。

本章需要安装tpu_mlir。


安装tpu-mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[all]

准备工作目录
------------------

建立 ``yolov3_tiny`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都
放入 ``yolov3_tiny`` 目录中。

操作如下:

.. code-block:: shell
  :linenos:

   $ mkdir yolov3_tiny && cd yolov3_tiny
   $ wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx
   $ tpu_mlir_get_resource regression/dataset/COCO2017 .
   $ mkdir workspace && cd workspace

这里的 tpu_mlir_get_resource 命令用于从tpu_mlir的包安装根目录向外复制文件。

.. code-block:: shell

  $ tpu_mlir_get_resource [source_dir/source_file] [dst_dir]

source_dir/source_file的路径为相对于tpu_mlir的包安装根目录的位置，tpu_mlir包根目录下文件结构如下:

.. code ::
tpu_mlir
    ├── bin
    ├── customlayer
    ├── docs
    ├── lib
    ├── python
    ├── regression
    ├── src
    ├── entry.py
    ├── entryconfig.py
    ├── __init__.py
    └── __version__

注意如果 ``tiny-yolov3-11.onnx`` 用wget下载失败, 请用其他方式下载后放到 ``yolov3_tiny`` 目录。

验证原始模型
----------------

``detect_yolov3`` 是已经写好的验证命令, 可以用来对 ``yolov3_tiny`` 网络进行验证。执行过程如下:

.. code-block:: shell

   $ detect_yolov3 \
        --model ../tiny-yolov3-11.onnx \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_onnx.jpg

执行完后打印检测到的结果如下:

.. code-block:: shell

    person:60.7%
    orange:77.5%

并得到图片 ``yolov3_onnx.jpg``, 如下( :ref:`yolov3_onnx_result` ):

.. _yolov3_onnx_result:
.. figure:: ../assets/yolov3_onnx.jpg
   :height: 13cm
   :align: center

   yolov3_tiny ONNX执行效果


转成INT8对称量化模型
----------------------

如前面章节介绍的转模型方法, 这里不做参数说明, 只有操作过程。

第一步: 转成F32 mlir
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

第二步: 生成calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table

第三步: 转对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_int8.bmodel

第四步: 验证模型
~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3 \
        --model yolov3_int8.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_int8.jpg

执行完后有如下打印信息，表示检测到一个目标:

.. code-block:: shell

    orange:72.9%


得到图片 ``yolov3_int8.jpg``, 如下( :ref:`yolov3_int8_result` ):

.. _yolov3_int8_result:
.. figure:: ../assets/yolov3_int8.jpg
   :height: 13cm
   :align: center

   yolov3_tiny int8对称量化执行效果

可以看出int8对称量化模型相对原始模型, 在这张图上效果不佳，只检测到一个目标。


转成混精度量化模型
-----------------------

在转int8对称量化模型的基础上, 执行如下步骤。

第一步: 生成混精度量化表
~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``run_qtable`` 生成混精度量化表, 相关参数说明如下:

.. list-table:: run_qtable 参数功能
   :widths: 23 8 50
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
     - 指定模型将要用到的平台, 支持bm1688/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
   * - fp_type
     - 否
     - 指定混精度使用的float类型, 支持auto,F16,F32,BF16，默认为auto，表示由程序内部自动选择
   * - input_num
     - 否
     - 指定输入样本数量, 默认用10个
   * - expected_cos
     - 否
     - 指定期望网络最终输出层的最小cos值,一般默认为0.99即可，越小时可能会设置更多层为浮点计算
   * - min_layer_cos
     - 否
     - 指定期望每层输出cos的最小值，低于该值会尝试设置浮点计算, 一般默认为0.99即可
   * - debug_cmd
     - 否
     - 指定调试命令字符串，开发使用, 默认为空
   * - o
     - 是
     - 输出混精度量化表
   * - global_compare_layers
     - 否
     - 指定用于替换最终输出层的层，并用于全局比较,例如：\'layer1,layer2\' or \'layer1:0.3,layer2:0.7\'
   * - fp_type
     - 否
     - 指定混合精度的浮点类型
   * - loss_table
     - 否
     - 指定保存所有被量化成浮点类型的层的损失值的文件名，默认为full_loss_table.txt

本例中采用默认10张图片校准, 执行命令如下（对于CV18xx系列的芯片，将chip设置为对应的芯片名称即可）:

.. code-block:: shell

   $ run_qtable yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --min_layer_cos 0.999 \ #若这里使用默认的0.99时，程序会检测到原始int8模型已满足0.99的cos，从而直接不再搜索
       --expected_cos 0.9999 \
       -o yolov3_qtable

执行完后最后输出如下打印:

.. code-block:: shell

    int8 outputs_cos:0.999115 old
    mix model outputs_cos:0.999517
    Output mix quantization table to yolov3_qtable
    total time:44 second

上面int8 outputs_cos表示int8模型原本网络输出和fp32的cos相似度，mix model outputs_cos表示部分层使用混精度后网络输出的cos相似度，total time表示搜索时间为44秒，
另外，生成的混精度量化表 ``yolov3_qtable``, 内容如下:

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

该表中, 第一列表示相应的layer, 第二列表示类型, 支持的类型有F32/F16/BF16/INT8。
另外同时也会生成一个loss表文件 ``full_loss_table.txt``, 内容如下:

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


该表按cos从小到大顺利排列, 表示该层的前驱Layer根据各自的cos已换成相应的浮点模式后, 该层计算得到的cos, 若该cos仍小于前面min_layer_cos参数，则会将该层及直接后继层设置为浮点计算。
``run_qtable`` 会在每次设置某相邻2层为浮点计算后，接续计算整个网络的输出cos，若该cos大于指定的expected_cos，则退出搜素。因此，若设置更大的expected_cos，会尝试将更多层设为浮点计算


第二步: 生成混精度量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel

第三步: 验证混精度模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3 \
        --model yolov3_mix.bmodel \
        --input ../COCO2017/000000366711.jpg \
        --output yolov3_mix.jpg

执行完后打印结果为:

.. code-block:: shell

    person:63.9%
    orange:72.9%


得到图片yolov3_mix.jpg, 如下( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny 混精度对称量化执行效果

可以看出混精度后, 检测结果更接近原始模型的结果。

需要说明的是，除了使用run_qtable生成量化表外，也可根据模型中每一层的相似度对比结果，自行设置量化表中需要做混精度量化的OP的名称和量化类型。


敏感层搜索使用方法
==================

本章以检测网络 ``mobilenet-v2`` 网络模型为例, 介绍如何使用敏感层搜索。
该模型来自nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt。

本章需要安装tpu_mlir。


安装tpu-mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[all]

准备工作目录
------------------

建立 ``mobilenet-v2`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都放入 ``mobilenet-v2`` 目录中。

操作如下:

.. code-block:: shell
  :linenos:

   $ mkdir mobilenet-v2 && cd mobilenet-v2
   $ tpu_mlir_get_resource regression/dataset/ILSVRC2012 .
   $ wget https://github.com/sophgo/tpu-mlir/releases/download/v1.4-beta.0/mobilenet_v2.pt
   $ mkdir workspace && cd workspace

这里的 ``tpu_mlir_get_resource`` 命令用于从tpu_mlir的包安装根目录向外复制文件。

.. code-block:: shell

  $ tpu_mlir_get_resource [source_dir/source_file] [dst_dir]

source_dir/source_file的路径为相对于tpu_mlir的包安装根目录的位置，tpu_mlir包根目录下文件结构如下:

.. code ::
tpu_mlir
    ├── bin
    ├── customlayer
    ├── docs
    ├── lib
    ├── python
    ├── regression
    ├── src
    ├── entry.py
    ├── entryconfig.py
    ├── __init__.py
    └── __version__
    
测试Float和INT8对称量化模型分类效果
---------------------------------

如前面章节介绍的转模型方法, 这里不做参数说明, 只有操作过程。

第一步: 转成FP32 mlir
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

第二步: 生成calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

第三步: 转FP32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --chip bm1684 \
       --model mobilenet_v2_bm1684_f32.bmodel

第四步: 转对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table mobilenet_v2_cali_table \
       --model mobilenet_v2_bm1684_int8_sym.bmodel

第五步: 验证FP32模型和INT8对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classify_mobilenet_v2是已经写好的验证程序，可以用来对mobilenet_v2网络进行验证。执行过程如下，FP32模型：

.. code-block:: shell

   $ classify_mobilenet_v2 \
       --model_def mobilenet_v2_bm1684_f32.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_fp32_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

在输出结果图片上可以看到如下分类信息，正确结果tench排在第一名：

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
    n02422106 hartebeest
    n02749479 assault rifle, assault gun
    n02916936 bulletproof vest

INT8对称量化模型：

.. code-block:: shell

   $ classify_mobilenet_v2 \
       --model_def mobilenet_v2_bm1684_int8_sym.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

在输出结果图片上可以看到如下分类信息，正确结果tench排在第一名：

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02749479 assault 日file, assau
    n02536864 coho, cohoe, coho
    n02916936 bulletproof vest
    n04336792 stretcher

转成混精度量化模型
-----------------------

在转int8对称量化模型的基础上, 执行如下步骤。

第一步: 进行敏感层搜索
~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``run_sensitive_layer`` 搜索损失较大的layer，注意尽量使用bad cases进行敏感层搜索，相关参数说明如下:

.. list-table:: run_sensitive_layer 参数功能
   :widths: 23 8 50
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
     - 指定模型将要用到的平台, 支持bm1688/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
   * - fp_type
     - 否
     - 指定混精度使用的float类型, 支持auto,F16,F32,BF16，默认为auto，表示由程序内部自动选择
   * - input_num
     - 否
     - 指定用于量化的输入样本数量, 默认用10个
   * - inference_num
     - 否
     - 指定用于推理的输入样本数量, 默认用10个
   * - max_float_layers
     - 否
     - 指定用于生成qtable的op数量, 默认用5个
   * - tune_list
     - 否
     - 指定用于调整threshold的样本路径
   * - tune_num
     - 否
     - 指定用于调整threshold的样本数量，默认为5
   * - histogram_bin_num
     - 否
     - 指定用于kld方法中使用的bin数量，默认为2048
   * - post_process
     - 否
     - 用户自定义后处理文件路径, 默认为空
   * - expected_cos
     - 否
     - 指定期望网络最终输出层的最小cos值,一般默认为0.99即可，越小时可能会设置更多层为浮点计算
   * - debug_cmd
     - 否
     - 指定调试命令字符串，开发使用, 默认为空
   * - o
     - 是
     - 输出混精度量化表
   * - global_compare_layers
     - 否
     - 指定用于替换最终输出层的层，并用于全局比较,例如：\'layer1,layer2\' or \'layer1:0.3,layer2:0.7\'
   * - fp_type
     - 否
     - 指定混合精度的浮点类型

本例中采用100张图片做量化, 30张图片做推理，执行命令如下（对于CV18xx系列的芯片，将chip设置为对应的芯片名称即可）:

.. code-block:: shell

   $ run_sensitive_layer mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --calibration_table mobilenet_v2_cali_table \
       --chip bm1684 \
       --post_process post_process_func.py \
       -o mobilenet_v2_qtable

敏感层搜索支持用户自定义的后处理方法post_process_func.py，可以放在当前工程目录下，也可以放在其他位置，如果放在其他位置需要在post_process中指明文件的完整路径。
后处理方法函数名称需要定义为PostProcess，输入数据为网络的输出，输出数据为后处理结果：

.. code-block:: shell

   $ def PostProcess(data):
       print("in post process")
       return data

执行完后最后输出如下打印:

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

上面int8 outputs_cos表示int8模型原本网络输出和fp32的cos相似度，mix model outputs_cos表示前五个敏感层使用混精度后网络输出的cos相似度，total time表示搜索时间为402秒，
另外，生成的混精度量化表 ``mobilenet_v2_qtable``, 内容如下:

.. code-block:: shell

    # op_name   quantize_mode
    input3.1 F32
    input11.1 F32
    input128.1 F32
    input130.1 F32
    input127.1 F32

该表中, 第一列表示相应的layer, 第二列表示类型, 支持的类型有F32/F16/BF16/INT8。
与此同时，也会生成一个log日志文件 ``SensitiveLayerSearch``, 内容如下:

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


日志文件记录了每个op在每种量化方法（MAX/Percentile9999/KL）得到的threshold下，设置为int8后，混精度模型与原始float模型输出的相似度的loss（1-余弦相似度）。
同时也包含了屏幕端输出的每个op的loss信息以及最后的混精度模型与原始float模型的余弦相似度。
用户可以使用程序输出的qtable，也可以根据loss信息对qtable进行修改，然后生成混精度模型。
在敏感层搜索结束后，最优的threshold会被更新到一个新的量化表new_cali_table.txt，该量化表存储在当前工程目录下，在生成混精度模型时需要调用新量化表。
在本例中，根据输出的loss信息，观察到input3.1的loss比其他op高很多，可以在qtable中只设置input3.1为FP32。

第二步: 生成混精度量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table new_cali_table.txt \
       --quantize_table mobilenet_v2_qtable \
       --model mobilenet_v2_bm1684_mix.bmodel

第三步: 验证混精度模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ classify_mobilenet_v2 \
       --model_def mobilenet_v2_bm1684_mix.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

在输出结果图片上可以看到如下分类信息，可以看出混精度后, 正确结果tench排到了第一名。

.. code-block:: shell

    Top-5
    n01440764 tench, Tinca tinca
    n02749479 assault rifle, assault gun
    n02916936 bulletproof vest
    n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
    n04090263 rifle


局部不量化
==================

对于特定网络，部分层由于数据分布差异大，量化成INT8会大幅降低模型精度，使用局部不量化功能，可以一键将部分层之前、之后、之间添加到混精度表中，在生成混精度模型时，这部分层将不被量化。

使用方法
------------------
本章将沿用第三章提到的yolov5s网络的例子，介绍如何使用局部不量化功能，快速生成混精度模型。

生成FP32和INT8模型的过程与第三章相同，下面仅介绍精度测试方案与混精度流程。

对于yolo系列模型来说，最后三个卷积层由于数据分布差异较大，常常手动添加混精度表以提升精度。使用局部不量化功能，从FP32 mlir文件搜索到对应的层。快速添加混精度表。

.. code-block:: shell

   $ fp_forward \
       yolov5s.mlir \
       --quantize INT8 \
       --chip bm1684x \
       --fpfwd_outputs 474_Conv,326_Conv,622_Conv\
       --chip bm1684x \
       -o yolov5s_qtable

点开yolov5s_qtable可以看见相关层都被加入到qtable中。

生成混精度模型

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --quantize_table yolov5s_qtable\
       --chip bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_1684x_mix.bmodel


验证FP32模型和混精度模型的精度
model-zoo中有对目标检测模型进行精度验证的程序yolo，可以在mlir.config.yaml中使用harness字段调用yolo：

相关字段修改如下

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

切换到model-zoo顶层目录，使用tpu_perf.precision_benchmark进行精度测试，命令如下：

.. code-block:: shell

   $ python3 -m tpu_perf.precision_benchmark yolov5s_path --mlir --target BM1684X --devices 0

执行完后，精度测试的结果存放在output/yolo.csv中:

FP32模型mAP为： 37.14%

INT8模型mAP为： 34.70%

混精度模型mAP为： 36.18%

在yolov5以外的检测模型上，使用混精度的方式常会有更明显的效果。


参数说明
------------------
.. list-table:: fp_forward 参数功能
   :widths: 23 8 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - 无
     - 是
     - 指定mlir文件
   * - fpfwd_inputs
     - 否
     - 指定层（包含本层）之前不执行量化，多输入用,间隔
   * - fpfwd_outputs
     - 否
     - 指定层（包含本层）之后不执行量化，多输入用,间隔
   * - fpfwd_blocks
     - 否
     - 指定起点和终点之间的层不执行量化，起点和终点之间用:间隔，多个block之间用空格间隔
   * - chip
     - 是
     - 指定模型将要用到的平台, 支持bm1688/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
   * - fp_type
     - 否
     - 指定混精度使用的float类型, 支持auto,F16,F32,BF16，默认为auto，表示由程序内部自动选择
   * - o
     - 是
     - 输出混精度量化表
