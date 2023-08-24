.. _sensitive layer:

敏感层搜索使用方法
==================

本章以检测网络 ``mobilenet-v2`` 网络模型为例, 介绍如何使用敏感层搜索。
该模型来自nnmodels/pytorch_models/accuracy_test/classification/mobilenet_v2.pt。

本章需要如下文件(其中xxxx对应实际的版本信息):

**tpu-mlir_xxxx.tar.gz (tpu-mlir的发布包)**

加载tpu-mlir
------------------

.. include:: env_var.rst

准备工作目录
------------------

建立 ``mobilenet-v2`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都放入 ``mobilenet-v2`` 目录中。

操作如下:

.. code-block:: shell
  :linenos:

   $ mkdir mobilenet-v2 && cd mobilenet-v2
   $ cp -rf $TPUC_ROOT/regression/dataset/ILSVRC2012 .
   $ mkdir workspace && cd workspace

这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。
注意 ``mobilenet-v2.pt`` 需要自己从nnmodels下载后放到 ``mobilenet-v2`` 目录。

测试Float和INT8对称量化模型分类效果
---------------------------------

如前面章节介绍的转模型方法, 这里不做参数说明, 只有操作过程。

第一步: 转成FP32 mlir
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

第二步: 生成calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

第三步: 转FP32 bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --chip bm1684 \
       --model mobilenet_v2_bm1684_f32.bmodel

第四步: 转对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table mobilenet_v2_cali_table \
       --model mobilenet_v2_bm1684_int8_sym.bmodel

第五步: 验证FP32模型和INT8对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classify_mobilenet_v2.py是已经写好的验证程序，可以用来对mobilenet_v2网络进行验证。执行过程如下，FP32模型：

.. code-block:: shell

   $ classify_mobilenet_v2.py \
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

   $ classify_mobilenet_v2.py \
       --model_def mobilenet_v2_bm1684_int8_sym.bmodel \
       --input ../ILSVRC2012/n01440764_9572.JPEG \
       --output mobilenet_v2_INT8_sym_bmodel.JPEG \
       --category_file ../ILSVRC2012/synset_words.txt

在输出结果图片上可以看到如下分类信息，正确结果tench排在第二名：

.. code-block:: shell

    Top-5
    n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis
    n01440764 tench, Tinca tinca
    n01871265 tusker
    n02396427 wild boar, boar, Sus scrofa
    n02074367 dugong, Dugong dugon

转成混精度量化模型
-----------------------

在转int8对称量化模型的基础上, 执行如下步骤。

第一步: 进行敏感层搜索
~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``run_sensitive_layer.py`` 搜索损失较大的layer，注意尽量使用bad cases进行敏感层搜索，相关参数说明如下:

.. list-table:: run_sensitive_layer.py 参数功能
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
     - 指定模型将要用到的平台, 支持bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
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

   $ run_sensitive_layer.py mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       --inference_num 30 \
       --calibration_table mobilenet_cali_table \
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

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --chip bm1684 \
       --calibration_table new_cali_table.txt \
       --quantize_table mobilenet_v2_qtable \
       --model mobilenet_v2_bm1684_mix.bmodel

第三步: 验证混精度模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ classify_mobilenet_v2.py \
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
