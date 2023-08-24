.. _mix precision:

混精度使用方法
==================

本章以检测网络 ``yolov3 tiny`` 网络模型为例, 介绍如何使用混精度。
该模型来自https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3。

本章需要如下文件(其中xxxx对应实际的版本信息):

**tpu-mlir_xxxx.tar.gz (tpu-mlir的发布包)**

加载tpu-mlir
------------------

.. include:: env_var.rst

准备工作目录
------------------

建立 ``yolov3_tiny`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都
放入 ``yolov3_tiny`` 目录中。

操作如下:

.. code-block:: shell
  :linenos:

   $ mkdir yolov3_tiny && cd yolov3_tiny
   $ wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ mkdir workspace && cd workspace

这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。
注意如果 ``tiny-yolov3-11.onnx`` 用wget下载失败, 请用其他方式下载后放到 ``yolov3_tiny`` 目录。

验证原始模型
----------------

``detect_yolov3.py`` 是已经写好的验证程序, 可以用来对 ``yolov3_tiny`` 网络进行验证。执行过程如下:

.. code-block:: shell

   $ detect_yolov3.py \
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

第二步: 生成calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ run_calibration.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table

第三步: 转对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_int8.bmodel

第四步: 验证模型
~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3.py \
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

使用 ``run_qtable.py`` 生成混精度量化表, 相关参数说明如下:

.. list-table:: run_qtable.py 参数功能
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

   $ run_qtable.py yolov3_tiny.mlir \
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
``run_qtable.py`` 会在每次设置某相邻2层为浮点计算后，接续计算整个网络的输出cos，若该cos大于指定的expected_cos，则退出搜素。因此，若设置更大的expected_cos，会尝试将更多层设为浮点计算


第二步: 生成混精度量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel

第三步: 验证混精度模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   $ detect_yolov3.py \
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
