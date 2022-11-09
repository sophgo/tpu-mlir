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

建立 ``yolov3_tiny`` 目录, 注意是与tpu-mlir同级目录；并把模型文件和图片文件都
放入 ``yolov3_tiny`` 目录中。

操作如下:

.. code-block:: console
  :linenos:

   $ mkdir yolov3_tiny && cd yolov3_tiny
   $ wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ mkdir workspace && cd workspace

这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。

验证原始模型
----------------

``detect_yolov3.py`` 是已经写好的验证程序, 可以用来对 ``yolov3_tiny`` 网络进行验证。执行过程如下:

.. code-block:: console

   $ detect_yolov3.py \
        --model ../tiny-yolov3-11.onnx \
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_onnx.jpg

执行完后打印检测到的结果如下：

.. code-block:: console

    car:81.7%
    car:72.6%
    car:71.1%
    car:66.0%
    bus:69.5%

并得到图片 ``yolov3_onnx.jpg``, 如下( :ref:`yolov3_onnx_result` ):

.. _yolov3_onnx_result:
.. figure:: ../assets/yolov3_onnx.jpg
   :height: 13cm
   :align: center

   yolov3_tiny ONNX执行效果


转成INT8对称量化模型
----------------------

如果前面章节介绍的转模型方法, 这里不做参数说明, 只有操作过程。

第一步: 转成F32 mlir
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ model_transform.py \
       --model_name yolov3_tiny \
       --model_def ../tiny-yolov3-11.onnx \
       --input_shapes [[1,3,416,416]] \
       --scale 0.0039216,0.0039216,0.0039216 \
       --pixel_format rgb \
       --keep_aspect_ratio \
       --pad_value 128 \
       --output_names=transpose_output1,transpose_output \
       --mlir yolov3_tiny.mlir

第二步: 生成calibartion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ run_calibration.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov3_cali_table

第三步: 转对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_int8.bmodel

第四步: 验证模型
~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ detect_yolov3.py \
        --model yolov3_int8.bmodel \
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_int8.jpg

执行完后打印结果为：

.. code-block:: console

  car:79.0%
  car:72.4%
  bus:65.8%

得到图片 ``yolov3_int8.jpg``, 如下( :ref:`yolov3_int8_result` ):

.. _yolov3_int8_result:
.. figure:: ../assets/yolov3_int8.jpg
   :height: 13cm
   :align: center

   yolov3_tiny int8对称量化执行效果

可以看出int8对称量化模型相对原始模型, 有一定损失。

转成混精度量化模型
-----------------------

在转int8对称量化模型的基础上, 执行如下步骤。

第一步: 生成混精度量化表
~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - 输入calibration table文件
   * - chip
     - 是
     - 指定模型将要用到的平台, 支持bm1684x（目前只支持这一种, 后续会支持多款TPU
       平台
   * - input_num
     - 否
     - 指定输入样本数量, 默认用10个
   * - num_layers
     - 否
     - 指定采用浮点计算的op的数量, 默认8个
   * - o
     - 是
     - 输出混精度表

本例中采用默认10张图片校准并8个OP转换成浮点, 执行命令如下:

.. code-block:: console

   $ run_qtable.py yolov3_tiny.mlir \
       --dataset ../COCO2017 \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       -o yolov3_qtable

生成的混精度量化表 ``yolov3_qtable``, 内容如下:

.. code-block:: console

  convolution_output11_Conv F32
  model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu F32
  model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu F32
  model_1/concatenate_1/concat:0_Concat F32
  model_1/leaky_re_lu_10/LeakyRelu:0_LeakyRelu F32
  convolution_output4_Conv F32
  model_1/leaky_re_lu_9/LeakyRelu:0_LeakyRelu F32
  model_1/leaky_re_lu_11/LeakyRelu:0_LeakyRelu F32


该表中, 第一列表示相应的operation, 第二列表示类型。
另外同时也会生成一个loss表文件 ``full_loss_table.txt``, 内容如下:

.. code-block:: console
    :linenos:

    No.0 : Layer: convolution_output11_Conv                     Loss: -15.688898181915283
    No.1 : Layer: model_1/leaky_re_lu_4/LeakyRelu:0_LeakyRelu   Loss: -16.927041554450987
    No.2 : Layer: model_1/leaky_re_lu_2/LeakyRelu:0_LeakyRelu   Loss: -17.018644523620605
    No.3 : Layer: model_1/concatenate_1/concat:0_Concat         Loss: -17.04096896648407
    No.4 : Layer: model_1/leaky_re_lu_10/LeakyRelu:0_LeakyRelu  Loss: -17.053492522239686
    No.5 : Layer: convolution_output4_Conv                      Loss: -17.065047717094423
    No.6 : Layer: model_1/leaky_re_lu_9/LeakyRelu:0_LeakyRelu   Loss: -17.067219734191895
    No.7 : Layer: model_1/leaky_re_lu_11/LeakyRelu:0_LeakyRelu  Loss: -17.072936034202577
    No.8 : Layer: convolution_output1_Conv                      Loss: -17.075703692436218
    No.9 : Layer: model_1/leaky_re_lu_6/LeakyRelu:0_LeakyRelu   Loss: -17.07633330821991
    No.10: Layer: convolution_output3_Conv                      Loss: -17.078122758865355
    No.11: Layer: convolution_output_Conv                       Loss: -17.080725646018983
    No.12: Layer: model_1/leaky_re_lu_8/LeakyRelu:0_LeakyRelu   Loss: -17.08555600643158
    No.13: Layer: model_1/leaky_re_lu_1/LeakyRelu:0_LeakyRelu   Loss: -17.085753679275513
    ......

该表按Loss顺利排列, 可以看出混精度量化表取的该Loss表的前8层。如果业务中8层仍然不满足要求,
可以继续把Loss表逐步添加到混精度表中。

第二步: 生成混精度量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ model_deploy.py \
       --mlir yolov3_tiny.mlir \
       --quantize INT8 \
       --quantize_table yolov3_qtable \
       --calibration_table yolov3_cali_table \
       --chip bm1684x \
       --model yolov3_mix.bmodel

第三步: 验证混精度模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ detect_yolov3.py \
        --model yolov3_mix.bmodel \
        --input ../COCO2017/000000124798.jpg \
        --output yolov3_mix.jpg

执行完后打印结果为：

.. code-block:: console

    car:81.9%
    car:72.4%
    car:64.5%
    bus:66.2%

得到图片yolov3_mix.jpg, 如下( :ref:`yolov3_mix_result` ):

.. _yolov3_mix_result:
.. figure:: ../assets/yolov3_mix.jpg
   :height: 13cm
   :align: center

   yolov3_tiny 混精度对称量化执行效果

可以看出混精度后, 检测结果更接近原始模型的结果。
