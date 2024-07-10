编译Caffe模型
=============

本章以 ``mobilenet_v2_deploy.prototxt`` 和 ``mobilenet_v2.caffemodel`` 为例, 介绍如何编译迁移一个caffe模型至 BM1684X 平台运行。

本章需要安装TPU-MLIR。


安装TPU-MLIR
------------------

进入Docker容器，并执行以下命令安装TPU-MLIR：

.. code-block:: shell

   $ pip install tpu_mlir[caffe]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[caffe]


准备工作目录
------------------

.. include:: get_resource.rst

建立 ``mobilenet_v2`` 目录, 并把模型文件和图片文件都放入 ``mobilenet_v2`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir mobilenet_v2 && cd mobilenet_v2
   $ wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt
   $ wget https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
   $ cp -rf tpu_mlir_resource/dataset/ILSVRC2012 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


Caffe转MLIR
------------------

本例中的模型是 `BGR` 输入, mean和scale分别为 ``103.94,116.78,123.68`` 和 ``0.017,0.017,0.017`` 。

模型转换命令如下:


.. code-block:: shell

   $ model_transform \
       --model_name mobilenet_v2 \
       --model_def ../mobilenet_v2_deploy.prototxt \
       --model_data ../mobilenet_v2.caffemodel \
       --input_shapes [[1,3,224,224]] \
       --resize_dims=256,256 \
       --mean 103.94,116.78,123.68 \
       --scale 0.017,0.017,0.017 \
       --pixel_format bgr \
       --test_input ../image/cat.jpg \
       --test_result mobilenet_v2_top_outputs.npz \
       --mlir mobilenet_v2.mlir

转成mlir文件后, 会生成一个 ``${model_name}_in_f32.npz`` 文件, 该文件是模型的输入文件。


MLIR转F32模型
------------------

将mlir文件转换成f32的bmodel, 操作方法如下:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --processor bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --model mobilenet_v2_1684x_f32.bmodel

编译完成后, 会生成名为 ``${model_name}_1684x_f32.bmodel`` 的文件。


MLIR转INT8模型
------------------

生成校准表
~~~~~~~~~~~~~~~~~~~~

转INT8模型前需要跑calibration, 得到校准表; 输入数据的数量根据情况准备100~1000张左右。

然后用校准表, 生成对称或非对称bmodel。如果对称符合需求, 一般不建议用非对称, 因为非对称的性能会略差于对称模型。

这里用现有的100张来自ILSVRC2012的图片举例, 执行calibration:


.. code-block:: shell

   $ run_calibration mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

运行完成后会生成名为 ``${model_name}_cali_table`` 的文件, 该文件用于后续编译INT8
模型的输入文件。


编译为INT8对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~

转成INT8对称量化模型, 执行如下命令:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --model mobilenet_v2_1684x_int8.bmodel

编译完成后, 会生成名为 ``${model_name}_1684x_int8.bmodel`` 的文件。
