编译Caffe模型
=============

本章以 ``mobilenet_v2_deploy.prototxt`` 和 ``mobilenet_v2.caffemodel`` 为例，介绍如何编译迁移一个caffe模型至BM1684x TPU平台运行。

本章需要如下文件(其中xxxx对应实际的版本信息)：

**tpu-mlir_xxxx.tar.gz (tpu-mlir的发布包)**

加载tpu-mlir
------------------

.. include:: env_var.rst


准备工作目录
------------------

建立 ``mobilenet_v2`` 目录，注意是与tpu-mlir同级目录；并把模型文件和图片文件都
放入 ``mobilenet_v2`` 目录中。


操作如下：

.. code-block:: console
   :linenos:

   $ mkdir mobilenet_v2 && cd mobilenet_v2
   $ cp $TPUC_ROOT/regression/model/mobilenet_v2_deploy.prototxt .
   $ cp $TPUC_ROOT/regression/model/mobilenet_v2.caffemodel .
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


这里的 ``$TPUC_ROOT`` 是环境变量，对应tpu-mlir_xxxx目录。


Caffe转MLIR
------------------

本例中的模型是 `BGR` 输入，mean和scale分别为 ``103.94,116.78,123.68`` 和 ``0.017,0.017,0.017``。

模型转换命令如下：


.. code-block:: console

   $ model_transform.py \
       --model_name mobilenet_v2 \
       --model_def ../mobilenet_v2_deploy.prototxt \
       --model_data ../mobilenet_v2.caffemodel \
       --input_shapes [[1,3,224,224]] \
       --mean 103.94,116.78,123.68 \
       --scale 0.017,0.017,0.017 \
       --keep_aspect_ratio \
       --pixel_format bgr \
       --test_input ../image/cat.jpg \
       --test_result mobilenet_v2_top_outputs.npz \
       --mlir mobilenet_v2.mlir

转成mlir文件后，会生成一个 ``${model_name}_in_f32.npz`` 文件，该文件是模型的输入文件。


MLIR转F32模型
------------------

将mlir文件转换成f32的bmodel，操作方法如下：

.. code-block:: console

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --chip bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model mobilenet_v2_1684x_f32.bmodel

编译完成后，会生成名为 ``${model_name}_1684x_f32.bmodel`` 的文件。


MLIR转INT8模型
------------------

生成量化表
~~~~~~~~~~~~~~~~~~~~

转INT8模型前需要跑calibration，得到量化表；输入数据的数量根据情况准备100~1000张左右。

然后用量化表，生成对称或非对称bmodel。如果对称符合需求，一般不建议用非对称，因为
非对称的性能会略差于对称模型。

这里用现有的100张来自COCO2017的图片举例，执行calibration：


.. code-block:: console

   $ run_calibration.py mobilenet_v2.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

运行完成后会生成名为 ``${model_name}_cali_table`` 的文件，该文件用于后续编译INT8
模型的输入文件。


编译为INT8对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~

转成INT8对称量化模型，执行如下命令：

.. code-block:: console

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --chip bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --model mobilenet_v2_1684x_int8_sym.bmodel

编译完成后，会生成名为 ``${model_name}_1684x_int8_sym.bmodel`` 的文件。


编译为INT8非对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~

转成INT8非对称量化模型，执行如下命令：

.. code-block:: console

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --asymmetric \
       --calibration_table mobilenet_v2_cali_table \
       --chip bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.95,0.69 \
       --model mobilenet_v2_1684x_int8_asym.bmodel


编译完成后，会生成名为 ``${model_name}_1684x_int8_asym.bmodel`` 的文件。
