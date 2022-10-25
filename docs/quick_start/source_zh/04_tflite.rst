编译TFLite模型
================

本章以 ``resnet50_int8.tflite`` 模型为例，介绍如何编译迁移一个TFLite模型至BM1684x TPU平台运行。

本章需要如下文件(其中xxxx对应实际的版本信息)：

**tpu-mlir_xxxx.tar.gz (tpu-mlir的发布包)**

加载tpu-mlir
------------------

.. include:: env_var.rst


准备工作目录
------------------

建立 ``model_resnet50_tf`` 目录，注意是与tpu-mlir同级目录；并把测试图片文件放入
``model_resnet50_tf`` 目录中。


操作如下：

.. code-block:: console
   :linenos:

   $ mkdir model_resnet50_tf && cd model_resnet50_tf
   $ cp $TPUC_ROOT/regression/model/resnet50_int8.tflite .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


这里的 ``$TPUC_ROOT`` 是环境变量，对应tpu-mlir_xxxx目录。


TFLite转MLIR
------------------

本例中的模型是bgr输入，mean为 ``103.939,116.779,123.68``，scale为 ``1.0,1.0,1.0``

模型转换命令如下：


.. code-block:: console

    $ model_transform.py \
        --model_name resnet50_tf \
        --model_def  ../resnet50_int8.tflite \
        --input_shapes [[1,3,224,224]] \
        --mean 103.939,116.779,123.68 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input ../image/cat.jpg \
        --test_result resnet50_tf_top_outputs.npz \
        --mlir resnet50_tf.mlir


转成mlir文件后，会生成一个 ``resnet50_tf_in_f32.npz`` 文件，该文件是模型的输入文件。


MLIR转模型
------------------

该模型是tflite非对称量化模型，可以按如下参数转成模型：

.. code-block:: console

   $ model_deploy.py \
       --mlir resnet50_tf.mlir \
       --quantize INT8 \
       --asymmetric \
       --chip bm1684x \
       --test_input resnet50_tf_in_f32.npz \
       --test_reference resnet50_tf_top_outputs.npz \
       --model resnet50_tf_1684x.bmodel


编译完成后，会生成名为 ``resnet50_tf_1684x.bmodel`` 的文件。
