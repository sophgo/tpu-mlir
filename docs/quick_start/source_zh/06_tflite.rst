编译TFLite模型
================

本章以 ``lite-model_mobilebert_int8_1.tflite`` 模型为例, 介绍如何编译迁移一个TFLite模型至BM1684X TPU平台运行。

本章需要如下文件(其中xxxx对应实际的版本信息):

**tpu-mlir_xxxx.tar.gz (tpu-mlir的发布包)**

加载tpu-mlir
------------------

.. include:: env_var.rst


准备工作目录
------------------

建立 ``mobilebert_tf`` 目录, 注意是与tpu-mlir同级目录; 并把测试图片文件放入
``mobilebert_tf`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir mobilebert_tf && cd mobilebert_tf
   $ wget -O lite-model_mobilebert_int8_1.tflite https://storage.googleapis.com/tfhub-lite-models/iree/lite-model/mobilebert/int8/1.tflite
   $ cp ${REGRESSION_PATH}/npz_input/squad_data.npz .
   $ mkdir workspace && cd workspace


这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。


TFLite转MLIR
------------------

模型转换命令如下:


.. code-block:: shell

    $ model_transform.py \
        --model_name mobilebert_tf \
        --mlir mobilebert_tf.mlir \
        --model_def ../lite-model_mobilebert_int8_1.tflite \
        --test_input ../squad_data.npz \
        --test_result mobilebert_tf_top_outputs.npz \
        --input_shapes [[1,384],[1,384],[1,384]] \
        --channel_format none


转成mlir文件后, 会生成一个 ``mobilebert_tf_in_f32.npz`` 文件, 该文件是模型的输入文件。


MLIR转模型
------------------

该模型是tflite int8模型, 可以按如下参数转成模型:

.. code-block:: shell

    $ model_deploy.py \
        --mlir mobilebert_tf.mlir \
        --quantize INT8 \
        --chip bm1684x \
        --test_input mobilebert_tf_in_f32.npz \
        --test_reference mobilebert_tf_top_outputs.npz \
        --model mobilebert_tf_bm1684x_int8.bmodel


编译完成后, 会生成名为 ``mobilebert_tf_bm1684x_int8.bmodel`` 的文件。
