编译TFLite模型
================

本章以 ``lite-model_mobilebert_int8_1.tflite`` 模型为例, 介绍如何编译迁移一个TFLite模型至BM1684X TPU平台运行。

本章需要安装tpu_mlir。


安装tpu-mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[tensorflow]


准备工作目录
------------------

建立 ``mobilebert_tf`` 目录, 注意是与tpu-mlir同级目录; 并把测试图片文件放入
``mobilebert_tf`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir mobilebert_tf && cd mobilebert_tf
   $ wget -O lite-model_mobilebert_int8_1.tflite https://storage.googleapis.com/tfhub-lite-models/iree/lite-model/mobilebert/int8/1.tflite
   $ tpu_mlir_get_resource regression/npz_input/squad_data.npz .
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

TFLite转MLIR
------------------

模型转换命令如下:


.. code-block:: shell

    $ model_transform \
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

    $ model_deploy \
        --mlir mobilebert_tf.mlir \
        --quantize INT8 \
        --chip bm1684x \
        --test_input mobilebert_tf_in_f32.npz \
        --test_reference mobilebert_tf_top_outputs.npz \
        --model mobilebert_tf_bm1684x_int8.bmodel


编译完成后, 会生成名为 ``mobilebert_tf_bm1684x_int8.bmodel`` 的文件。
