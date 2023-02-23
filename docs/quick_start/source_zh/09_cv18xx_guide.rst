CV18XX芯片使用指南
===================

CV18XX支持ONNX系列和Caffe模型,目前不支持TFLite模型。在量化数据类型方面,CV18XX支持BF16格式的量化
和INT8格式的非对称量化。本章节以CV183X芯片为例,介绍CV18XX系列芯片编译模型和运行runtime sample。

编译yolov5模型
------------------

加载tpu-mlir
~~~~~~~~~~~~~~~~~~~~

.. include:: env_var.rst

准备工作目录
~~~~~~~~~~~~~~~~~~~~

建立 ``model_yolov5s`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都
放入 ``model_yolov5s`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s && cd model_yolov5s
   $ cp $TPUC_ROOT/regression/model/yolov5s.onnx .
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。

ONNX转MLIR
~~~~~~~~~~~~~~~~~~~~

如果模型是图片输入, 在转模型之前我们需要了解模型的预处理。如果模型用预处理后的npz文件做输入, 则不需要考虑预处理。
预处理过程用公式表达如下( :math:`x` 代表输入):

.. math::

   y = (x - mean) \times scale


官网yolov5的图片是rgb, 每个值会乘以 ``1/255`` , 转换成mean和scale对应为
``0.0,0.0,0.0`` 和 ``0.0039216,0.0039216,0.0039216`` 。

模型转换命令如下:

.. code-block:: shell

   $ model_transform.py \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 350,498,646 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

``model_transform`` 的相关参数说明参考“编译ONNX模型-ONNX转MLIR”部分。

MLIR转BF16模型
~~~~~~~~~~~~~~~~~~~~

将mlir文件转换成bf16的cvimodel, 操作方法如下:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize BF16 \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model yolov5s_cv183x_bf16.cvimodel

``model_deploy.py`` 的相关参数说明参考“编译ONNX模型-MLIR转F32模型”部分。

MLIR转INT8模型
~~~~~~~~~~~~~~~~~~~~
转INT8模型前需要跑calibration, 得到校准表; 输入数据的数量根据情况准备100~1000张左右。然后用校准表, 生成INT8对称cvimodel

这里用现有的100张来自COCO2017的图片举例, 执行calibration:

.. code-block:: shell

   $ run_calibration.py yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

运行完成后会生成名为 ``${model_name}_cali_table`` 的文件, 该文件用于后续编译INT8
模型的输入文件。

转成INT8对称量化cvimodel模型, 执行如下命令:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_cv183x_int8_sym.cvimodel

编译完成后, 会生成名为 ``${model_name}_cv183x_int8_sym.cvimodel`` 的文件。


效果对比
~~~~~~~~~~~~~~~~~~~~

onnx模型的执行方式如下, 得到 ``dog_onnx.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model ../yolov5s.onnx \
       --output dog_onnx.jpg

FP32 mlir模型的执行方式如下,得到 ``dog_mlir.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s.mlir \
       --output dog_mlir.jpg

BF16 cvimodel的执行方式如下, 得到 ``dog_bf16.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_bf16.cvimodel \
       --output dog_bf16.jpg

INT8 cvimodel的执行方式如下, 得到 ``dog_int8.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_int8_sym.cvimodel \
       --output dog_int8.jpg


四张图片对比如 :numref:`yolov5s_result1` ，由于运行环境不同, 最终的效果和精度与 :numref:`yolov5s_result1` 会有些差异。

.. _yolov5s_result1:
.. figure:: ../assets/yolov5s_cvi.jpg
   :height: 13cm
   :align: center

   不同模型效果对比



上述教程介绍了TPU-MLIR编译CV18XX系列芯片的ONNX模型的过程,caffe模型的转换过程可参考“编译Caffe模型”章节,只需要将对应的芯片名称换成实际的CV18XX芯片名称即可。


合并cvimodel模型文件
--------------------------
待补充


编译和运行runtime sample
--------------------------
待补充


