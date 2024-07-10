编译TORCH模型
=============

本章以 ``yolov5s.pt`` 为例，介绍如何编译迁移一个pytorch模型至BM1684X 平台运行。

本章需要安装TPU-MLIR。


安装TPU-MLIR
------------------

进入Docker容器，并执行以下命令安装TPU-MLIR：

.. code-block:: shell

   $ pip install tpu_mlir[torch]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[torch]


准备工作目录
------------------

.. include:: get_resource.rst

建立 ``model_yolov5s_pt`` 目录，并把模型文件和图片文件都放入 ``model_yolov5s_pt`` 目录中。

操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s_pt && cd model_yolov5s_pt
   $ wget -O yolov5s.pt "https://github.com/sophgo/tpu-mlir/raw/master/regression/model/yolov5s.pt"
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


TORCH转MLIR
------------------

本例中的模型是 `RGB` 输入，mean和scale分别为 ``0.0,0.0,0.0`` 和 ``0.0039216,0.0039216,0.0039216``。


模型转换命令如下:


.. code-block:: shell

   $ model_transform \
       --model_name yolov5s_pt \
       --model_def ../yolov5s.pt \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_pt_top_outputs.npz \
       --mlir yolov5s_pt.mlir


转成mlir文件后，会生成一个 ``${model_name}_in_f32.npz`` 文件，该文件是模型的输入文件。值得注意的是，目前仅支持静态模型，模型在编译前需要调用 ``torch.jit.trace()`` 以生成静态模型。


MLIR转F16模型
------------------

将mlir文件转换成f16的bmodel，操作方法如下:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s_pt.mlir \
       --quantize F16 \
       --processor bm1684x \
       --test_input yolov5s_pt_in_f32.npz \
       --test_reference yolov5s_pt_top_outputs.npz \
       --model yolov5s_pt_1684x_f16.bmodel


编译完成后，会生成名为 ``yolov5s_pt_1684x_f16.bmodel`` 的文件。


MLIR转INT8模型
------------------

生成校准表
~~~~~~~~~~~~~~~~~~~~

转INT8模型前需要跑calibration，得到校准表; 这里用现有的100张来自COCO2017的图片举例，执行calibration:


.. code-block:: shell

   $ run_calibration yolov5s_pt.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_pt_cali_table

运行完成后会生成名为 ``yolov5s_pt_cali_table`` 的文件，该文件用于后续编译INT8
模型的输入文件。


编译为INT8对称量化模型
~~~~~~~~~~~~~~~~~~~~~~~~

转成INT8对称量化模型，执行如下命令:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s_pt.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_pt_cali_table \
       --processor bm1684x \
       --test_input yolov5s_pt_in_f32.npz \
       --test_reference yolov5s_pt_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_pt_1684x_int8_sym.bmodel

编译完成后，会生成名为 ``yolov5s_pt_1684x_int8_sym.bmodel`` 的文件。


效果对比
------------------

利用 ``detect_yolov5`` 命令，对图片进行目标检测。

用以下代码分别来验证pytorch/f16/int8的执行结果。


pytorch模型的执行方式如下，得到 ``dog_torch.jpg`` :

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model ../yolov5s.pt \
       --output dog_torch.jpg


f16 bmodel的执行方式如下，得到 ``dog_f16.jpg`` :

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_pt_1684x_f16.bmodel \
       --output dog_f16.jpg



int8对称bmodel的执行方式如下，得到 ``dog_int8_sym.jpg`` :

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_pt_1684x_int8_sym.bmodel \
       --output dog_int8_sym.jpg


对比结果如下:

.. _yolov5s_pt_result:
.. figure:: ../assets/yolov5s_pt.png
   :height: 13cm
   :align: center

   TPU-MLIR对YOLOv5s编译效果对比

由于运行环境不同，最终的效果和精度与 :numref:`yolov5s_pt_result` 会有些差异。
