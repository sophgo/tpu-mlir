从 NNTC 迁移至 tpu-mlir
==================

NNTC 所使用 docker 版本为 sophgo/tpuc_dev:v2.1, MLIR 使用的docker版本参考及环境初始化请参考

.. include:: env_var.rst

下面将以 yolov5s 为例, 讲解nntc和mlir在量化方面的异同, 浮点模型编译方面可以直接参考:ref:`编译ONNX模型`。

首先参考 :ref:`编译ONNX模型` 的章节准备yolov5s模型。

ONNX转MLIR
------------

在mlir中要对模型进行量化首先要把原始模型转为top层的mlir文件, 这一步可以类比为nntc中分步量化中的生成fp32umodel

MLIR的模型转换命令:

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


NNTC的模型转换命令:

.. code-block:: shell

    $ python3 -m ufw.tools.on_to_umodel \
        -m ../yolov5s.onnx \
        -s '(1,3,640,640)' \
        -d 'compilation' \
        --cmp


制作量化校准表
------------

想要生成定点模型都需要经过量化工具对模型进行量化, nntc中分步量化这里使用的是 calibration_use_pb, mlir使用的是run_calibration.py

输入数据的数量根据情况准备100~1000张左右,用现有的100张来自COCO2017的图片举例, 执行calibration:

在nntc中使用分步量化还需要自行使用图片量化数据集制作mdb量化数据集,并且修改fp32_protoxt,将数据输入指向lmdb文件

注意:
关于NNTC量化数据集制作方式可以参考《TPU-NNTC开发参考手册》的“模型量化”章节内容,且注意该lmdb数据集与TPU-MLIR并不兼容。
TPU-MLIR可以直接使用原始图片作为量化工具输入。如果是语音、文字等非图片数据,需要将其转化为npz文件。

MLIR 量化模型

.. code-block:: shell

   $ run_calibration.py yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

经过量化之后会得到量化表yolov5s_cali_table

NNTC 量化模型

.. code-block:: shell

    $ calibration_use_pb quantize \
         --model=./compilation/yolov5s_bmneto_test_fp32.prototxt \
         --weights=./compilation/yolov5s_bmneto.fp32umodel \
         -save_test_proto=True --bitwidth=TO_INT8

在nntc中,量化之后得到的是int8umodel以及prototxt

值得一提的是, mlir还有run_qtable工具帮助生成混精度模型

生成int8模型
-------------

转成INT8对称量化模型, 执行如下命令:

MLIR:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --chip bm1684 \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_1684_int8_sym.bmodel

运行结束之后得到yolov5s_1684_int8_sym.bmodel。

在nntc中,则是使用int8umodel以及prototxt使用bmnetu工具生成int8的bmodel

   .. code-block:: shell

      $ bmnetu --model=./compilation/yolov5s_bmneto_deploy_int8_unique_top.prototxt \
          --weight=./compilation/yolov5s_bmneto.int8umodel

运行结束之后得到compilation.bmodel。
