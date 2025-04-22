附录01：从 NNTC 迁移至 TPU-MLIR
===============================

NNTC 所使用 Docker 版本为 sophgo/tpuc_dev:v2.1, MLIR 使用的版本及环境初始化请参考
:ref:`开发环境配置 <docker configuration>` 。

下面将以 yolov5s 为例, 讲解NNTC和TPU-MLIR在量化方面的异同, 浮点模型编译方面可以直接
参考《TPU-MLIR快速入门指南》的“编译ONNX模型”章节内容，以下内容假设已经按照
《TPU-MLIR快速入门指南》中描述准备好了yolov5s模型。


ONNX模型导入
------------

在TPU-MLIR中要对模型进行量化首先要把原始模型转为top层的mlir文件, 这一步可以类比为NNTC中分步量化生成fp32umodel的过程。

#. TPU-MLIR的模型转换命令

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

   TPU-MLIR可以直接把图片预处理编码到转换出的mlir文件中。


#. NNTC的模型转换命令

   .. code-block:: shell

      $ python3 -m ufw.tools.on_to_umodel \
           -m ../yolov5s.onnx \
           -s '(1,3,640,640)' \
           -d 'compilation' \
           --cmp

   NNTC导入模型的时候不能指定预处理方式。


制作量化校准表
--------------

想要生成定点模型都需要经过量化工具对模型进行量化, nntc中分步量化这里使用的是 calibration_use_pb, mlir使用的是run_calibration.py

输入数据的数量根据情况准备100~1000张左右,用现有的100张来自COCO2017的图片举例, 执行calibration:

在nntc中使用分步量化还需要自行使用图片量化数据集制作lmdb量化数据集,并且修改fp32_protoxt,将数据输入指向lmdb文件

.. note::

   关于NNTC量化数据集制作方式可以参考《TPU-NNTC开发参考手册》的“模型量化”章节内
   容,且注意该lmdb数据集与TPU-MLIR并不兼容。TPU-MLIR可以直接使用原始图片作为量化
   工具输入。如果是语音、文字等非图片数据,需要将其转化为npz文件。


#. MLIR 量化模型

   .. code-block:: shell

      $ run_calibration.py yolov5s.mlir \
          --dataset ../COCO2017 \
          --input_num 100 \
          -o yolov5s_cali_table

   经过量化之后会得到量化表yolov5s_cali_table。


#. NNTC 量化模型

   .. code-block:: shell

       $ calibration_use_pb quantize \
            --model=./compilation/yolov5s_bmneto_test_fp32.prototxt \
            --weights=./compilation/yolov5s_bmneto.fp32umodel \
            -save_test_proto=True --bitwidth=TO_INT8

   在nntc中,量化之后得到的是int8umodel以及prototxt。


生成int8模型
------------

转成INT8对称量化模型, 执行如下命令:

#. MLIR:

   .. code-block:: shell

      $ model_deploy.py \
          --mlir yolov5s.mlir \
          --quantize INT8 \
          --calibration_table yolov5s_cali_table \
          --processor bm1684 \
          --test_input yolov5s_in_f32.npz \
          --test_reference yolov5s_top_outputs.npz \
          --tolerance 0.85,0.45 \
          --model yolov5s_1684_int8_sym.bmodel

   运行结束之后得到yolov5s_1684_int8_sym.bmodel。

#. NNTC:

   在NNTC中,则是使用int8umodel以及prototxt使用bmnetu工具生成int8的bmodel。

   .. code-block:: shell

      $ bmnetu --model=./compilation/yolov5s_bmneto_deploy_int8_unique_top.prototxt \
          --weight=./compilation/yolov5s_bmneto.int8umodel

   运行结束之后得到compilation.bmodel。
