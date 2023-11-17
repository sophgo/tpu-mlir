附录03: BM168x使用指南
===============================

BM168x支持ONNX系列、pytorch模型、Caffe模型和TFLite模型。本章节以BM1684x为例,介绍BM168x系列bmodel文件的合并方法。


.. _merge weight:

合并bmodel模型文件
--------------------------

对于同一个模型,可以依据输入的batch size以及分辨率(不同的h和w)分别生成独立的bmodel文件。不过为了节省外存和运存,可以选择将这些相关的bmodel文件合并为一个bmodel文件,共享其权重部分。具体步骤如下:

步骤0: 生成batch 1的bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

请参考前述章节,新建workspace目录,通过model_transform将yolov5s转换成mlir fp32模型。

.. admonition:: 注意 :
  :class: attention

  1.需要合并的bmodel使用同一个workspace目录,并且不要与不需要合并的bmodel
  共用一个workspace;

  2.步骤0、步骤1中 --merge_weight是必需选项。


.. code-block:: shell

   $ model_transform \
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
       --mlir yolov5s_bs1.mlir

使用前述章节生成的yolov5s_cali_table;如果没有,则通过run_calibration工具对yolov5s.mlir进行量化校验获得calibration table文件。
然后将模型量化并生成bmodel:

.. code-block:: shell

  # 加上 --merge_weight参数
   $ model_deploy \
       --mlir yolov5s_bs1.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --processor bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --merge_weight \
       --model yolov5s_bm1684x_int8_sym_bs1.bmodel

步骤1: 生成batch 2的bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

同步骤0,在同一个workspace中生成batch为2的mlir fp32文件:

.. code-block:: shell

   $ model_transform \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[2,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 350,498,646 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s_bs2.mlir

.. code-block:: shell

  # 加上 --merge_weight参数
   $ model_deploy \
       --mlir yolov5s_bs2.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --processor bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --merge_weight \
       --model yolov5s_bm1684x_int8_sym_bs2.bmodel

步骤2: 合并batch 1和batch 2的bmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用model_tool合并两个bmodel文件:

.. code-block:: shell

  model_tool \
    --combine \
      yolov5s_bm1684x_int8_sym_bs1.bmodel \
      yolov5s_bm1684x_int8_sym_bs2.bmodel \
      -o yolov5s_bm1684x_int8_sym_bs1_bs2.bmodel


综述: 合并过程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用上面命令,不论是相同模型还是不同模型,均可以进行合并。
合并的原理是: 模型生成过程中,会叠加前面模型的weight(如果相同则共用)。

主要步骤在于:

1. 用model_deploy生成模型时,加上--merge_weight参数
2. 要合并的模型的生成目录必须是同一个,且在合并模型前不要清理任何中间文件(叠加前面模型weight通过中间文件_weight_map.csv实现)
3. 用model_tool --combine 将多个bmodel合并

