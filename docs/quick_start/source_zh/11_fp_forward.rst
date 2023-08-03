.. _fp_forward:

局部不量化
==================
对于特定网络，部分层由于数据分布差异大，量化成INT8会大幅降低模型精度，使用局部不量化功能，可以一键将部分层之前、之后、之间添加到混精度表中，在生成混精度模型时，这部分层将不被量化。

使用方法
------------------
本章将沿用第三章提到的yolov5s网络的例子，介绍如何使用局部不量化功能，快速生成混精度模型。

生成FP32和INT8模型的过程与第三章相同，下面仅介绍精度测试方案与混精度流程。

对于yolo系列模型来说，最后三个卷积层由于数据分布差异较大，常常手动添加混精度表以提升精度。使用局部不量化功能，从FP32 mlir文件搜索到对应的层。快速添加混精度表。

.. code-block:: shell

   $ fp_forward.py \
       yolov5s.mlir \
       --quantize INT8 \
       --chip bm1684x \
       --fpfwd_outputs 474_Conv,326_Conv,622_Conv\
       --chip bm1684x \
       -o yolov5s_qtable

点开yolov5s_qtable可以看见相关层都被加入到qtable中。

生成混精度模型

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --quantize_table yolov5s_qtable\
       --chip bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_1684x_mix.bmodel
 

验证FP32模型和混精度模型的精度
model-zoo中有对目标检测模型进行精度验证的程序yolo，可以在mlir.config.yaml中使用harness字段调用yolo：

相关字段修改如下

.. code-block:: shell

    $ dataset:
        imagedir: $(coco2017_val_set)
        anno: $(coco2017_anno)/instances_val2017.json

      harness:
        type: yolo
        args:
          - name: FP32
            bmodel: $(workdir)/$(name)_bm1684_f32.bmodel
          - name: INT8
            bmodel: $(workdir)/$(name)_bm1684_int8_sym.bmodel
          - name: mix
            bmodel: $(workdir)/$(name)_bm1684_mix.bmodel

切换到model-zoo顶层目录，使用tpu_perf.precision_benchmark进行精度测试，命令如下：

.. code-block:: shell

   $ python3 -m tpu_perf.precision_benchmark yolov5s_path --mlir --target BM1684X --devices 0

执行完后，精度测试的结果存放在output/yolo.csv中:

FP32模型mAP为： 37.14%

INT8模型mAP为： 34.70%

混精度模型mAP为： 36.18%

在yolov5以外的检测模型上，使用混精度的方式常会有更明显的效果。


参数说明
------------------
.. list-table:: fp_forward.py 参数功能
   :widths: 23 8 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - 无
     - 是
     - 指定mlir文件
   * - fpfwd_inputs
     - 否
     - 指定层（包含本层）之前不执行量化，多输入用,间隔
   * - fpfwd_outputs
     - 否
     - 指定层（包含本层）之后不执行量化，多输入用,间隔
   * - fpfwd_blocks
     - 否
     - 指定起点和终点之间的层不执行量化，起点和终点之间用:间隔，多个block之间用空格间隔
   * - chip
     - 是
     - 指定模型将要用到的平台, 支持bm1686/bm1684x/bm1684/cv186x/cv183x/cv182x/cv181x/cv180x
   * - fp_type
     - 否
     - 指定混精度使用的float类型, 支持auto,F16,F32,BF16，默认为auto，表示由程序内部自动选择
   * - o
     - 是
     - 输出混精度量化表
