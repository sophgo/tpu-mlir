精度验证
============

整体介绍
--------

验证对象
~~~~~~~~~~~~
TPU-MLIR中的精度验证主要针对mlir模型, fp32采用top层的mlir模型进行精度验证, 而int8对称与非对称量化模式则采用tpu层的mlir模型。

评估指标
~~~~~~~~~~~~
当前主要用于测试的网络有分类网络与目标检测网络，分类网络的精度指标采用Top-1与Top-5准确率，而目标检测网络采用COCO的12个评估指标，如下所示。通常记录精度时采用IoU=0.5时的Average Precision（即PASCAL VOC metric）。

.. math::

   \boldsymbol{Average Precision (AP):} & \\
   AP\quad & \text{\% AP at IoU=.50:.05:.95 (primary challenge metric)} \\
   AP^{IoU}=.50\quad & \text{\% AP at IoU=.50 (PASCAL VOC metric)} \\
   AP^{IoU}=.75\quad & \text{\% AP at IoU=.75 (strict metric)} \\
   \boldsymbol{AP Across Scales:} & \\
   AP^{small}\quad & \text{\% AP for small objects: $area < 32^2$} \\
   AP^{medium}\quad & \text{\% AP for medium objects: $32^2 < area < 96^2$} \\
   AP^{large}\quad & \text{\% AP for large objects: $area > 96^2$} \\
   \boldsymbol{Average Recall (AR):} & \\
   AR^{max=1}\quad & \text{\% AR given 1 detection per image} \\
   AR^{max=10}\quad & \text{\% AR given 10 detections per image} \\
   AR^{max=100}\quad & \text{\% AR given 100 detections per image} \\
   \boldsymbol{AP Across Scales:} & \\
   AP^{small}\quad & \text{\% AP for small objects: $area < 32^2$} \\
   AP^{medium}\quad & \text{\% AP for medium objects: $32^2 < area < 96^2$} \\
   AP^{large}\quad & \text{\% AP for large objects: $area > 96^2$}

数据集
~~~~~~~~~~~~
另外, 验证时使用的数据集需要自行下载, 分类网络使用ILSVRC2012的验证集(共50000张图片, https://www.image-net.org/challenges/LSVRC/2012/)。数据集中的图片有两种摆放方式, 一种是数据集目录下有1000个子目录, 对应1000个类别, 每个子目录下有50张该类别的图片, 该情况下无需标签文件; 另外一种是所有图片均在同一个数据集目录下, 有一个额外的txt标签文件, 按照图片编号顺序每行用1-1000的数字表示每一张图片的类别。

目标检测网络使用COCO2017验证集(共5000张图片, https://cocodataset.org/#download), 所有图片均在同一数据集目录下, 另外还需要下载与该数据集对应的标签文件.json。

精度验证接口
------------

TPU-MLIR的精度验证命令参考如下:

.. code-block:: shell

    $ model_eval.py \
        --model_file mobilenet_v2.mlir \
        --count 50 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

所支持的参数如下:

.. list-table:: model_eval.py 参数功能
   :widths: 20 9 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - model_file
     - 是
     - 指定模型文件
   * - dataset
     - 否
     - 数据集目录
   * - dataset_type
     - 否
     - 数据集类型, 当前主要支持imagenet, coco, 默认为imagenet
   * - postprocess_type
     - 是
     - 精度评估方式, 当前支持topx和coco_mAP
   * - label_file
     - 否
     - txt标签文件, 在验证分类网络精度时可能需要
   * - coco_annotation
     - 否
     - json标签文件, 在验证目标检测网络时需要
   * - count
     - 否
     - 用来验证精度的图片数量, 默认使用整个数据集


精度验证样例
------------
本节以mobilenet_v2和yolov5s分别作为分类网络与目标检测网络的代表进行精度验证。

mobilenet_v2
~~~~~~~~~~~~~
1. 数据集下载

   下载ILSVRC2012验证集到datasets/ILSVRC2012_img_val_with_subdir目录下, 数据集的图片采用带有子目录的摆放方式, 因此不需要额外的标签文件。

2. 模型转换

   使用model_transform.py接口将原模型转换为mobilenet_v2.mlir模型, 并通过run_calibration.py接口获得mobilenet_v2_cali_table。具体使用方法请参照“用户界面”章节。tpu层的INT8模型则通过下方的命令获得，运行完命令后会获得一个名为mobilenet_v2_bm1684x_int8_sym_tpu.mlir的中间文件，接下来我们将用该文件进行INT8对称量化模型的精度验证：

.. code-block:: shell

    # INT8 对称量化模型
    $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor BM1684X \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.95,0.69 \
       --model mobilenet_v2_int8.bmodel



3. 精度验证

   使用model_eval.py接口进行精度验证:

.. code-block:: shell

    # F32 模型精度验证
    $ model_eval.py \
        --model_file mobilenet_v2.mlir \
        --count 50000 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

    # INT8 对称量化模型精度验证
    $ model_eval.py \
        --model_file mobilenet_v2_bm1684x_int8_sym_tpu.mlir \
        --count 50000 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

F32模型与INT8对称量化模型的精度验证结果如下:

.. code-block:: shell

    # mobilenet_v2.mlir精度验证结果
    2022/11/08 01:30:29 - INFO : idx:50000, top1:0.710, top5:0.899
    INFO:root:idx:50000, top1:0.710, top5:0.899

    # mobilenet_v2_bm1684x_int8_sym_tpu.mlir精度验证结果
    2022/11/08 05:43:27 - INFO : idx:50000, top1:0.702, top5:0.895
    INFO:root:idx:50000, top1:0.702, top5:0.895

yolov5s
~~~~~~~~~~~~~

1. 数据集下载

   下载COCO2017验证集到datasets/val2017目录下, 该目录下即包含5000张用于验证的图片。对应的标签文件instances_val2017.json下载到datasets目录下。

2. 模型转换

   转换流程与mobilenet_v2相似。

3. 精度验证

   使用model_eval.py接口进行精度验证:

.. code-block:: shell

    # F32 模型精度验证
    $ model_eval.py \
        --model_file yolov5s.mlir \
        --count 5000 \
        --dataset_type coco \
        --postprocess_type coco_mAP \
        --coco_annotation datasets/instances_val2017.json \
        --dataset datasets/val2017

    # INT8 对称量化模型精度验证
    $ model_eval.py \
        --model_file yolov5s_bm1684x_int8_sym_tpu.mlir \
        --count 5000 \
        --dataset_type coco \
        --postprocess_type coco_mAP \
        --coco_annotation datasets/instances_val2017.json \
        --dataset datasets/val2017

F32模型与INT8对称量化模型的精度验证结果如下:

.. code-block:: shell

    # yolov5s.mlir精度验证结果
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.470
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.300
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.359
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.602
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

    # yolov5s_bm1684x_int8_sym_tpu.mlir精度验证结果
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.544
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.382
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.432
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.473
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.514
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.337
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636




