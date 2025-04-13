.. _add_postprocess:

使用智能深度学习处理器做后处理
=================================
目前TPU-MLIR支持将yolo系列和ssd网络模型的后处理集成到模型中, 目前支持该功能的处理器有BM1684X、BM1688、CV186X、BM1690。

本章将yolov5s和yolov8s_seg转成为F16模型为例, 介绍该功能如何被使用。

本章需要安装TPU-MLIR
进入Docker容器，并执行以下命令安装TPU-MLIR：

.. code-block:: shell

   $ pip install tpu_mlir[onnx]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[onnx]

.. include:: get_resource.rst

检测模型后处理添加（yolov5s）
-------------------------

准备工作目录
^^^^^^^^^^

建立 ``model_yolov5s`` 目录, 并把模型文件和图片文件都放入 ``model_yolov5s`` 目录中。

操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir yolov5s_onnx && cd yolov5s_onnx
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


ONNX转MLIR
^^^^^^^^^^

模型转换命令如下:

.. code-block:: shell

   $ model_transform \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 326,474,622 \
       --add_postprocess yolov5 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

这里要注意两点, 一是命令中需要加入 ``--add_postprocess`` 参数; 二是指定的 ``--output_names`` 对应最后的卷积操作。

生成后的 ``yolov5s.mlir`` 文件最后被插入了一个 ``top.YoloDetection``, 如下:

.. code-block:: text
    :linenos:

    %260 = "top.Weight"() : () -> tensor<255x512x1x1xf32> loc(#loc261)
    %261 = "top.Weight"() : () -> tensor<255xf32> loc(#loc262)
    %262 = "top.Conv"(%253, %260, %261) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], relu_limit = -1.000000e+00 : f64, strides = [1, 1]} : (tensor<1x512x20x20xf32>, tensor<255x512x1x1xf32>, tensor<255xf32>) -> tensor<1x255x20x20xf32> loc(#loc263)
    %263 = "top.YoloDetection"(%256, %259, %262) {agnostic_nms = false, anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], class_num = 80 : i64, keep_topk = 200 : i64, net_input_h = 640 : i64, net_input_w = 640 : i64, nms_threshold = 5.000000e-01 : f64, num_boxes = 3 : i64, obj_threshold = 5.000000e-01 : f64, version = "yolov5"} : (tensor<1x255x80x80xf32>, tensor<1x255x40x40xf32>, tensor<1x255x20x20xf32>) -> tensor<1x1x200x7xf32> loc(#loc264)
    return %263 : tensor<1x1x200x7xf32> loc(#loc)

这里看到 ``top.YoloDetection`` 包括了anchors、num_boxes等等参数, 如果并非标准的yolo后处理, 需要改成其他参数, 可以直接修改mlir文件的这些参数。

另外输出也变成了1个, shape为 ``1x1x200x7``, 其中200代表最大检测框数, 当有多个batch时, 它的数值会变为 ``batch x 200``;
7分别指 ``[batch_number, class_id, score, center_x, center_y, width, height]``。
其中坐标是相对模型输入长宽的坐标, 比如本例中640x640, 数值参考如下：

.. code-block:: text

   [0., 16., 0.924488, 184.21094, 401.21973, 149.66412, 268.50336 ]


MLIR转换成BModel
^^^^^^^^^^^^^^^

将mlir文件转换成F16的bmodel, 操作方法如下:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --processor bm1684x \
       --fuse_preprocess \
       --test_input ../image/dog.jpg \
       --test_reference yolov5s_top_outputs.npz \
       --model yolov5s_1684x_f16.bmodel

这里加上参数 ``--fuse_preprocess``, 是为了将前处理也合并到模型中。
这样转换后的模型就是包含了前后处理的模型, 用 ``model_tool`` 查看模型信息如下:

.. code-block:: shell

   $ model_tool --info yolov5s_1684x_f16.bmodel


.. code-block:: text
    :linenos:

    bmodel version: B.2.2
    processor: BM1684X
    create time: Wed Jan  3 07:29:14 2024

    kernel_module name: libbm1684x_kernel_module.so
    kernel_module size: 2677600
    ==========================================
    net 0: [yolov5s]  static
    ------------
    stage 0:
    subnet number: 2
    input: images_raw, [1, 3, 640, 640], uint8, scale: 1, zero_point: 0
    output: yolo_post, [1, 1, 200, 7], float32, scale: 1, zero_point: 0

    device mem size: 31238060 (coeff: 14757888, instruct: 124844, runtime: 16355328)
    host mem size: 0 (coeff: 0, runtime: 0)

这里的 ``[1, 1, 200, 7]`` 是最大shape, 实际输出根据检测的框数有所不同。


模型验证
^^^^^^^

在本发布包中有用python写好的yolov5用例, 使用 ``detect_yolov5`` 命令, 用于对图片进行目标检测。
该命令对应源码路径 ``{package/path/to/tpu_mlir}/python/samples/detect_yolov5.py`` 。
阅读该代码可以了解最终输出结果是怎么转换画框的。

命令执行如下:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_1684x_f16.bmodel \
       --net_input_dims 640,640 \
       --fuse_preprocess \
       --fuse_postprocess \
       --output dog_out.jpg

分割模型后处理添加（yolov8s_seg）
------------------------------

准备工作目录
^^^^^^^^^^

建立 ``model_yolov8s_seg`` 目录, 使用官方模型导出onnx模型文件，并将图片文件放入 ``model_yolov8s_seg`` 目录中。

操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir yolov8s_seg_onnx && cd yolov8s_seg_onnx
   $ python -c "import torch; from ultralytics import YOLO; model = YOLO('yolov8s-seg.pt'); model.export(format='onnx')"
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace

ONNX转MLIR
^^^^^^^^^^

模型转换命令如下:

.. code-block:: shell

   $ model_transform \
       --model_name yolov8s_seg \
       --model_def ../yolov8s-seg.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --add_postprocess yolov8_seg \
       --test_input ../image/dog.jpg \
       --test_result yolov8s_seg_top_outputs.npz \
       --mlir yolov8s_seg.mlir


生成后的 ``yolov8s_seg.mlir`` 文件最后插入了若干命名为 ``yolo_seg_post*`` 的算子，用于执行后处理中的坐标变换、nms、矩阵乘等运算。

.. code-block:: text
    :linenos:

    %429 = "top.Sigmoid"(%428) {bias = 0.000000e+00 : f64, log = false, round_mode = "HalfAwayFromZero", scale = 1.000000e+00 : f64} : (tensor<8400x25600xf32>) -> tensor<8400x25600xf32> loc(#loc431)
    %430 = "top.Reshape"(%429) {flatten_start_dim = -1 : i64} : (tensor<8400x25600xf32>) -> tensor<8400x160x160xf32> loc(#loc432)
    %431 = "top.Slice"(%430, %0, %0, %0) {axes = [], ends = [100, 160, 160], hasparamConvert_axes = [], offset = [0, 0, 0], steps = [1, 1, 1]} : (tensor<8400x160x160xf32>, none, none, none) -> tensor<100x160x160xf32> loc(#loc433)
    %432 = "top.Slice"(%425, %0, %0, %0) {axes = [], ends = [100, 6], hasparamConvert_axes = [], offset = [0, 0], steps = [1, 1]} : (tensor<8400x38xf32>, none, none, none) -> tensor<100x6xf32> loc(#loc434)
    return %431, %432 : tensor<100x160x160xf32>, tensor<100x6xf32> loc(#loc)

模型的输出共有2个, 其中masks_uncrop_uncompare是原始的分割掩码，shape为 ``100x160x160``, 其中100代表最大检测框数，160x160代表了最后一层特征图的像素大小。

seg_out是检测框，shape为 ``100x6``,其中100代表最大检测框数，6分别指 ``[x_left, y_up, x_right, y_bottom, score, class_id]``。目前暂不支持多batch的分割模型后处理添加，因此没有batch信息。
其中坐标是相对模型输入长宽的坐标, 比如本例中640x640, 数值参考如下：

.. code-block:: text

   [-74.06776, 263.67566, 74.06777, 531.1172, 0.9523437, 16.]


从原始掩码到最后的掩码输出，还需要将其进行resize运算，放大回原始图片大小；并根据seg_out中检测框对mask多余部分进行crop；
最后对掩码进行阈值过滤，得到全分辨率的掩码。以上的处理代码可参考该源码路径 ``{package/path/to/tpu_mlir}/python/samples/segment_yolo.py`` 中的流程。




MLIR转换成BModel
^^^^^^^^^^^^^^^

将mlir文件转换成F16的bmodel, 操作方法如下:

.. code-block:: shell

   $ fp_forward.py yolov8s_seg.mlir \
       --fpfwd_outputs yolo_seg_post_mulconst3 \
       --chip bm1684x \
       --fp_type F32 \
       -o yolov8s_seg_qtable

   $ model_deploy \
       --mlir yolov8s_seg.mlir \
       --quantize F16 \
       --processor bm1684x \
       --fuse_preprocess \
       --quantize_table yolov8s_seg_qtable \
       --model yolov8s_seg_1684x_f16.bmodel


这里加上参数 ``--fuse_preprocess``, 是为了将前处理也合并到模型中；还使用了yolov8s_seg_qtable，这是因为后处理中会对box施加偏移运算，将框图的坐标数值乘以一个大整数，F16的特性会导致涉及大整数的运算存在偏差，
最终会导致产生过多的mask，影响后处理性能，因此需要将这一部分算子使用F32混合精度进行运算。


模型验证
^^^^^^^

在本发布包中有用python写好的yolov8s_seg用例, 使用 ``segment_yolo`` 命令, 用于对图片进行分割。
该命令对应源码路径  ``{package/path/to/tpu_mlir}/python/samples/segment_yolo.py``。
阅读该代码可以了解最终输出结果；并且如何产生分割掩膜和框图。

命令执行如下:

.. code-block:: shell

   $ segment_yolo \
       --input ../image/dog.jpg \
       --model yolov8s_seg_1684x_f16.bmodel \
       --net_input_dims 640,640 \
       --fuse_preprocess \
       --fuse_postprocess \
       --output dog_out.jpg
