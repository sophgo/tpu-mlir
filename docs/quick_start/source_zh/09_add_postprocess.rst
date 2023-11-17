使用TPU做后处理
==================
目前TPU-MLIR支持将yolo系列和ssd网络模型的后处理集成到模型中, 目前支持该功能的芯片有BM1684X、BM1688、CV186X芯片。

本章将yolov5s转成为F16模型为例, 介绍该功能如何被使用。

本章需要安装tpu_mlir。


安装tpu-mlir
------------------

.. code-block:: shell

   $ pip install tpu_mlir[onnx]


准备工作目录
------------------

建立 ``model_yolov5s`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都
放入 ``model_yolov5s`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir yolov5s_onnx && cd yolov5s_onnx
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ tpu_mlir_get_resource regression/dataset/COCO2017 .
   $ tpu_mlir_get_resource regression/image .
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

ONNX转MLIR
--------------------

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
    :linenos:

    [0., 16., 0.924488, 184.21094, 401.21973, 149.66412, 268.50336 ]


MLIR转换成BModel
--------------------

将mlir文件转换成F16的bmodel, 操作方法如下:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --chip bm1684x \
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
    chip: BM1684X
    create time: Fri May 26 16:30:20 2023

    kernel_module name: libbm1684x_kernel_module.so
    kernel_module size: 2037536
    ==========================================
    net 0: [yolov5s]  static
    ------------
    stage 0:
    subnet number: 2
    input: images_raw, [1, 3, 640, 640], uint8, scale: 1, zero_point: 0
    output: yolo_post, [1, 1, 200, 7], float32, scale: 1, zero_point: 0

    device mem size: 24970588 (coeff: 14757888, instruct: 1372, runtime: 10211328)
    host mem size: 0 (coeff: 0, runtime: 0)

这里的 ``[1, 1, 200, 7]`` 是最大shape, 实际输出根据检测的框数有所不同。

模型验证
-------------

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
