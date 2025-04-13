.. _add_postprocess:

Use Tensor Computing Processor for Postprocessing
==================================================
Currently, TPU-MLIR supports integrating the post-processing of YOLO series and SSD network models into the model. The processors currently supporting this function include BM1684X, BM1688, CV186X and BM1690. This chapter will take the conversion of YOLOv5s and YOLOv8s_seg to F16 model as an example to introduce how this function is used.

This chapter requires the tpu_mlir python package.

Go to the Docker container and execute the following command to install tpu_mlir:

.. code-block:: shell

   $ pip install tpu_mlir[onnx]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[onnx]

.. include:: get_resource.rst


Add detection post_process(YOLOv5s)
-----------------------------------

Prepare working directory
^^^^^^^^^^^^^^^^^^^^^^^^^


Create a ``model_yolov5s`` directory, and put both model files and image files into the ``model_yolov5s`` directory.

The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir yolov5s_onnx && cd yolov5s_onnx
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


ONNX to MLIR
^^^^^^^^^^^^

The model conversion command is as follows:

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

There are two points to note here. The first is that the ``--add_postprocess`` argument needs to be included in the command.
The second is that the specified ``--output_names`` should correspond to the final convolution operation.

The generated yolov5s.mlir file finally has a top.YoloDetection inserted at the end as follows:

.. code-block:: text
    :linenos:

    %260 = "top.Weight"() : () -> tensor<255x512x1x1xf32> loc(#loc261)
    %261 = "top.Weight"() : () -> tensor<255xf32> loc(#loc262)
    %262 = "top.Conv"(%253, %260, %261) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], relu_limit = -1.000000e+00 : f64, strides = [1, 1]} : (tensor<1x512x6x32xf32>, tensor<255x512x1x1xf32>, tensor<255xf32>) -> tensor<1x255x6x32xf32> loc(#loc263)
    %263 = "top.YoloDetection"(%256, %259, %262) {anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], class_num = 80 : i64, keep_topk = 200 : i64, net_input_h = 640 : i64, net_input_w = 640 : i64, nms_threshold = 5.000000e-01 : f64, num_boxes = 3 : i64, obj_threshold = 0.69999999999999996 : f64, version = "yolov5"} : (tensor<1x255x24x128xf32>, tensor<1x255x12x64xf32>, tensor<1x255x6x32xf32>) -> tensor<1x1x200x7xf32> loc(#loc264)
    return %263 : tensor<1x1x200x7xf32> loc(#loc)

Here you can see that top.YoloDetection includes parameters such as anchors, num_boxes, and so on. If the post-processing is not standard YOLO, and needs to be changed to other parameters, these parameters in the MLIR file can be directly modified.
Also, the output has been changed to one, with the shape of 1x1x200x7, where 200 represents the maximum number of detection boxes. When there are multiple batches, its value will change to batchx200. The 7 elements respectively represent [batch_number, class_id, score, center_x, center_y, width, height].
The coordinates are relative to the width and length of the model input, it's 640x640 in this example, with the following values:

.. code-block:: text

   [0., 16., 0.924488, 184.21094, 401.21973, 149.66412, 268.50336 ]


MLIR to Bmodel
^^^^^^^^^^^^^^

To convert the MLIR file to an F16 bmodel, proceed as follows:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --processor bm1684x \
       --fuse_preprocess \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --model yolov5s_1684x_f16.bmodel

Here, the ``--fuse_preprocess`` parameter is added in order to integrate the preprocessing into the model as well.
In this way, the converted model is a model that includes post-processing. The model information can be viewed with ``model_tool`` as follows:

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

Here, [1, 1, 200, 7] is the maximum shape, and the actual output varies depending on the number of detected boxes.


Bmodel Verification
^^^^^^^^^^^^^^^^^^^

In tpu_mlir package, there are yolov5 use cases written in python, using the ``detect_yolov5`` command to detect objects in images.
This command corresponds to the source code path ``{package/path/to/tpu_mlir}/python/samples/detect_yolov5.py``.
It is used for object detection in images.
By reading this code, you can understand how the final output result is transformed into bounding boxes.

The command execution is as follows:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_1684x_f16.bmodel \
       --net_input_dims 640,640 \
       --fuse_preprocess \
       --fuse_postprocess \
       --output dog_out.jpg

Add segmentation post_processing(YOLOv8s_seg)
------------------------------

Prepare working directory
^^^^^^^^^^^^^^^^^^^^^^^^^

Create a ``model_yolov8s_seg`` directory, export ONNX model file by the official model, and put image files into the ``model_yolov8s_seg`` directory.

The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir yolov8s_seg_onnx && cd yolov8s_seg_onnx
   $ python -c "import torch; from ultralytics import YOLO; model = YOLO('yolov8s-seg.pt'); model.export(format='onnx')"
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace

ONNX to MLIR
^^^^^^^^^^^^

The model conversion command is as follows:

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


The generated ``yolov8s_seg.mlir`` file has several operators named ``yolo_seg_post*`` inserted at the end, which are used for post-processing operations such as coordinate transformation, NMS, and matrix multiplication.

.. code-block:: text
    :linenos:

    %429 = "top.Sigmoid"(%428) {bias = 0.000000e+00 : f64, log = false, round_mode = "HalfAwayFromZero", scale = 1.000000e+00 : f64} : (tensor<8400x25600xf32>) -> tensor<8400x25600xf32> loc(#loc431)
    %430 = "top.Reshape"(%429) {flatten_start_dim = -1 : i64} : (tensor<8400x25600xf32>) -> tensor<8400x160x160xf32> loc(#loc432)
    %431 = "top.Slice"(%430, %0, %0, %0) {axes = [], ends = [100, 160, 160], hasparamConvert_axes = [], offset = [0, 0, 0], steps = [1, 1, 1]} : (tensor<8400x160x160xf32>, none, none, none) -> tensor<100x160x160xf32> loc(#loc433)
    %432 = "top.Slice"(%425, %0, %0, %0) {axes = [], ends = [100, 6], hasparamConvert_axes = [], offset = [0, 0], steps = [1, 1]} : (tensor<8400x38xf32>, none, none, none) -> tensor<100x6xf32> loc(#loc434)
    return %431, %432 : tensor<100x160x160xf32>, tensor<100x6xf32> loc(#loc)


The model has 2 outputs,  which masks_uncrop_uncompare is the raw segmentation mask with a shape of ``100x160x160``, where 100 represents the maximum number of detection boxes, and 160x160 corresponds to the pixel size of the final feature map.

The seg_out represents the detection boxes with a shape of ``100x6``, where 100 indicates the maximum number of detection boxes,
and the 6 elements respectively represent ``[x_left, y_up, x_right, y_bottom, score, class_id]``.
Currently, post-processing for segmentation models with multiple batches is not supported, so batch information is not included.
The coordinates are relative to the width and length of the model input, it's 640x640 in this example, with the following values:

.. code-block:: text

   [-74.06776, 263.67566, 74.06777, 531.1172, 0.9523437, 16.]


From the raw mask to the final mask output, a resize operation is required to scale it back to the original image size.
Then, the excess parts of the mask are cropped based on the detection boxes in seg_out.
Finally, threshold filtering is applied to the mask to obtain the full-resolution mask.
The processing code can be referenced in the workflow of the source file located at ``{package/path/to/tpu_mlir}/python/samples/segment_yolo.py``.




MLIR to Bmodel
^^^^^^^^^^^^^^

To convert the MLIR file to an F16 bmodel, proceed as follows:

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


Here, the ``--fuse_preprocess`` parameter is added in order to integrate the preprocessing into the model as well.
The yolov8s_seg_qtable is used because post-processing applies an offset operation to the boxes,
multiplying their coordinates by a large integer.
FP16 can introduce inaccuracies with such operations, leading to excessive masks and reduced post-processing performance.
To address this, mixed precision with FP32 is used for these operations.

Bmodel Verification
^^^^^^^^^^^^^^^^^^^

In tpu_mlir package, there are yolov8s_seg use cases written in python, using the ``segment_yolo`` command to detect objects in images.
This command corresponds to the source code path ``{package/path/to/tpu_mlir}/python/samples/segment_yolo.py``.
It is used for instance segmentation in images.
By reading this code, you can understand how the final output result is transformed into masks and bounding boxes.

The command execution is as follows:

.. code-block:: shell

   $ segment_yolo \
       --input ../image/dog.jpg \
       --model yolov8s_seg_1684x_f16.bmodel \
       --net_input_dims 640,640 \
       --fuse_preprocess \
       --fuse_postprocess \
       --output dog_out.jpg
