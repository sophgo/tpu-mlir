Use TPU for Postprocessing
==============================
Currently, TPU-MLIR supports integrating the post-processing of YOLO series and SSD network models into the model. The chips currently supporting this function include BM1684X, BM1688, and CV186X. This chapter will take the conversion of YOLOv5s to F16 model as an example to introduce how this function is used.

This chapter requires the following files (where xxxx corresponds to the actual version information):


**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

Load tpu-mlir
------------------

.. include:: env_var.rst


Prepare working directory
-------------------------

Create a ``model_yolov5s`` directory, note that it is the same level directory as tpu-mlir; and put both model files and image files
into the ``model_yolov5s`` directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir yolov5s_onnx && cd yolov5s_onnx
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.


ONNX to MLIR
--------------------

The model conversion command is as follows:

.. code-block:: shell

   $ model_transform.py \
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


MLIR to Bmodel
--------------------

To convert the MLIR file to an F16 bmodel, proceed as follows:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --chip bm1684x \
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

Here, [1, 1, 200, 7] is the maximum shape, and the actual output varies depending on the number of detected boxes.

Bmodel Verification
-----------------------

In this release package, there is a YOLOv5 use case written in Python, with the source code located at
``$TPUC_ROOT/python/samples/detect_yolov5.py``. It is used for object detection in images.
By reading this code, you can understand how the final output result is transformed into bounding boxes.

The command execution is as follows:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_1684x_f16.bmodel \
       --net_input_dims 640,640 \
       --fuse_preprocess \
       --fuse_postprocess \
       --output dog_out.jpg
