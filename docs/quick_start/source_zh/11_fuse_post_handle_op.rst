后处理算子融合
==========================

BM1684x 支持融合后处理算子到bmodel中
model转换示例
-------------------------

以yolov5s作为例子，使用model_transform融合后处理算子到mlir中，操作步骤如下：
.. code-block:: shell

    $model_transform.py \
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
      --mlir yolov5s.mlir \
      --post_handle_type yolo

按如上命令，通过配置post_handle_type,将后处理算子融合到mlir中, 剩下的转换命令和以往一样


