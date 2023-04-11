fuse post handle op
==========================

BM1684x now support fuse the post handle op into bmodel

Model transform Example
-------------------------
Take the yolov5s model as an example, use the model_transform tool to fuse the post handle op into mlir as below

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

In the above command, fuse the post handle op
into mlir as the post_handle_type config.
the remaining steps is the same as not fused.

