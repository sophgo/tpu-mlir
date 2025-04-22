Appendix.01: Migrating from NNTC to tpu-mlir
============================================

NNTC is using docker version sophgo/tpuc_dev:v2.1, for MLIR docker version
reference and environment initialization please refer to :ref:`Environment Setup <docker configuration>`.


In the following, we will use yolov5s as an example to explain the similarities
and differences between nntc and mlir in terms of quantization, and for
compiling floating-point models, please refer to <TPU-MLIR_Quick_Start>
`Compile the ONNX model`.

First, refer to the section `Compile the ONNX model` to prepare the yolov5s model.

ONNX to MLIR
------------

To quantize a model in mlir, first convert the original model to a top-level mlir file, this step can be compared to generating a fp32umodel in step-by-step quantization in nntc.

#. MLIR's model conversion command

   .. code-block:: shell

      $ model_transform.py \
          --model_name yolov5s \
          --model_def ../yolov5s.onnx \
          --input_shapes [[1,3,640,640]] \
          --mean 0.0,0.0,0.0 \
          ---scale 0.0039216,0.0039216,0.0039216 \
          --keep_aspect_ratio \
          --pixel_format rgb \
          --output_names 350,498,646 \
          ---test_input ./image/dog.jpg \
          ---test_result yolov5s_top_outputs.npz \
          --mlir yolov5s.mlir

   TPU-MLIR can directly encode image preprocessing into the converted MLIR file.

#. Model transformation commands for NNTC

   .. code-block:: shell

       $ python3 -m ufw.tools.on_to_umodel \
           -m ../yolov5s.onnx \
           -s '(1,3,640,640)' \
           -d 'compilation' \
           --cmp

   When importing a model with NNTC, you cannot specify the preprocessing
   method.

Make a quantization calibration table
-------------------------------------

If you want to generate a fixed-point model, you need a quantization tool to quantize the model, nntc uses calibration_use_pb for step-by-step quantization, and mlir uses run_calibration.py for step-by-step quantization.

The number of input data is about 100~1000 depending on the situation, using the existing 100 images from COCO2017 as an example, perform calibration.

To use stepwise quantization in nntc, you need to make your own mdb quantization dataset using the image quantization dataset, and modify fp32_protoxt to point the data input to the lmdb file.

.. note::

   For the NNTC quantization dataset, please refer to the "Model Quantization"
   chapter in the <TPU-NNTC Development Reference Manual>, and note that the
   lmdb dataset is not compatible with TPU-MLIR. TPU-MLIR can directly use raw
   images as input for quantization tools. If it is voice, text or other
   non-image data, it needs to be converted to npz file.


#. MLIR Quantization Model

   .. code-block:: shell

      $ run_calibration.py yolov5s.mlir \
          --dataset ../COCO2017 \
          --input_num 100 \
          -o yolov5s_cali_table

   After quantization you will get the quantization table yolov5s_cali_table


#. NNTC Quantization Model

   .. code-block:: shell

       $ calibration_use_pb quantize \
            --model=./compilation/yolov5s_bmneto_test_fp32.prototxt \
            --weights=./compilation/yolov5s_bmneto.fp32umodel \
            -save_test_proto=True --bitwidth=TO_INT8

   In nntc, after quantization, you get int8umodel and prototxt.


Generating int8 models
------------------------

To convert to an INT8 symmetric quantized model, execute the following command.

#. MLIR:

   .. code-block:: shell

      $ model_deploy.py \
          ---mlir yolov5s.mlir \
          --quantize INT8 \
          --calibration_table yolov5s_cali_table \
          --processor bm1684 \
          ---test_input yolov5s_in_f32.npz \
          --test_reference yolov5s_top_outputs.npz \
          --tolerance 0.85,0.45 \
          --model yolov5s_1684_int8_sym.bmodel

   At the end of the run you get yolov5s_1684_int8_sym.bmodel.


#. NNTC:

   In nntc, the int8 bmodel is generated using int8umodel and prototxt using the bmnetu tool.

   .. code-block:: shell

      $ bmnetu --model=./compilation/yolov5s_bmneto_deploy_int8_unique_top.prototxt \
          --weight=./compilation/yolov5s_bmneto.int8umodel

   At the end of the run you get compilation.bmodel.
