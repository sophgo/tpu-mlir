Appendix.03: BM168x Guidance
=============================

BM168x series processor currently supports ONNX, pytorch, Caffe and TFLite models.
This chapter takes the BM1684x processor as an example to introduce merging bmodel files of the BM168x series processors.

.. _merge weight:

Merge bmodel Files
---------------------------
For the same model, independent bmodel files can be generated according to the input batch size and resolution(different H and W). However, in order to save storage, you can merge these related bmodel files into one bmodel file and share its weight part. The steps are as follows:

Step 0: generate the bmodel for batch 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the previous section to create a new workspace directory and convert yolov5s to the mlir fp32 model by model_transform

.. admonition:: Attention ：
  :class: attention

  1.Use the same workspace directory for the bmodels that need to be merged, and do not share the workspace with other bmodes that do not need to be merged.

  2.In Step 0, Step 1, --merge_weight is required


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

Use the yolov5s_cali_table generated in preceding sections, or generate calibration table by run_calibration.

.. code-block:: shell

  # Add --merge_weight
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

Step 1: generate the bmodel for batch 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate mlir fp32 file in the same workspace:

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

  # Add --merge_weight
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

Step 2: merge the bmodel of batch 1 and batch 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use model_tool to merge two bmodel files:

.. code-block:: shell

  model_tool \
    --combine \
      yolov5s_bm1684x_int8_sym_bs1.bmodel \
      yolov5s_bm1684x_int8_sym_bs2.bmodel \
      -o yolov5s_bm1684x_int8_sym_bs1_bs2.bmodel


Overview:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the above command, you can merge either the same models or different models

The main steps are:

1. When generating a bmodel through model_deploy, add the --merge_weight parameter.
2. The work directory of the model to be merged must be the same, and do not clean up any intermediate files before merging the models(Reuse the previous model's weight is implemented through the intermediate file _weight_map.csv).
3. Use model_tool to merge bmodels.
