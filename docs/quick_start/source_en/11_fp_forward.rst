.. _fp_forward:

Local Non-Quantization

For specific neural networks, some layers may not be suitable for quantization due to significant differences in data distribution. The "Local Non-Quantization" allows you to add certain layers before, after, or between other layers to a mixed-precision table. These layers will not be quantized when generating a mixed-precision model.

In this chapter, we will continue using the example of the YOLOv5s network mentioned in Chapter 3 and demonstrate how to use the Local Non-Quantization to quickly generate a mix-precision model.

The process of generating FP32 and INT8 models is the same as in Chapter 3. Here, we focus on generating mix-precision model and the accuracy testing.

For YOLO series models, the last three convolutional layers often have significantly different data distributions, and adding them manually to the mixed-precision table can improve accuracy. With the Local Non-Quantization feature, you can search for the corresponding layers from the FP32 MLIR file and quickly add them to the mixed-precision table using the following command:

.. code-block:: shell

   $ fp_forward.py \
       yolov5s.mlir \
       --quantize INT8 \
       --chip bm1684x \
       --fpfwd_outputs 474_Conv,326_Conv,622_Conv\
       --chip bm1684x \
       -o yolov5s_qtable

Opening the file "yolov5s_qtable" will reveal that the relevant layers have been added to the qtable.

Generating the Mixed-Precision Model

.. code-block:: shell

  $ model_deploy.py \
      --mlir yolov5s.mlir \
      --quantize INT8 \
      --calibration_table yolov5s_cali_table \
      --quantize_table yolov5s_qtable \
      --chip bm1684x \
      --test_input yolov5s_in_f32.npz \
      --test_reference yolov5s_top_outputs.npz \
      --tolerance 0.85,0.45 \
      --model yolov5s_1684x_mix.bmodel

Validating the Accuracy of FP32 and Mixed-Precision Models
In the model-zoo, there is a program called "yolo" used for accuracy validation of object detection models. You can use the "harness" field in the mlir.config.yaml file to invoke "yolo" as follows:

Modify the relevant fields as follows:

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

Switch to the top-level directory of model-zoo and use tpu_perf.precision_benchmark for accuracy testing, as shown in the following command:
.. code-block:: shell

  $ python3 -m tpu_perf.precision_benchmark yolov5s_path --mlir --target BM1684X --devices 0

The accuracy test results will be stored in output/yolo.csv:

mAP for the FP32 model:
mAP for the mixed-precision model using the default mixed-precision table:

Performance Testing

mAP for the mixed-precision model using the manually added mixed-precision table:

Parameter Description


.. list-table:: fp_forward.py parameters
   :widths: 23 8 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - (None)
     - Y
     - mlir file
   * - chip
     - Y
     - The platform that the model will use. Support bm1684x/bm1684/cv183x/cv182x/cv181x/cv180x.
   * - fpfwd_inputs
     - N
     - Specify layers (including this layer) to skip quantization before them. Multiple inputs are separated by commas.
   * - fpfwd_outputs
     - N
     - Specify layers (including this layer) to skip quantization after them. Multiple inputs are separated by commas.
   * - fpfwd_blocks
     - N
     - Specify the start and end layers between which quantization will be skipped. Start and end layers are separated by space, and multiple blocks are separated by spaces.
   * - fp_type
     - N
     - Specifies the type of float used for mixing precision. Support auto,F16,F32,BF16. Default is auto, indicating that it is automatically selected by program
   * - o
     - Y
     - output quantization table