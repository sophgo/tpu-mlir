CV18XX Guidance
===================

CV18XX series chip currently supports ONNX and Caffe models but not TFLite models. In terms of quantization, CV18XX supports BF16 and asymmetric INT8 format. This chapter takes the CV183X as an example to introduce the compilation and runtime sample of the CV18XX series chip.

Compile yolov5 model
--------------------

TPU-MLIR Setup
~~~~~~~~~~~~~~~~~~~~

.. include:: env_var.rst

Prepare working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the ``model_yolov5s`` directory in the same directory as tpu-mlir, and put the model and image files in this directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s && cd model_yolov5s
   $ cp $TPUC_ROOT/regression/model/yolov5s.onnx .
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


Here ``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.

ONNX to MLIR
~~~~~~~~~~~~~~~~~~~~

If the input is an image, we need to learn the preprocessing of the model before conversion. If the model uses the preprocessed npz file as input, there is no need to consider preprocessing. The preprocessing process is expressed as follows ( :math:`x` stands for input):

.. math::

   y = (x - mean) \times scale


The input of yolov5 on the official website is rgb image, each value of it will be multiplied by ``1/255``, and converted into mean and scale corresponding to ``0.0,0.0,0.0`` and ``0.0039216,0.0039216,0.0039216``.

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
       --output_names 350,498,646 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

For the argument description of ``model_transform``, refer to the section "Compile ONNX Model - ONNX to MLIR".

MLIR to BF16 Model
~~~~~~~~~~~~~~~~~~~~

Convert the mlir file to the cvimodel of bf16, the operation is as follows:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize BF16 \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model yolov5s_cv183x_bf16.cvimodel

For the argument description of ``model_deploy.py``, refer to the section "Compile ONNX model - MLIR to F32 model".

MLIR to INT8 Model
~~~~~~~~~~~~~~~~~~~~
Before converting to the INT8 model, you need to do calibration to get the calibration table. The number of input data depends on the situation but is normally around 100 to 1000. Then use the calibration table to generate INT8 symmetric cvimodel.

Here we use the 100 images from COCO2017 as an example to perform calibration:

.. code-block:: shell

   $ run_calibration.py yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

After the operation is completed, a file named ``${model_name}_cali_table`` will be generated, which is used as the input of the following compilation work.

To convert to symmetric INT8 cvimodel model, execute the following command:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_cv183x_int8_sym.cvimodel

After compiling, a file named ``${model_name}_cv183x_int8_sym.cvimodel`` will be generated.


Result Comparison
~~~~~~~~~~~~~~~~~~~~

The onnx model is run as follows to get ``dog_onnx.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model ../yolov5s.onnx \
       --output dog_onnx.jpg

The FP32 mlir model is run as follows to get ``dog_mlir.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s.mlir \
       --output dog_mlir.jpg

The BF16 cvimodel is run as follows to get ``dog_bf16.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_bf16.cvimodel \
       --output dog_bf16.jpg

The INT8 cvimodel is run as follows to get ``dog_int8.jpg``:

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_int8_sym.cvimodel \
       --output dog_int8.jpg


The comparison of the four images is shown in :numref:`yolov5s_result1`, due to the different operating environments, the final effect and accuracy will be slightly different from :numref:`yolov5s_result1`.

.. _yolov5s_result1:
.. figure:: ../assets/yolov5s_cvi.jpg
   :height: 13cm
   :align: center

   Comparing the results of different models



The above tutorial introduces the process of TPU-MLIR deploying the ONNX model to the CV18XX series chip. For the conversion process of the Caffe model, please refer to the chapter "Compiling the Caffe Model". You only need to replace the chip name with the specific CV18XX chip.


Merge cvimodel Model Files
---------------------------
To be completed


Compile and Run the Runtime Sample
-----------------------------------
To be completed


