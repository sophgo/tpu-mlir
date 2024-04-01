Compile the Caffe model
=======================

This chapter takes ``mobilenet_v2_deploy.prototxt`` and ``mobilenet_v2.caffemodel`` as examples to introduce how to compile and transfer a caffe model to run on the BM1684X platform.

This chapter requires the tpu_mlir python package.


Install tpu_mlir
------------------

Go to the Docker container and execute the following command to install tpu_mlir:

.. code-block:: shell

   $ pip install tpu_mlir[caffe]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[caffe]


Prepare working directory
-------------------------

.. include:: get_resource.rst

Create a ``mobilenet_v2`` directory, and put both model files and image files into the ``mobilenet_v2`` directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir mobilenet_v2 && cd mobilenet_v2
   $ wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt
   $ wget https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
   $ cp -rf tpu_mlir_resource/dataset/ILSVRC2012 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


Caffe to MLIR
------------------

The model in this example has a `BGR` input with mean and scale of ``103.94, 116.78, 123.68`` and ``0.017, 0.017, 0.017`` respectively.

The model conversion command:


.. code-block:: shell

   $ model_transform \
       --model_name mobilenet_v2 \
       --model_def ../mobilenet_v2_deploy.prototxt \
       --model_data ../mobilenet_v2.caffemodel \
       --input_shapes [[1,3,224,224]] \
       --resize_dims=256,256 \
       --mean 103.94,116.78,123.68 \
       --scale 0.017,0.017,0.017 \
       --pixel_format bgr \
       --test_input ../image/cat.jpg \
       --test_result mobilenet_v2_top_outputs.npz \
       --mlir mobilenet_v2.mlir

After converting to mlir file, a ``${model_name}_in_f32.npz`` file will be generated, which is the input file of the model.


MLIR to F32 bmodel
------------------

Convert the mlir file to the bmodel of f32, the operation method is as follows:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize F32 \
       --processor bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model mobilenet_v2_1684x_f32.bmodel

After compilation, a file named ``mobilenet_v2_1684x_f32.bmodel`` is generated.


MLIR to INT8 bmodel
-------------------

Calibration table generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before converting to the INT8 model, you need to run calibration to get the calibration table. The number of input data is about 100 to 1000 according to the situation.

Then use the calibration table to generate a symmetric or asymmetric bmodel. It is generally not recommended to use the asymmetric one if the symmetric one already meets the requirements, because
the performance of the asymmetric model will be slightly worse than the symmetric model.

Here is an example of the existing 100 images from ILSVRC2012 to perform calibration:


.. code-block:: shell

   $ run_calibration mobilenet_v2.mlir \
       --dataset ../ILSVRC2012 \
       --input_num 100 \
       -o mobilenet_v2_cali_table

After running the command above, a file named ``mobilenet_v2_cali_table`` will be generated, which is used as the input file for subsequent compilation of the INT8 model.


Compile to INT8 symmetric quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the following command to convert to the INT8 symmetric quantized model:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor bm1684x \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --model mobilenet_v2_1684x_int8_sym.bmodel

After compilation, a file named ``mobilenet_v2_1684x_int8_sym.bmodel`` is generated.
