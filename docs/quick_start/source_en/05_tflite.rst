Compile the TFLite model
========================

This chapter takes the ``resnet50_int8.tflite`` model as an example to introduce how to compile and transfer a TFLite model to run on the BM1684X TPU platform.

This chapter requires the following files (where xxxx corresponds to the actual version information):

**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

Load tpu-mlir
------------------

.. include:: env_var.rst


Prepare working directory
-------------------------

Create a ``model_resnet50_tf`` directory, note that it is the same level as tpu-mlir, and put the test image file into the ``model_resnet50_tf`` directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir model_resnet50_tf && cd model_resnet50_tf
   $ cp $TPUC_ROOT/regression/model/resnet50_int8.tflite .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


``$TPUC_ROOT`` is an environment variable, corresponding to the tpu-mlir_xxxx directory.


TFLite to MLIR
------------------

The model in this example has a bgr input, whose mean is 103.939,116.779,123.68 and scale is 1.0,1.0,1.0.

The model conversion command:


.. code-block:: shell

    $ model_transform.py \
        --model_name resnet50_tf \
        --model_def  ../resnet50_int8.tflite \
        --input_shapes [[1,3,224,224]] \
        --mean 103.939,116.779,123.68 \
        --scale 1.0,1.0,1.0 \
        --pixel_format bgr \
        --test_input ../image/cat.jpg \
        --test_result resnet50_tf_top_outputs.npz \
        --mlir resnet50_tf.mlir


After converting to mlir file, a ``resnet50_tf_in_f32.npz`` file will be generated, which is the input file of the model.


MLIR to bmodel
------------------

This model is a tflite asymmetric quantized model, which can be converted into a bmodel according to the following parameters:

.. code-block:: shell

   $ model_deploy.py \
       --mlir resnet50_tf.mlir \
       --quantize INT8 \
       --asymmetric \
       --chip bm1684x \
       --test_input resnet50_tf_in_f32.npz \
       --test_reference resnet50_tf_top_outputs.npz \
       --model resnet50_tf_1684x.bmodel


Once compiled, a file named ``resnet50_tf_1684x.bmodel`` is generated.
