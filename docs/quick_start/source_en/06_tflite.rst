Compile the TFLite model
========================

This chapter takes the ``lite-model_mobilebert_int8_1.tflite`` model as an example to introduce how to compile and transfer a TFLite model to run on the BM1684X TPU platform.

This chapter requires the following files (where xxxx corresponds to the actual version information):

**tpu-mlir_xxxx.tar.gz (The release package of tpu-mlir)**

Load tpu-mlir
------------------

.. include:: env_var.rst


Prepare working directory
-------------------------

Create a ``mobilebert_tf`` directory, note that it is the same level as tpu-mlir, and put the test image file into the ``mobilebert_tf`` directory.


The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir mobilebert_tf && cd mobilebert_tf
   $ wget -O lite-model_mobilebert_int8_1.tflite https://storage.googleapis.com/tfhub-lite-models/iree/lite-model/mobilebert/int8/1.tflite
   $ cp ${REGRESSION_PATH}/npz_input/squad_data.npz .
   $ mkdir workspace && cd workspace

``REGRESSION_PATH`` is an environment variable, corresponding to the tpu-mlir_xxxx/regression directory.


TFLite to MLIR
------------------

The model conversion command:


.. code-block:: shell

    $ model_transform.py \
        --model_name mobilebert_tf \
        --mlir mobilebert_tf.mlir \
        --model_def ../lite-model_mobilebert_int8_1.tflite \
        --test_input ../squad_data.npz \
        --test_result mobilebert_tf_top_outputs.npz \
        --input_shapes [[1,384],[1,384],[1,384]] \
        --channel_format none


After converting to mlir file, a ``mobilebert_tf_in_f32.npz`` file will be generated, which is the input file of the model.


MLIR to bmodel
------------------

This model is a tflite asymmetric quantized model, which can be converted into a bmodel according to the following parameters:

.. code-block:: shell

    $ model_deploy.py \
        --mlir mobilebert_tf.mlir \
        --quantize INT8 \
        --chip bm1684x \
        --test_input mobilebert_tf_in_f32.npz \
        --test_reference mobilebert_tf_top_outputs.npz \
        --model mobilebert_tf_bm1684x_int8.bmodel


Once compiled, a file named ``mobilebert_tf_bm1684x_int8.bmodel`` is generated.
