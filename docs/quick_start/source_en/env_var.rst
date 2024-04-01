The following operations need to be in a Docker container. For the use of Docker, please refer to :ref:`Setup Docker Container <docker container_setup>`.

.. code-block:: shell
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ source tpu-mlir_xxxx/envsetup.sh

``envsetup.sh`` adds the following environment variables:

.. list-table:: Environment variables
   :widths: 25 30 30
   :header-rows: 1

   * - Name
     - Value
     - Explanation
   * - TPUC_ROOT
     - tpu-mlir_xxx
     - The location of the SDK package after decompression
   * - MODEL_ZOO_PATH
     - ${TPUC_ROOT}/../model-zoo
     - The location of the model-zoo folder, at the same level as the SDK
   * - REGRESSION_PATH
     - ${TPUC_ROOT}/regression
     - The location of the regression folder

``envsetup.sh`` modifies the environment variables as follows:

.. code-block:: shell
   :linenos:

    export PATH=${TPUC_ROOT}/bin:$PATH
    export PATH=${TPUC_ROOT}/python/tools:$PATH
    export PATH=${TPUC_ROOT}/python/utils:$PATH
    export PATH=${TPUC_ROOT}/python/test:$PATH
    export PATH=${TPUC_ROOT}/python/samples:$PATH
    export PATH=${TPUC_ROOT}/customlayer/python:$PATH
    export LD_LIBRARY_PATH=$TPUC_ROOT/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=${TPUC_ROOT}/python:$PYTHONPATH
    export PYTHONPATH=${TPUC_ROOT}/customlayer/python:$PYTHONPATH
    export MODEL_ZOO_PATH=${TPUC_ROOT}/../model-zoo
    export REGRESSION_PATH=${TPUC_ROOT}/regression
