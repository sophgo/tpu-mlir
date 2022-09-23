.. code-block:: console
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

``envsetup.sh`` modify the environment variables as follows:

.. code-block:: shell
   :linenos:

   export PATH=${TPUC_ROOT}/bin:$PATH
   export PATH=${TPUC_ROOT}/python/tools:$PATH
   export PATH=${TPUC_ROOT}/python/utils:$PATH
   export PATH=${TPUC_ROOT}/python/test:$PATH
   export PATH=${TPUC_ROOT}/python/samples:$PATH
   export LD_LIBRARY_PATH=$TPUC_ROOT/lib:$LD_LIBRARY_PATH
   export PYTHONPATH=${TPUC_ROOT}/python:$PYTHONPATH
   export MODEL_ZOO_PATH=${TPUC_ROOT}/../model-zoo
