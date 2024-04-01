以下操作需要在Docker容器中。关于Docker的使用, 请参考 :ref:`启动Docker Container <docker container_setup>` 。


.. code-block:: shell
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ source tpu-mlir_xxxx/envsetup.sh

``envsetup.sh`` 会添加以下环境变量:

.. list-table:: 环境变量
   :widths: 25 30 30
   :header-rows: 1

   * - 变量名
     - 值
     - 说明
   * - TPUC_ROOT
     - tpu-mlir_xxx
     - 解压后SDK包的位置
   * - MODEL_ZOO_PATH
     - ${TPUC_ROOT}/../model-zoo
     - model-zoo文件夹位置, 与SDK在同一级目录
   * - REGRESSION_PATH
     - ${TPUC_ROOT}/regression
     - regression文件夹的位置

``envsetup.sh`` 对环境变量的修改内容为:

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
    export PYTHONPATH=/usr/local/python_packages/mlir_core:$PYTHONPATH
    export PYTHONPATH=/usr/local/python_packages/:$PYTHONPATH
    export PYTHONPATH=${TPUC_ROOT}/customlayer/python:$PYTHONPATH
    export MODEL_ZOO_PATH=${TPUC_ROOT}/../model-zoo
    export REGRESSION_PATH=${TPUC_ROOT}/regression
