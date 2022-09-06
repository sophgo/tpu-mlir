使用TPU-PERF测试SDK发布包
=========================


配置系统环境
~~~~~~~~~~~~

如果是首次使用Docker，那么请使用 :ref:`首次使用Docker <docker configuration>` 中的方法安装
并配置Docker。同时，本章中会使用到 ``git-lfs`` ，如果首次使用 ``git-lfs`` 可执行下述命
令进行安装和配置（仅首次执行，同时该配置是在用户自己系统中，并非Docker container中）:

.. code-block:: console
   :linenos:

   $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   $ sudo apt-get install git-lfs


获取 ``model-zoo`` 模型
~~~~~~~~~~~~~~~~~~~~~~~

在 ``tpu-mlir_xxxx.tar.gz`` （tpu-mlir的发布包）的同级目录下，使用以下命令克隆 ``model-zoo`` 工程：

.. code-block:: console
   :linenos:

   $ git clone --depth=1 https://github.com/sophgo/model-zoo
   $ cd model-zoo
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

如果已经克隆过 ``model-zoo`` 可以执行以下命令同步模型到最新状态：

.. code-block:: console
   :linenos:

   $ cd model-zoo
   $ git pull
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

此过程会从 ``GitHub`` 上下载大量数据。由于具体网络环境的差异，此过程可能耗时较长。

获取 ``tpu-perf`` 工具
~~~~~~~~~~~~~~~~~~~~~~

从 https://github.com/sophgo/tpu-perf/releases 地址下载最新的 ``tpu-perf``
wheel安装包。例如：tpu_perf-x.x.x-py3-none-any.whl 。并将 ``tpu-perf`` 包放置到
与 ``model-zoo`` 同一级目录下。此时的目录结构应该为如下形式：


::

   ├── tpu_perf-x.x.x-py3-none-any.whl
   ├── tpu-mlir_xxxx.tar.gz
   └── model-zoo


测试流程
~~~~~~~~

解压SDK并创建Docker容器
+++++++++++++++++++++++

在 ``tpu-mlir_xxxx.tar.gz`` 目录下（注意，``tpu-mlir_xxxx.tar.gz`` 和
``model-zoo`` 需要在同一级目录），执行以下命令：

.. code-block:: console
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ docker pull sophgo/tpuc_dev:latest
   $ docker run --rm --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest

运行命令后会处于Docker的容器中。


设置环境变量并安装 ``tpu-perf``
+++++++++++++++++++++++++++++++

使用以下命令完成设置运行测试所需的环境变量：

.. code-block:: console
   :linenos:

   $ cd tpu-mlir_xxxx
   $ source envsetup.sh

该过程结束后不会有任何提示。之后使用以下命令安装 ``tpu-perf``：

.. code-block:: console

   $ pip3 install ../tpu_perf-*-py3-none-any.whl


.. _test_main:

运行测试
++++++++

编译模型
````````

``model-zoo`` 的相关 ``confg.yaml`` 配置了SDK的测试内容。例如：resnet18的
配置文件为 ``model-zoo/vision/classification/resnet18-v2/config.yaml`` 。

执行以下命令，运行全部测试样例：

.. code-block:: console
   :linenos:

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --mlir --full

该过程耗时较久（预计在1~2小时），请耐心等待。此时会编译以下模型：

::

   * efficientnet-lite4
   * mobilenet_v2
   * resnet18
   * resnet50_v2
   * shufflenet_v2
   * squeezenet1.0
   * vgg16
   * yolov5s


命令正常结束后，会看到新生成的 ``output`` 文件夹（测试输出内容都在该文件夹中）。
修改 ``output`` 文件夹的属性，以保证其可以被Docker外系统访问。


.. code-block:: console
   :linenos:

   $ chmod -R a+rw output


测试模型性能
````````````

运行测试需要在 Docker 外面的环境（此处假设您已经安装并配置好了1684X设备和
驱动）中进行，可以退出 Docker 环境：

.. code :: console

   $ exit

在Docker外环境下运行以下命令，测试生成的 ``bmodel`` 性能。

.. code-block:: console
   :linenos:

   $ pip3 install ./tpu_perf-*-py3-none-any.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --mlir --full

运行结束后，性能数据在 ``output/stats.csv`` 中可以获得。该文件中记录了相关模型的
运行时间、计算资源利用率和带宽利用率。
