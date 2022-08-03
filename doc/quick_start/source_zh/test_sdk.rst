测试SDK发布包
=============


配置系统环境
~~~~~~~~~~~~

使用 :ref:`env_setup` 中的方法设置好Docker环境。由于本章中会使用到
``git-lfs`` ，如果首次使用 ``git-lfs`` 可执行下述命令进行安装和配置（仅首次执
行，同时该配置是在用户自己系统中，并非Docker container中）:

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
   $ git lfs pull --include "*.onnx" --exclude=""

此过程会从 ``GitHub`` 上下载大量数据。由于具体网络环境的差异，此过程可能耗时较长。


测试流程
~~~~~~~~

解压SDK并创建Docker容器
+++++++++++++++++++++++

在 ``tpu-mlir_xxxx.tar.gz`` 目录下，执行以下命令：

.. code-block:: console
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ cd tpu-mlir_xxxx
   $ docker pull sophgo/tpuc_dev:v1.2
   $ docker run --rm --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v1.2

运行命令后会处于Docker的容器中。


设置环境变量
++++++++++++

使用以下命令完成设置运行测试所需的环境变量：

.. code-block:: console
   :linenos:

   $ cd tpu-mlir_xxxx
   $ source envsetup.sh

该过程结束后不会有任何提示。

.. _test_main:

运行测试
++++++++

SDK的测试内容位于 ``tpu-mlir`` 的 ``regression`` 目录下，regression目录中与本测试相关
的文件目录结构如下：

::

   regression
   ├── cali_tables
   ├── config
   │   ├── mobilenet_v2.cfg
   │   ├── resnet18.cfg
   │   ├── resnet34_ssd1200.cfg
   │   ├── resnet50_v2.cfg
   │   │      ...
   │   ├── squeezenet1.0.cfg
   │   ├── vgg16.cfg
   │   └── yolov5s.cfg
   ├── image
   │   ├── COCO2017
   │   └── ILSVRC2012
   ├── model
   │   ├── resnet50_int8.tflite
   │   └── yolov5s.onnx
   └── run_all.sh


:cali_tables:
   预存的一些量化参数表，用于编译INT8模型。
:config:
   模型的配置文件，用于记录模型位置、预处理参数以及编译相关的信息。
:image:
   提供了部分COCO2017和ILSVRC2012的图片，用于统计相关模型的数据分布，生成量化表。
:model:
   提供了两个经典模型resnet50_int8.tflite和yolov5s.onnx。
:run_all.sh:
   执行SDK全部测试的脚本文件。

执行以下命令，运行全部测试样例：

.. code-block:: console
   :linenos:

   $ cd regression
   $ ./run_all.sh

该过程耗时较久（预计在1~2小时），请耐心等待。该过程会测试以下模型：

* resnet18
* resnet50_v2
* mobilenet_v2
* squeezenet
* vgg16
* resnet34_ssd1200
* yolov5s

命令正常结束后，会看到新生成的regression_out文件夹（测试输出内容都在该文件夹中）。
执行以下命令将regression_out中的文件整理到单独的目录下（此处为 ``bmodels`` ）：

.. code-block:: console
   :linenos:

   $ mkdir -m 777 -p bmodels
   $ pushd bmodels
   $ ../prepare_bmrttest.py ../regression_out
   $ cp -f ../run_bmrttest.py ./run.py
   $ popd

运行完成后，会看到在 ``bmodels`` 文件夹中存放着相关模型的编译结果（bmodel，
以及用于验证模型正确性的参考输入、输出数据）。

测试模型的正确性和性能
++++++++++++++++++++++

由于 ``TPU-MLIR SDK`` 中没有1684X设备运行的环境，所以需要在Docker外测试模型
的性能。此处假设您的系统中已经部署了1684X设备并安装了相关驱动。进入
:ref:`test_main` 中生成的 ``bmodels`` 文件夹，运行以下命令：

.. code-block:: console

   $ ./run.py

命令结束后，会产生一个后缀名为 ``csv`` 的文件。该文件中记录了相关模型的运行时间
、计算资源利用率和带宽利用率。
