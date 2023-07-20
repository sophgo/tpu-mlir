附录03：BM168x测试指南
==============================

配置系统环境
~~~~~~~~~~~~

如果是首次使用Docker, 那么请使用 :ref:`开发环境配置 <docker configuration>` 中的方法安装
并配置Docker。同时, 本章中会使用到 ``git-lfs`` , 如果首次使用 ``git-lfs`` 可执行下述命
令进行安装和配置(仅首次执行, 同时该配置是在用户自己系统中, 并非Docker container中):

.. code :: shell

   $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   $ sudo apt-get install git-lfs

获取 ``model-zoo`` 模型
~~~~~~~~~~~~~~~~~~~~~~~~

在 ``tpu-mlir_xxxx.tar.gz`` (tpu-mlir的发布包)的同级目录下, 使用以下命令克隆 ``model-zoo`` 工程:

.. code :: shell

   $ git clone --depth=1 https://github.com/sophgo/model-zoo
   $ cd model-zoo
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG,*.npz" --exclude=""
   $ cd ../

model-zoo的目录结构如下：

.. code ::

    ├── config.yaml
    ├── requirements.txt
    ├── data
    ├── dataset
    ├── harness
    ├── output
    └── ...

* config.yaml 中包含通用的配置：数据集的目录、模型的根目录等，以及一些复用的参数和命令
* requirements.txt 为 model-zoo 的 python 依赖
* dataset 目录中包含模型的 imagenet 数据集预处理，将作为 plugin 被 tpu_perf 调用
* data 目录将用于存放 lmdb 数据集
* output 目录将用于存放编译输出的 bmodel 和一些中间数据
* 其他目录包含各个模型的信息和配置。每个模型对应的目录都有一个 config.yaml 文件，该配置文件中配置了模型的名称、路径和 FLOPs、数据集制作参数，以及模型的量化编译命令。

如果已经克隆过 ``model-zoo`` 可以执行以下命令同步模型到最新状态:

.. code :: shell

   $ cd model-zoo
   $ git pull
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG,*.npz" --exclude=""
   $ cd ../

此过程会从 ``GitHub`` 上下载大量数据。由于具体网络环境的差异, 此过程可能耗时较长。

注意：如果您获得了SOPHGO提供的 ``model-zoo`` 测试包, 可以执行以下操作创建并
设置好 ``model-zoo``，完成此步骤后直接进入下一节。

.. code :: console

    $ mkdir -p model-zoo
    $ tar -xvf path/to/model-zoo_<date>.tar.bz2 --strip-components=1 -C model-zoo

准备运行环境
~~~~~~~~~~~~

安装运行 ``model-zoo`` 所需的依赖：

.. code ::

   # for ubuntu 操作系统
   sudo apt-get install build-essential
   sudo apt install python3-dev
   sudo apt-get install -y libgl1 # For OpenCV
   # for centos 操作系统
   sudo yum install make automake gcc gcc-c++ kernel-devel
   sudo yum install python-devel
   sudo yum install mesa-libGL
   # 精度测试需要执行以下操作，性能测试不执行
   cd path/to/model-zoo
   pip3 install -r requirements.txt

另外，运行环境中调用 tpu 硬件进行性能和精度测试，请根据 libsophon 使用手册安装 libsophon。

准备数据集
~~~~~~~~~~

ImageNet
--------

下载 `imagenet 2012 数据集 <https://image-net.org/challenges/LSVRC/2012/>`_ 的

ILSVRC2012_img_val.tar（MD5 29b22e2961454d5413ddabcf34fc5622）。

将 ILSVRC2012_img_val.tar 解压到 ``dataset/ILSVRC2012/ILSVRC2012_img_val`` 目录中：

.. code :: shell

   $ cd path/to/model-zoo
   $ tar xvf path/to/ILSVRC2012_img_val.tar -C dataset/ILSVRC2012/ILSVRC2012_img_val

COCO (可选)
-----------

如果精度测试用到了 coco 数据集（如yolo等用coco训练的网络），请按照如下步骤下载解压：

.. code :: shell

   cd path/to/model-zoo/dataset/COCO2017/
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip annotations_trainval2017.zip
   unzip val2017.zip

在非 x86 环境运行性能与精度测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

注意: 如果您的设备是 PCIE 板卡, 可以直接跳过该节内容。

性能测试只依赖于 ``libsophon`` 运行环境, 所以在工具链编译环境编译完的模型连同
``model-zoo`` 整个打包, 就可以在 SOC 环境使用 ``tpu_perf`` 进行性能与精度测试。
但是, SOC设备上存储有限, 完整的 ``model-zoo`` 与编译输出内容可能无法完整拷贝到
SOC 中。这里介绍一种通过 linux nfs 远程文件系统挂载来实现在 SOC 设备上运行测试的
方法。

首先, 在工具链环境服务器『host 系统』安装 nfs 服务:

.. code :: shell

   $ sudo apt install nfs-kernel-server

在 ``/etc/exports`` 中添加以下内容(配置共享目录):

.. code ::

   /the/absolute/path/of/model-zoo *(rw,sync,no_subtree_check,no_root_squash)

其中 ``*`` 表示所有人都可以访问该共享目录, 也可以配置成特定网段或 IP 可访问, 如:

.. code ::

   /the/absolute/path/of/model-zoo 192.168.43.0/24(rw,sync,no_subtree_check,no_root_squash)


然后执行如下命令使配置生效:

.. code-block:: shell

   $ sudo exportfs -a
   $ sudo systemctl restart nfs-kernel-server

另外, 需要为 dataset 目录下的图片添加读取权限:

.. code-block:: shell

   chmod -R +r path/to/model-zoo/dataset

在 SOC 设备上安装客户端并挂载该共享目录:

.. code-block:: shell

   $ mkdir model-zoo
   $ sudo apt-get install -y nfs-common
   $ sudo mount -t nfs <IP>:/path/to/model-zoo ./model-zoo

这样便可以在 SOC 环境访问测试目录。SOC 测试其余的操作与 PCIE 基本一致, 请参考下
文进行操作; 运行环境命令执行位置的差别, 已经在执行处添加说明。

获取 ``tpu-perf`` 工具
~~~~~~~~~~~~~~~~~~~~~~

从 https://github.com/sophgo/tpu-perf/releases 地址下载最新的 ``tpu-perf``
wheel安装包。例如: tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl 。并将
``tpu-perf`` 包放置到与 ``model-zoo`` 同一级目录下。此时的目录结构应该为如下形式:

::

   ├── tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl
   ├── tpu-mlir_xxxx.tar.gz
   └── model-zoo

准备工具链编译环境
~~~~~~~~~~~~~~~~~~

建议在 docker 环境使用工具链软件，最新版本的 docker 可以参考 `官方教程 <https://docs.docker.com/engine/install/ubuntu/>`_ 进行安装。安装完成后，执行下面的脚本将当前用户加入 docker 组，获得 docker 执行权限。

.. code :: shell

   $ sudo usermod -aG docker $USER
   $ newgrp docker

然后，在 ``tpu-mlir_xxxx.tar.gz`` 目录下(注意, ``tpu-mlir_xxxx.tar.gz`` 和
``model-zoo`` 需要在同一级目录), 执行以下命令:

.. code :: shell

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ docker pull sophgo/tpuc_dev:v2.2
   $ docker run --rm --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v2.2

运行命令后会处于Docker的容器中。

模型性能和精度测试流程
~~~~~~~~~~~~~~~~~~~~~~

模型编译
---------

使用以下命令完成设置运行测试所需的环境变量:

.. code :: shell

   $ cd tpu-mlir_xxxx
   $ source envsetup.sh

该过程结束后不会有任何提示。之后使用以下命令安装 ``tpu-perf``:

.. code :: shell

   $ pip3 install ../tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl

``model-zoo`` 的相关 ``confg.yaml`` 配置了SDK的测试内容。例如: resnet18的
配置文件为 ``model-zoo/vision/classification/resnet18-v2/config.yaml`` 。

执行以下命令, 运行全部测试样例:

.. code :: shell

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --target BM1684X --mlir -l full_cases.txt

``--target`` 用于指定芯片型号，目前支持 ``BM1684`` 和 ``BM1684X`` 。

此时会编译以下模型（由于model-zoo的模型在持续添加中，这里只列出部分模型；同时该
过程也编译了用于测试精度的模型，后续精度测试部分无需再编译模型。）:

::

   * efficientnet-lite4
   * mobilenet_v2
   * resnet18
   * resnet50_v2
   * shufflenet_v2
   * squeezenet1.0
   * vgg16
   * yolov5s
   * ...

命令正常结束后, 会看到新生成的 ``output`` 文件夹(测试输出内容都在该文件夹中)。
修改 ``output`` 文件夹的属性, 以保证其可以被Docker外系统访问。

.. code :: shell

   $ chmod -R a+rw output

性能测试
---------

运行测试需要在 Docker 外面的环境(此处假设您已经安装并配置好了1684X设备和
驱动)中进行, 可以退出 Docker 环境:

.. code :: shell

   $ exit

1. PCIE 板卡下运行以下命令, 测试生成的 ``bmodel`` 性能。

.. code :: shell

   $ pip3 install ./tpu_perf-*-py3-none-manylinux2014_x86_64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --target BM1684X --mlir -l full_cases.txt

``--target`` 用于指定芯片型号，目前支持 ``BM1684`` 和 ``BM1684X`` 。

注意：如果主机上安装了多块SOPHGO的加速卡，可以在使用 ``tpu_perf`` 的时候，通过添加
``--devices id`` 来指定 ``tpu_perf`` 的运行设备。如：

.. code :: shell

   $ python3 -m tpu_perf.run --target BM1684X --devices 2 --mlir -l full_cases.txt


2. SOC 设备使用以下步骤, 测试生成的 ``bmodel`` 性能。

从 https://github.com/sophgo/tpu-perf/releases 地址下载最新的 ``tpu-perf``
``tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl`` 文件到SOC设备上并执行
以下操作:

.. code :: shell

   $ pip3 install ./tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --target BM1684X --mlir -l full_cases.txt

运行结束后, 性能数据在 ``output/stats.csv`` 中可以获得。该文件中记录了相关模型的
运行时间、计算资源利用率和带宽利用率。

精度测试
---------

运行测试需要在 Docker 外面的环境(此处假设您已经安装并配置好了1684X设备和
驱动)中进行, 可以退出 Docker 环境:

.. code :: shell

   $ exit

PCIE 板卡下运行以下命令, 测试生成的 ``bmodel`` 精度。

.. code :: shell

   $ pip3 install ./tpu_perf-*-py3-none-manylinux2014_x86_64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.precision_benchmark --target BM1684X --mlir -l full_cases.txt

``--target`` 用于指定芯片型号，目前支持 ``BM1684`` 和 ``BM1684X`` 。

各类精度数据在 output 目录中的各个 csv 文件可以获得。

注意：如果主机上安装了多块SOPHGO的加速卡，可以在使用 ``tpu_perf`` 的时候，通过添加
``--devices id`` 来指定 ``tpu_perf`` 的运行设备。如：

.. code :: shell

   $ python3 -m tpu_perf.precision_benchmark --target BM1684X --devices 2 --mlir -l full_cases.txt

具体参数说明可以通过以下命令获得：

.. code :: shell

  python3 -m tpu_perf.precision_benchmark --help

FAQ
~~~~

此章节列出一些tpu_perf安装、使用中可能会遇到的问题及解决办法。

invalid command 'bdist_wheel'
-----------------------------
tpu_perf编译之后安装，如提示如下图错误，由于没有安装wheel工具导致。

.. image :: ../assets/invalid-bdist_wheel.png

则先运行：

.. code :: shell

   pip3 install wheel

再安装whl包

not a supported wheel
---------------------
tpu_perf编译之后安装，如提示如下图错误，由于pip版本导致。

.. image :: ../assets/not-support-wheel.png

则先运行：

.. code :: shell

   pip3 install --upgrade pip

再安装whl包

no module named 'xxx'
---------------------

安装运行model-zoo所需的依赖时，如提示如下图错误，由于pip版本导致。

.. image :: ../assets/no-module-named-skbuild.png

则先运行：

.. code :: shell

   pip3 install --upgrade pip

再安装运行 model-zoo 所需的依赖

精度测试因为内存不足被kill
--------------------------
对于YOLO系列的模型精度测试，可能需要4G左右的内存空间。SOC环境如果存在内存不足被kill的情况，可以参考SOPHON
BSP 开发手册的板卡预制内存布局章节扩大内存。
