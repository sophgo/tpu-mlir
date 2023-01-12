使用TPU-PERF测试SDK发布包
=========================


配置系统环境
~~~~~~~~~~~~

如果是首次使用Docker, 那么请使用 :ref:`首次使用Docker <docker configuration>` 中的方法安装
并配置Docker。同时, 本章中会使用到 ``git-lfs`` , 如果首次使用 ``git-lfs`` 可执行下述命
令进行安装和配置(仅首次执行, 同时该配置是在用户自己系统中, 并非Docker container中):

.. code-block:: shell
   :linenos:

   $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   $ sudo apt-get install git-lfs


获取 ``model-zoo`` 模型 [#extra]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``tpu-mlir_xxxx.tar.gz`` (tpu-mlir的发布包)的同级目录下, 使用以下命令克隆 ``model-zoo`` 工程:

.. code-block:: shell
   :linenos:

   $ git clone --depth=1 https://github.com/sophgo/model-zoo
   $ cd model-zoo
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

如果已经克隆过 ``model-zoo`` 可以执行以下命令同步模型到最新状态:

.. code-block:: shell
   :linenos:

   $ cd model-zoo
   $ git pull
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

此过程会从 ``GitHub`` 上下载大量数据。由于具体网络环境的差异, 此过程可能耗时较长。

.. rubric:: Footnotes

.. [#extra] 如果您获得了SOPHGO提供的 ``model-zoo`` 测试包, 可以执行以下操作创建并
   设置好 ``model-zoo``。完成此步骤后直接进入下一节 :ref:`get tpu-perf`。

   .. code :: console

      $ mkdir -p model-zoo
      $ tar -xvf path/to/model-zoo_<date>.tar.bz2 --strip-components=1 -C model-zoo


.. _get tpu-perf:

获取 ``tpu-perf`` 工具
~~~~~~~~~~~~~~~~~~~~~~

从 https://github.com/sophgo/tpu-perf/releases 地址下载最新的 ``tpu-perf``
wheel安装包。例如: tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl 。并将
``tpu-perf`` 包放置到与 ``model-zoo`` 同一级目录下。此时的目录结构应该为如下形式:


::

   ├── tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl
   ├── tpu-mlir_xxxx.tar.gz
   └── model-zoo


测试流程
~~~~~~~~

解压SDK并创建Docker容器
+++++++++++++++++++++++

在 ``tpu-mlir_xxxx.tar.gz`` 目录下(注意, ``tpu-mlir_xxxx.tar.gz`` 和
``model-zoo`` 需要在同一级目录), 执行以下命令:

.. code-block:: shell
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ docker pull sophgo/tpuc_dev:latest
   $ docker run --rm --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest

运行命令后会处于Docker的容器中。


设置环境变量并安装 ``tpu-perf``
+++++++++++++++++++++++++++++++

使用以下命令完成设置运行测试所需的环境变量:

.. code-block:: shell
   :linenos:

   $ cd tpu-mlir_xxxx
   $ source envsetup.sh

该过程结束后不会有任何提示。之后使用以下命令安装 ``tpu-perf``:

.. code-block:: shell

   $ pip3 install ../tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl


.. _test_main:

运行测试
++++++++

编译模型
````````

``model-zoo`` 的相关 ``confg.yaml`` 配置了SDK的测试内容。例如: resnet18的
配置文件为 ``model-zoo/vision/classification/resnet18-v2/config.yaml`` 。

执行以下命令, 运行全部测试样例:

.. code-block:: shell
   :linenos:

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --mlir -l full_cases.txt

此时会编译以下模型:

::

   * efficientnet-lite4
   * mobilenet_v2
   * resnet18
   * resnet50_v2
   * shufflenet_v2
   * squeezenet1.0
   * vgg16
   * yolov5s


命令正常结束后, 会看到新生成的 ``output`` 文件夹(测试输出内容都在该文件夹中)。
修改 ``output`` 文件夹的属性, 以保证其可以被Docker外系统访问。


.. code-block:: shell
   :linenos:

   $ chmod -R a+rw output


测试模型性能
````````````

配置SOC设备
+++++++++++

注意: 如果您的设备是 PCIE 板卡, 可以直接跳过该节内容。

性能测试只依赖于 ``libsophon`` 运行环境, 所以在工具链编译环境编译完的模型连同
``model-zoo`` 整个打包, 就可以在 SOC 环境使用 ``tpu_perf`` 进行性能与精度测试。
但是, SOC设备上存储有限, 完整的 ``model-zoo`` 与编译输出内容可能无法完整拷贝到
SOC 中。这里介绍一种通过 linux nfs 远程文件系统挂载来实现在 SOC 设备上运行测试的
方法。

首先, 在工具链环境服务器『host 系统』安装 nfs 服务:

.. code-block:: shell

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


运行测试
++++++++

运行测试需要在 Docker 外面的环境(此处假设您已经安装并配置好了1684X设备和
驱动)中进行, 可以退出 Docker 环境:

.. code :: console

   $ exit

1. PCIE 板卡下运行以下命令, 测试生成的 ``bmodel`` 性能。

.. code-block:: shell
   :linenos:

   $ pip3 install ./tpu_perf-*-py3-none-manylinux2014_x86_64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --mlir -l full_cases.txt

注意：如果主机上安装了多块SOPHGO的加速卡，可以在使用 ``tpu_perf`` 的时候，通过添加
``--devices id`` 来指定 ``tpu_perf`` 的运行设备。如：

.. code-block:: shell

   $ python3 -m tpu_perf.run --devices 2 --mlir -l full_cases.txt


2. SOC 设备使用以下步骤, 测试生成的 ``bmodel`` 性能。

从 https://github.com/sophgo/tpu-perf/releases 地址下载最新的 ``tpu-perf``
``tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl`` 文件到SOC设备上并执行
以下操作:

.. code-block:: shell
   :linenos:

   $ pip3 install ./tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --mlir -l full_cases.txt


运行结束后, 性能数据在 ``output/stats.csv`` 中可以获得。该文件中记录了相关模型的
运行时间、计算资源利用率和带宽利用率。
