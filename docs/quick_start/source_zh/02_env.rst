开发环境配置
============
首先检查当前系统环境是否满足Ubuntu 22.04和Python 3.10。如不满足，请进行下一节 :ref:`基础环境配置 <env setup>` ；如满足，直接跳至 :ref:`TPU-MLIR 安装 <install tpu_mlir>` 。

.. _env setup:

基础环境配置
------------------
如不满足上述系统环境，则需要使用Docker，从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像文件，或使用下方命令直接拉取镜像：


.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.4

若下载失败，可从官网开发资料 https://developer.sophgo.com/site/index/material/86/all.html 下载所需镜像文件，或使用下方命令下载镜像：

.. code-block:: shell
   :linenos:

   $ wget https://sophon-assets.sophon.cn/sophon-prod-s3/drive/25/04/15/16/tpuc_dev_v3.4.tar.gz
   $ docker load -i tpuc_dev_v3.4.tar.gz


如果是首次使用Docker，可执行下述命令进行安装和配置（仅首次执行）：


.. _docker configuration:

.. code-block:: shell
   :linenos:

   $ sudo apt install docker.io
   $ sudo systemctl start docker
   $ sudo systemctl enable docker
   $ sudo groupadd docker
   $ sudo usermod -aG docker $USER
   $ newgrp docker


.. _docker container_setup:

若下载镜像文件，则需要确保镜像文件在当前目录，并在当前目录创建容器如下:

.. code-block:: shell

  # 使用 --privileged 参数以获取root权限，如果不需要root权限，请删除该参数
  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4

其中， ``myname`` 为容器名称，可以自定义； ``$PWD`` 为当前目录，与容器的 ``/workspace`` 目录同步。

后文假定用户已经处于 docker 里面的 ``/workspace`` 目录。


.. _install tpu_mlir:

安装TPU-MLIR
------------------

目前支持2种安装方法，分别是在线安装和离线安装。

**在线安装**

直接从pypi下载并安装，默认安装最新版：

.. code-block:: shell

   $ pip install tpu_mlir

**离线安装**

从Github的 `Assets <https://github.com/sophgo/tpu-mlir/releases/>`_ 处下载最新的 `tpu_mlir-*-py3-none-any.whl`，然后使用pip安装:


.. code-block:: shell

   $ pip install tpu_mlir-*-py3-none-any.whl


安装TPU-MLIR依赖
------------------

TPU-MLIR在对不同框架模型处理时所需的依赖不同，在线安装和离线安装方式都需要安装额外依赖。

**在线安装**

在线安装方式对于 ``onnx`` 或 ``torch`` 生成的模型文件，可使用下方命令安装额外的依赖环境:

.. code-block:: shell

   # 安装onnx依赖
   $ pip install tpu_mlir[onnx]
   # 安装torch依赖
   $ pip install tpu_mlir[torch]

目前支持5种配置:

.. code-block:: shell

   onnx, torch, tensorflow, caffe, paddle

可使用一条命令安装多个配置，也可直接安装全部依赖环境:

.. code-block:: shell

   # 同时安装onnx, torch, caffe依赖
   $ pip install tpu_mlir[onnx,torch,caffe]
   # 安装全部依赖
   $ pip install tpu_mlir[all]

**离线安装**

同理，离线安装方式可使用下方命令安装额外的依赖环境：

.. code-block:: shell

   # 安装onnx依赖
   $ pip install tpu_mlir-*-py3-none-any.whl[onnx]
   # 安装全部依赖
   $ pip install tpu_mlir-*-py3-none-any.whl[all]

