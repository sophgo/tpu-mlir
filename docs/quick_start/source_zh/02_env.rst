开发环境配置
============
首先检查当前系统环境是否满足ubuntu 22.04和python 3.10。如不满足，请进行下一节*基础环境配置*；如满足，直接跳至*tpu_mlir安装*。


基础环境配置
------------------
如不满足，从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像:


.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.1


如果是首次使用Docker, 可执行下述命令进行安装和配置(仅首次执行):


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

确保安装包在当前目录, 然后在当前目录创建容器如下:

.. code-block:: shell

  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.1
  # myname只是举个名字的例子, 请指定成自己想要的容器的名字

后文假定用户已经处于docker里面的/workspace目录。


tpu_mlir安装
------------------
使用pip安装tpu_mlir [#whl安装]_:


.. code-block:: shell

   $ pip install tpu_mlir


tpu_mlir在对不同框架模型处理时所需的依赖不同，对于onnx或torch生成的模型文件，使用下面命令安装额外的依赖环境:


.. code-block:: shell

   $ pip install tpu_mlir[onnx]

.. code-block:: shell

   $ pip install tpu_mlir[torch]

目前支持5中配置:
*onnx*, *torch*, *tensorflow*, *caffe*, *paddle*。
或者可使用下面命令安装全部依赖环境:

.. code-block:: shell

   $ pip install tpu_mlir[all]


.. [#whl安装] 当本地有tpu_mlir-{version}.whl文件时，也可以使用以下命令进行安装。
   
   .. code :: console

      $ pip install path/to/tpu_mlir-{version}.whl[all]