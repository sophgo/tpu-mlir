开发环境配置
============

从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像:


.. code-block:: console

   $ docker pull sophgo/tpuc_dev:latest


如果是首次使用Docker，可执行下述命令进行安装和配置（仅首次执行）:


.. _docker configuration:

.. code-block:: console
   :linenos:

   $ sudo apt install docker.io
   $ sudo systemctl start docker
   $ sudo systemctl enable docker
   $ sudo groupadd docker
   $ sudo usermod -aG docker $USER
   $ newgrp docker


.. _docker container_setup:

确保安装包在当前目录，然后在当前目录创建容器如下：

.. code-block:: console

  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
  # myname只是举个名字的例子，请指定成自己想要的容器的名字

后文假定用户已经处于docker里面的/workspace目录。
