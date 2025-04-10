开发环境配置
============

本章介绍开发环境配置, 代码在Docker中编译和运行。

.. _code_load:

代码下载
----------------

代码路径: https://github.com/sophgo/tpu-mlir

克隆该代码后, 需要在Docker中编译。参考下文配置Docker。

.. _env_setup:

Docker配置
----------------

TPU-MLIR在Docker环境开发, 配置好Docker就可以编译和运行了。

从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像:


.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.4


若下载失败，可从官网开发资料 https://developer.sophgo.com/site/index/material/86/all.html 下载所需镜像文件，或使用下方命令下载镜像：

.. code-block:: shell
   :linenos:

   $ wget https://sophon-assets.sophon.cn/sophon-prod-s3/drive/25/01/20/10/sophgo-tpuc_dev_v3.3.tar.gz
   $ docker load -i sophgo-tpuc_dev_v3.3.tar.gz


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


确保安装包在当前目录, 然后在当前目录创建容器如下:


.. code-block:: shell

  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4
  # myname只是举个名字的例子, 请指定成自己想要的容器的名字
  # 使用 --privileged 参数以获取root权限，如果不需要root权限，请删除该参数

注意TPU-MLIR工程在Docker中的路径应该是/workspace/tpu-mlir

.. _model_zoo:

ModelZoo(可选)
----------------

TPU-MLIR中自带yolov5s模型, 如果要跑其他模型, 需要下载ModelZoo, 路径如下:

https://github.com/sophgo/model-zoo

下载后放在与tpu-mlir同级目录, 在Docker中的路径应该是/workspace/model-zoo

.. _compiler :

代码编译
----------------

在Docker的容器中, 代码编译方式如下:

.. code-block:: shell

   $ cd tpu-mlir
   $ source ./envsetup.sh
   $ ./build.sh

回归验证, 如下:

.. code-block:: shell

   # 本工程包含yolov5s.onnx模型, 可以直接用来验证
   $ pushd regression
   $ python run_model.py yolov5s
   $ popd

如果要验证更多网络, 需要依赖model-zoo, 回归时间比较久。

操作如下: (可选)

.. code-block:: shell

   # 执行时间很长, 该步骤也可以跳过
   $ pushd regression
   $ ./run_all.sh
   $ popd
