Environment Setup
=================

This chapter describes the development environment configuration. The code is compiled and run in docker.

.. _code_load:

Code Download
----------------

Github link: https://github.com/sophgo/tpu-mlir

After cloning this code, it needs to be compiled in docker. For specific steps, please refer to the following.

.. _env_setup:

Docker Configuration
--------------------

TPU-MLIR is developed in the Docker environment, and it can be compiled and run after Docker is configured.

Download the required image from DockerHub https://hub.docker.com/r/sophgo/tpuc_dev :


.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.1


If you are using docker for the first time, you can execute the following commands to install and configure it (only for the first time):


.. _docker configuration:

.. code-block:: shell
   :linenos:

   $ sudo apt install docker.io
   $ sudo systemctl start docker
   $ sudo systemctl enable docker
   $ sudo groupadd docker
   $ sudo usermod -aG docker $USER
   $ newgrp docker


Make sure the installation package is in the current directory, and then create a container in the current directory as follows:

.. code-block:: shell

  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.1
  # "myname" is just an example, you can use any name you want

Note that the path of the TPU-MLIR project in docker should be /workspace/tpu-mlir

.. _model_zoo:

ModelZoo (Optional)
-------------------

TPU-MLIR comes with the yolov5s model. If you want to run other models, you need to download them from ModelZoo. The path is as follows:

https://github.com/sophgo/model-zoo

After downloading, put it in the same directory as tpu-mlir. The path in docker should be /workspace/model-zoo

.. _compiler :

Compilation
----------------

In the docker container, the code is compiled as follows:

.. code-block:: shell

   $ cd tpu-mlir
   $ source ./envsetup.sh
   $ ./build.sh

Regression validation:

.. code-block:: shell

   # This project contains the yolov5s.onnx model, which can be used directly for validation
   $ pushd regression
   $ python run_model.py yolov5s
   $ popd

You can validate more networks with model-zoo, but the whole regression takes a long time:

.. code-block:: shell

   # The running time is very long, so it is not necessary
   $ pushd regression
   $ ./run_all.sh
   $ popd
