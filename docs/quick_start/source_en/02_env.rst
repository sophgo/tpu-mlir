Environment Setup
=================

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

.. _docker container_setup:

Make sure the installation package is in the current directory, and then create a container in the current directory as follows:


.. code-block:: shell

  $ docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.1
  # "myname" is just an example, you can use any name you want

Subsequent chapters assume that the user is already in the /workspace directory inside docker.
