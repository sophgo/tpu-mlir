Test SDK release package with TPU-PERF
======================================


Configure the system environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using Docker for the first time, use the methods in :ref:`First-time Docker <docker configuration>` to install and configure Docker. At the same time, ``git-lfs`` will be used in this chapter. If you use ``git-lfs`` for the first time, you can execute the following commands for installation and configuration (only for the first time, and the configuration is in the user's own system, not in Docker container):

.. code-block:: shell
   :linenos:

   $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   $ sudo apt-get install git-lfs


Get the ``model-zoo`` model [#extra]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same directory of ``tpu-mlir_xxxx.tar.gz`` (tpu-mlir's release package), use the following command to clone the ``model-zoo`` project:

.. code-block:: shell
   :linenos:

   $ git clone --depth=1 https://github.com/sophgo/model-zoo
   $ cd model-zoo
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

If you have cloned ``model-zoo``, you can execute the following command to synchronize the model to the latest state:

.. code-block:: shell
   :linenos:

   $ cd model-zoo
   $ git pull
   $ git lfs pull --include "*.onnx,*.jpg,*.JPEG" --exclude=""
   $ cd ../

This process downloads a large amount of data from ``GitHub``. Due to differences in specific network environments, this process may take a long time.

.. rubric:: Footnotes

.. [#extra] If you get the ``model-zoo`` test package provided by SOPHGO, you can
   do the following to create and set up the ``model-zoo``. After completing
   this step, go directly to the next section :ref:`get tpu-perf`.

   .. code :: console

      $ mkdir -p model-zoo
      $ tar -xvf path/to/model-zoo_<date>.tar.bz2 --strip-components=1 -C model-zoo


.. _get tpu-perf:

Get the ``tpu-perf`` tool
~~~~~~~~~~~~~~~~~~~~~~~~~

Download the latest ``tpu-perf`` wheel installation package from https://github.com/sophgo/tpu-perf/releases. For example, tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl. And put the ``tpu-perf`` package in the same directory as ``model-zoo``. The directory structure at this point should look like this:


::

   ├── tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl
   ├── tpu-mlir_xxxx.tar.gz
   └── model-zoo


Test process
~~~~~~~~~~~~

Unzip the SDK and create a Docker container
+++++++++++++++++++++++++++++++++++++++++++

Execute the following command in the ``tpu-mlir_xxxx.tar.gz`` directory (note that ``tpu-mlir_xxxx.tar.gz`` and
``model-zoo`` needs to be at the same level):

.. code-block:: shell
   :linenos:

   $ tar zxf tpu-mlir_xxxx.tar.gz
   $ docker pull sophgo/tpuc_dev:latest
   $ docker run --rm --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest

After running the command, it will be in a Docker container.


Set environment variables and install ``tpu-perf``
++++++++++++++++++++++++++++++++++++++++++++++++++

Complete setting the environment variables needed to run the tests with the following command:

.. code-block:: shell
   :linenos:

   $ cd tpu-mlir_xxxx
   $ source envsetup.sh

There will be no prompts after the process ends. Then install ``tpu-perf`` with the following command:

.. code-block:: shell

   $ pip3 install ../tpu_perf-x.x.x-py3-none-manylinux2014_x86_64.whl


.. _test_main:

Run the test
++++++++++++

Compile the model
``````````````````

``confg.yaml`` in ``model-zoo`` configures the test content of the SDK. For example, the configuration file for resnet18 is ``model-zoo/vision/classification/resnet18-v2/config.yaml`` .

Execute the following command to run all test samples:

.. code-block:: shell
   :linenos:

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --mlir -l full_cases.txt

The following models are compiled:

::

   * efficientnet-lite4
   * mobilenet_v2
   * resnet18
   * resnet50_v2
   * shufflenet_v2
   * squeezenet1.0
   * vgg16
   * yolov5s


After the command is finished, you will see the newly generated ``output`` folder (where the test output is located).
Modify the properties of the ``output`` folder to make it accessible to systems outside of Docker.


.. code-block:: shell
   :linenos:

   $ chmod -R a+rw output


Test model performance
````````````````````````

Configure SOC device
++++++++++++++++++++++

Note: If your device is a PCIE board, you can skip this section directly.

The performance test only depends on the ``libsophon`` runtime environment, so after packaging models, compiled in the toolchain compilation environment, and ``model-zoo``, the performance test can be carried out in the SOC environment by ``tpu_perf``. However, the complete ``model-zoo`` as well as compiled output contents may not be fully copied to the SOC since the storage on the SOC device is limited. Here is a method to run tests on SOC devices through linux nfs remote file system mounts.

First, install the nfs service on the toolchain environment server "host system":

.. code-block:: shell

   $ sudo apt install nfs-kernel-server

Add the following content to ``/etc/exports`` (configure the shared directory):

.. code ::

   /the/absolute/path/of/model-zoo *(rw,sync,no_subtree_check,no_root_squash)

Where ``*`` means that everyone can access the shared directory. Moreover, it
can be configured to be accessible by a specific network segment or IP, such as:

.. code ::

   /the/absolute/path/of/model-zoo 192.168.43.0/24(rw,sync,no_subtree_check,no_root_squash)

Then execute the following command to make the configuration take effect:

.. code-block:: shell

   $ sudo exportfs -a
   $ sudo systemctl restart nfs-kernel-server

In addition, you need to add read permissions to the images in the dataset directory:

.. code-block:: shell

   chmod -R +r path/to/model-zoo/dataset

Install the client on the SOC device and mount the shared directory:

.. code-block:: shell

   $ mkdir model-zoo
   $ sudo apt-get install -y nfs-common
   $ sudo mount -t nfs <IP>:/path/to/model-zoo ./model-zoo

In this way, the test directory is accessible in the SOC environment. The rest of the SOC test operation is basically the same as that of PCIE. Please refer to the following content for operation. The difference in command execution position and operating environment has been explained in the execution place.


Run the test
+++++++++++++

Running the test needs to be done in an environment outside Docker (it is assumed that you have installed and configured the 1684X device and driver), so you can exit the Docker environment:

.. code :: console

   $ exit

1. Run the following commands under the PCIE board to test the performance of the generated ``bmodel``.

.. code-block:: shell
   :linenos:

   $ pip3 install ./tpu_perf-*-py3-none-manylinux2014_x86_64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --mlir -l full_cases.txt

2. The SOC device uses the following steps to test the performance of the generated ``bmodel``.

Download the latest ``tpu-perf``, ``tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl``, from https://github.com/sophgo/tpu-perf/releases to the SOC device and execute the following operations:

.. code-block:: shell
   :linenos:

   $ pip3 install ./tpu_perf-x.x.x-py3-none-manylinux2014_aarch64.whl
   $ cd model-zoo
   $ python3 -m tpu_perf.run --mlir -l full_cases.txt


After that, performance data is available in ``output/stats.csv``, in which the running time, computing resource utilization, and bandwidth utilization of the relevant models are recorded.
