Appendix.04: Model-zoo test
===================================================

Test Notification
~~~~~~~~~~~~~~~~~~~~~

If the test time exceeds the following time limit, it will be considered abnormal:

* Compilation test: 48 hours
* Performance test: 24 hours
* Accuracy test: 24 hours (currently only BM1684X PCIE needs to perform accuracy test)

Configure the system environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using Docker for the first time, use the methods in :ref:`Environment Setup <docker configuration>` to install and configure Docker. At the same time, ``git-lfs`` will be used in this chapter. If you use ``git-lfs`` for the first time, you need execute the following commands for installation and configuration in the user's own system (not in Docker container).

.. code-block:: shell

   $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   $ sudo apt-get install git-lfs


Get the ``model-zoo`` model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your working directory, get the ``model-zoo`` test package from the SDK package provided by SOPHGO, then create and set up ``model-zoo`` as follows:

.. code-block:: shell

    $ mkdir -p model-zoo
    $ tar -xvf path/to/model-zoo_<date>.tar.bz2 --strip-components=1 -C model-zoo

The directory structure of model-zoo is as follows:

.. code-block:: shell

    ├── config.yaml
    ├── requirements.txt
    ├── dataset
    ├── harness
    ├── output
    └── ...

* config.yaml: contains generic configuration: dataset directory, model root directory, etc., as well as some reused parameters and commands
* requirements.txt: contains python dependencies for model-zoo.
* dataset: directory contains the datasets of the models in modelzoo, which will be called by tpu_perf as plugins.
* output: directory will be used to store the compiled output bmodel and some intermediate data.
* The other directories contain information and configuration for each model. The directory corresponding to each model has a config.yaml file, which configures the model's name, path and FLOPs, dataset production parameters, and the model's quantization compilation commands.


Prepare the runtime environment
~~~~~~~~~~~~

Install the dependencies needed to run ``model-zoo`` on your system (outside of the Docker container):

.. code-block:: shell

   # for ubuntu operating system
   $ sudo apt install build-essential
   $ sudo apt install python3-dev
   $ sudo apt install -y libgl1
   # for centos operating system
   $ sudo yum install make automake gcc gcc-c++ kernel-devel
   $ sudo yum install python-devel
   $ sudo yum install mesa-libGL
   # accuracy tests require the following operations to be performed, performance tests can be performed without, it is recommended to use Anaconda to create a virtual environment of python 3.7 or above
   $ cd path/to/model-zoo
   $ pip3 install -r requirements.txt

In addition, tpu hardware needs to be invoked for performance and accuracy tests, so please install the runtime environment for the TPU hardware.


Configure SoC device
~~~~~~~~~~~~~~~~~~

Note: If your device is a PCIE board, you can skip this section directly.

The performance test only depends on the runtime environment for the TPU hardware, so after packaging models, compiled in the toolchain compilation environment, and ``model-zoo``, the performance test can be carried out in the SoC environment by ``tpu_perf``. However, the complete ``model-zoo`` as well as compiled output contents may not be fully copied to the SoC since the storage on the SoC device is limited. Here is a method to run tests on SoC devices through linux nfs remote file system mounts.

First, install the nfs service on the toolchain environment server "host system":

.. code-block:: shell

   $ sudo apt install nfs-kernel-server

Add the following content to ``/etc/exports`` (configure the shared directory):

.. code-block:: shell

   /the/absolute/path/of/model-zoo *(rw,sync,no_subtree_check,no_root_squash)

Where ``*`` means that everyone can access the shared directory. Moreover, it
can be configured to be accessible by a specific network segment or IP, such as:

.. code-block:: shell

   /the/absolute/path/of/model-zoo 192.168.43.0/24(rw,sync,no_subtree_check,no_root_squash)

Then execute the following command to make the configuration take effect:

.. code-block:: shell

   $ sudo exportfs -a
   $ sudo systemctl restart nfs-kernel-server

In addition, you need to add read permissions to the images in the dataset directory:

.. code-block:: shell

   $ chmod -R +r path/to/model-zoo/dataset

Install the client on the SoC device and mount the shared directory:

.. code-block:: shell

   $ mkdir model-zoo
   $ sudo apt-get install -y nfs-common
   $ sudo mount -t nfs <IP>:/path/to/model-zoo ./model-zoo

In this way, the test directory is accessible in the SoC environment. The rest of the SoC test operation is basically the same as that of PCIE. Please refer to the following content for operation. The difference in command execution position and operating environment has been explained in the execution place.


Prepare dataset
~~~~~~~~~~~~~~~~~~

Note: Due to the limited CPU resources of SoC devices, precision testing is not recommended. Therefore, the SoC device test can skip the data set preparation and precision test part.

ImageNet
--------

Download `ImageNet 2012 Dataset <https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC>`_ 。

After unzipping, move the data under ``Data/CLS_LOC/val`` to a directory like model-zoo:

.. code-block:: shell

   $ cd path/to/sophon/model-zoo
   $ mkdir -p dataset/ILSVRC2012/ILSVRC2012_img_val
   $ mv path/to/imagenet-object-localization-challenge/Data/CLS_LOC/val dataset/ILSVRC2012/ILSVRC2012_img_val
   # It is also possible to map the dataset directory to dataset/ILSVRC2012/ILSVRC2012_img_val through the soft link ln -s


COCO (optional)
-----------

If the precision test uses the coco dataset (networks trained with coco such as yolo), please download and unzip it as follows:

.. code-block:: shell

   $ cd path/to/model-zoo/dataset/COCO2017/
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   $ wget http://images.cocodataset.org/zips/val2017.zip
   $ unzip annotations_trainval2017.zip
   $ unzip val2017.zip


Vid4 (optional)
-----------

If you need precision test on BasicVSR, please download and unzip the Vid4 dataset as follows:

.. code-block:: shell

   $ pip3 install gdown
   $ cd path/to/model-zoo/dataset/basicvsr/
   $ gdown https://drive.google.com/open?id=1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW --fuzzy
   $ unzip -o Vid4.zip -d eval


Prepare the toolchain compilation environment
~~~~~~~~~~~~~~~~~~

It is recommended to use the toolchain software in a docker environment, see :ref:`Base environment configuration <docker configuration>` to install Docker. and execute the following commands in your working directory (the directory which ``model-zoo`` is located) to create a Docker container:

.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.4
   $ docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4

If you want to keep the container after it exits, simply remove the ``--rm`` parameter:

.. code-block:: shell

   $ docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4 --rm

After running the command, it will be in a Docker container. You can the latest ``tpu-mlir`` wheel installation package from the SDK package provided by SOPHGO, such as ``tpu_mlir-*-py3-none-any.whl``. Install tpu_mlir in the Docker container:

.. code-block:: shell

   $ pip install tpu_mlir-*-py3-none-any.whl[all]


.. _test_main:

Model performance and accuracy testing process
~~~~~~~~~~~~~~~~~~~~~~

Compile the model
---------

The model compilation process needs to be done within Docker, where ``tpu_mlir`` need to be installed as described above.

``confg.yaml`` in ``model-zoo`` configures the test content of the SDK. For example, the configuration file for resnet18 is ``model-zoo/vision/classification/resnet18-v2/mlir.config.yaml`` .

Execute the following command to compile the ``resnet18-v2`` model:

.. code-block:: shell

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --target BM1684X --mlir vision/classification/resnet18-v2/mlir.config.yaml

where the ``--target`` is used to specify the processor model, which currently supports ``BM1684`` , ``BM1684X`` , ``BM1688`` , ``BM1690`` and ``CV186X`` .

Execute the following command to compile all the high priority test samples:

.. code-block:: shell

   $ cd ../model-zoo
   $ python3 -m tpu_perf.build --target BM1684X --mlir -l full_cases.txt --priority_filter high

Full compilation may require reserving more than 2T of space, please adjust according to actual conditions. The ``--clear_if_success`` parameter can be used to delete intermediate files after successful compilation, saving space.

The following high priority models will be compiled (Due to continuous additions of models in the model-zoo, only a partial list of models is provided here):

.. code-block:: shell

   * efficientnet-lite4
   * mobilenetv2
   * resnet18-v2
   * resnet50-v2
   * shufflenet_v2
   * squeezenet1.0
   * vgg16
   * yolov5s
   * ...


After the command is finished, you will see the newly generated ``output`` folder. This compilation result can be used for performance and accuracy testing without recompilation. But you need modify the properties of the ``output`` folder to make it accessible to systems outside of Docker:

.. code-block:: shell

   $ chmod -R a+rw output


Performance test
---------

Running the test needs to be done in an environment outside Docker, it is assumed that you have installed and configured the runtime environment for the TPU hardware, so you can exit the Docker environment:

.. code-block:: shell

   $ exit

**PCIE board**

Run the following commands under the PCIE board to test the performance of the generated high priority model ``bmodel`` :

.. code-block:: shell

   $ cd model-zoo
   $ python3 -m tpu_perf.run --target BM1684X --mlir -l full_cases.txt --priority_filter high

where the ``--target`` is used to specify the processor model, which currently supports ``BM1684`` , ``BM1684X`` , ``BM1688`` , ``BM1690`` and ``CV186X`` .

Note: If multiple SOPHGO accelerator cards are installed on the host, you can
specify the running device of ``tpu_perf`` by adding ``--devices id`` when using
``tpu_perf``. Such as:

.. code-block:: shell

   $ python3 -m tpu_perf.run --target BM1684X --devices 2 --mlir -l full_cases.txt --priority_filter high

**SoC device**

The SoC device uses the following steps to test the performance of the generated high priority model ``bmodel``.

.. code-block:: shell

   $ cd model-zoo
   $ python3 -m tpu_perf.run --target BM1684X --mlir -l full_cases.txt --priority_filter high

**Output results**

After that, performance data is available in ``output/stats.csv``, in which the running time, computing resource utilization, and bandwidth utilization of the relevant models are recorded. The performance test results for ``resnet18-v2`` as follows:

.. code-block:: shell

   name,prec,shape,gops,time(ms),mac_utilization,ddr_utilization,processor_usage
   resnet18-v2,FP32,1x3x224x224,3.636,6.800,26.73%,10.83%,3.00%
   resnet18-v2,FP16,1x3x224x224,3.636,1.231,18.46%,29.65%,2.00%
   resnet18-v2,INT8,1x3x224x224,3.636,0.552,20.59%,33.20%,3.00%
   resnet18-v2,FP32,4x3x224x224,14.542,26.023,27.94%,3.30%,3.00%
   resnet18-v2,FP16,4x3x224x224,14.542,3.278,27.73%,13.01%,2.00%
   resnet18-v2,INT8,4x3x224x224,14.542,1.353,33.59%,15.46%,2.00%


Precision test
---------

Note: Due to the limited CPU resources of SoC devices, precision testing is not recommended. Therefore, the SoC device test can skip the precision test part.

Running the test needs to be done in an environment outside Docker, it is assumed that you have installed and configured the runtime environment for the TPU hardware, so you can exit the Docker environment:

.. code-block:: shell

   $ exit

Run the following commands under the PCIE board to test the precision of the generated high priority model ``bmodel`` :

.. code-block:: shell

   $ cd model-zoo
   $ python3 -m tpu_perf.precision_benchmark --target BM1684X --mlir -l full_cases.txt --priority_filter high

where the ``--target`` is used to specify the processor model, which currently supports ``BM1684`` , ``BM1684X`` , ``BM1688`` , ``BM1690`` and ``CV186X`` .

Note:

- If multiple SOPHGO accelerator cards are installed on the host, you can
specify the running device of ``tpu_perf`` by adding ``--devices id`` when using
``tpu_perf``. Such as:

.. code-block:: shell

   $ python3 -m tpu_perf.precision_benchmark --target BM1684X --devices 2 --mlir -l full_cases.txt --priority_filter high

- The precision test of ``BM1688``, ``BM1690``, and ``CV186X`` requires additional configuration of the following environment variables:

.. code-block:: shell

   $ export SET_NUM_SAMPLES_YOLO=200
   $ export SET_NUM_SAMPLES_TOPK=100
   $ export SET_NUM_SAMPLES_BERT=200


Specific parameter descriptions can be obtained with the following commands:

.. code-block:: shell

   $ python3 -m tpu_perf.precision_benchmark --help

The output precision data is available in ``output/topk.csv`` . The precision results for ``resnet18-v2``:

.. code-block:: shell

   name,top1,top5
   resnet18-v2-FP32,69.68%,89.23%
   resnet18-v2-INT8,69.26%,89.08%
