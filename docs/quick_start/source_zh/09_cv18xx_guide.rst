CV18XX芯片使用指南
===================

CV18XX支持ONNX系列和Caffe模型,目前不支持TFLite模型。在量化数据类型方面,CV18XX支持BF16格式的量化
和INT8格式的非对称量化。本章节以CV183X芯片为例,介绍CV18XX系列芯片编译模型和运行runtime sample。

编译yolov5模型
------------------

加载tpu-mlir
~~~~~~~~~~~~~~~~~~~~

.. include:: env_var.rst

准备工作目录
~~~~~~~~~~~~~~~~~~~~

建立 ``model_yolov5s`` 目录, 注意是与tpu-mlir同级目录; 并把模型文件和图片文件都
放入 ``model_yolov5s`` 目录中。


操作如下:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s && cd model_yolov5s
   $ cp $TPUC_ROOT/regression/model/yolov5s.onnx .
   $ cp -rf $TPUC_ROOT/regression/dataset/COCO2017 .
   $ cp -rf $TPUC_ROOT/regression/image .
   $ mkdir workspace && cd workspace


这里的 ``$TPUC_ROOT`` 是环境变量, 对应tpu-mlir_xxxx目录。

ONNX转MLIR
~~~~~~~~~~~~~~~~~~~~

如果模型是图片输入, 在转模型之前我们需要了解模型的预处理。如果模型用预处理后的npz文件做输入, 则不需要考虑预处理。
预处理过程用公式表达如下( :math:`x` 代表输入):

.. math::

   y = (x - mean) \times scale


官网yolov5的图片是rgb, 每个值会乘以 ``1/255`` , 转换成mean和scale对应为
``0.0,0.0,0.0`` 和 ``0.0039216,0.0039216,0.0039216`` 。

模型转换命令如下:

.. code-block:: shell

   $ model_transform.py \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 350,498,646 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

``model_transform`` 的相关参数说明参考“编译ONNX模型-ONNX转MLIR”部分。

MLIR转BF16模型
~~~~~~~~~~~~~~~~~~~~

将mlir文件转换成bf16的cvimodel, 操作方法如下:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize BF16 \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.99,0.99 \
       --model yolov5s_cv183x_bf16.cvimodel

``model_deploy.py`` 的相关参数说明参考“编译ONNX模型-MLIR转F32模型”部分。

MLIR转INT8模型
~~~~~~~~~~~~~~~~~~~~
转INT8模型前需要跑calibration, 得到校准表; 输入数据的数量根据情况准备100~1000张左右。然后用校准表, 生成INT8对称cvimodel

这里用现有的100张来自COCO2017的图片举例, 执行calibration:

.. code-block:: shell

   $ run_calibration.py yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

运行完成后会生成名为 ``${model_name}_cali_table`` 的文件, 该文件用于后续编译INT8
模型的输入文件。

转成INT8对称量化cvimodel模型, 执行如下命令:

.. code-block:: shell

   $ model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --chip cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_cv183x_int8_sym.cvimodel

编译完成后, 会生成名为 ``${model_name}_cv183x_int8_sym.cvimodel`` 的文件。


效果对比
~~~~~~~~~~~~~~~~~~~~

onnx模型的执行方式如下, 得到 ``dog_onnx.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model ../yolov5s.onnx \
       --output dog_onnx.jpg

FP32 mlir模型的执行方式如下,得到 ``dog_mlir.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s.mlir \
       --output dog_mlir.jpg

BF16 cvimodel的执行方式如下, 得到 ``dog_bf16.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_bf16.cvimodel \
       --output dog_bf16.jpg

INT8 cvimodel的执行方式如下, 得到 ``dog_int8.jpg`` :

.. code-block:: shell

   $ detect_yolov5.py \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_int8_sym.cvimodel \
       --output dog_int8.jpg


四张图片对比如 :numref:`yolov5s_result1` ，由于运行环境不同, 最终的效果和精度与 :numref:`yolov5s_result1` 会有些差异。

.. _yolov5s_result1:
.. figure:: ../assets/yolov5s_cvi.jpg
   :height: 13cm
   :align: center

   不同模型效果对比



上述教程介绍了TPU-MLIR编译CV18XX系列芯片的ONNX模型的过程,caffe模型的转换过程可参考“编译Caffe模型”章节,只需要将对应的芯片名称换成实际的CV18XX芯片名称即可。


合并cvimodel模型文件
--------------------------
待补充


编译和运行runtime sample
--------------------------

本章首先介绍EVB如何运行sample应用程序，然后介绍如何交叉编译sample应用程序，最后介绍docker仿真编译和运行sample。具体包括3个samples：
* Sample-1 : classifier (mobilenet_v2)

* Sample-2 : classifier_bf16 (mobilenet_v2)

* Sample-3 : classifier fused preprocess (mobilenet_v2)

* Sample-4 : classifier multiple batch (mobilenet_v2)

1) 在EVB运行release提供的sample预编译程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要如下文件：

* cvitek_tpu_sdk_[cv182x|cv182x_uclibc|cv183x|cv181x_glibc32|cv181x_musl_riscv64_rvv|cv180x_musl_riscv64_rvv|cv181x_glibc_riscv64].tar.gz
* cvimodel_samples_[cv182x|cv183x|cv181x|cv180x].tar.gz

将根据chip类型选择所需文件加载至EVB的文件系统，于evb上的linux console执行，以cv183x为例：

解压samples使用的model文件（以cvimodel格式交付），并解压TPU_SDK，并进入samples目录，执行测试，过程如下：

.. code-block:: shell

   #env
   tar zxf cvimodel_samples_cv183x.tar.gz
   export MODEL_PATH=$PWD/cvimodel_samples
   tar zxf cvitek_tpu_sdk_cv183x.tar.gz
   export TPU_ROOT=$PWD/cvitek_tpu_sdk
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh
   # get cvimodel info
   cd samples
   ./bin/cvi_sample_model_info $MODEL_PATH/mobilenet_v2.cvimodel

   ####################################
   # sample-1 : classifier
   ###################################
   ./bin/cvi_sample_classifier \
       $MODEL_PATH/mobilenet_v2.cvimodel \
       ./data/cat.jpg \
       ./data/synset_words.txt

   # TOP_K[5]:
   #  0.326172, idx 282, n02123159 tiger cat
   #  0.326172, idx 285, n02124075 Egyptian cat
   #  0.099609, idx 281, n02123045 tabby, tabby cat
   #  0.071777, idx 287, n02127052 lynx, catamount
   #  0.041504, idx 331, n02326432 hare

   ####################################
   # sample-2 : classifier_bf16
   ###################################
   ./bin/cvi_sample_classifier_bf16 \
       $MODEL_PATH/mobilenet_v2_bf16.cvimodel \
       ./data/cat.jpg \
       ./data/synset_words.txt

   # TOP_K[5]:
   #  0.314453, idx 285, n02124075 Egyptian cat
   #  0.040039, idx 331, n02326432 hare
   #  0.018677, idx 330, n02325366 wood rabbit, cottontail, cottontail rabbit
   #  0.010986, idx 463, n02909870 bucket, pail
   #  0.010986, idx 852, n04409515 tennis ball


   ############################################
   # sample-3 : classifier fused preprocess
   ############################################
   ./bin/cvi_sample_classifier_fused_preprocess \
       $MODEL_PATH/mobilenet_v2_fused_preprocess.cvimodel \
       ./data/cat.jpg \
       ./data/synset_words.txt

   # TOP_K[5]:
   #  0.326172, idx 282, n02123159 tiger cat
   #  0.326172, idx 285, n02124075 Egyptian cat
   #  0.099609, idx 281, n02123045 tabby, tabby cat
   #  0.071777, idx 287, n02127052 lynx, catamount
   #  0.041504, idx 331, n02326432 hare

   ############################################
   # sample-4 : classifier multiple batch
   ############################################
   ./bin/cvi_sample_classifier_multi_batch \
       $MODEL_PATH/mobilenet_v2_bs1_bs4.cvimodel \
       ./data/cat.jpg \
       ./data/synset_words.txt

   # TOP_K[5]:
   #  0.326172, idx 282, n02123159 tiger cat
   #  0.326172, idx 285, n02124075 Egyptian cat
   #  0.099609, idx 281, n02123045 tabby, tabby cat
   #  0.071777, idx 287, n02127052 lynx, catamount
   #  0.041504, idx 331, n02326432 hare

同时提供脚本作为参考，执行效果与直接运行相同，如下：

.. code-block:: shell

   ./run_classifier.sh
   ./run_classifier_bf16.sh
   ./run_classifier_fused_preprocess.sh
   ./run_classifier_multi_batch.sh

**在cvitek_tpu_sdk/samples/samples_extra目录下有更多的samples，可供参考：**

.. code-block:: shell

   ./bin/cvi_sample_detector_yolo_v3_fused_preprocess \
       $MODEL_PATH/yolo_v3_416_fused_preprocess_with_detection.cvimodel \
       ./data/dog.jpg \
       yolo_v3_out.jpg

   ./bin/cvi_sample_detector_yolo_v5_fused_preprocess \
       $MODEL_PATH/yolov5s_fused_preprocess.cvimodel \
       ./data/dog.jpg \
       yolo_v5_out.jpg

   ./bin/cvi_sample_detector_yolox_s \
       $MODEL_PATH/yolox_s.cvimodel \
       ./data/dog.jpg \
       yolox_s_out.jpg

   ./bin/cvi_sample_alphapose_fused_preprocess \
       $MODEL_PATH/yolo_v3_416_fused_preprocess_with_detection.cvimodel \
       $MODEL_PATH/alphapose_fused_preprocess.cvimodel \
       ./data/pose_demo_2.jpg \
       alphapose_out.jpg

   ./bin/cvi_sample_fd_fr_fused_preprocess \
       $MODEL_PATH/retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel \
       $MODEL_PATH/arcface_res50_fused_preprocess.cvimodel \
       ./data/obama1.jpg \
       ./data/obama2.jpg

2) 交叉编译samples程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

发布包有samples的源代码，按照本节方法在Docker环境下交叉编译samples程序，然后在evb上运行。

本节需要如下文件：

* cvitek_tpu_sdk_[cv182x|cv182x_uclibc|cv183x|cv181x_glibc32|cv181x_musl_riscv64_rvv|cv180x_musl_riscv64_rvv]].tar.gz
* cvitek_tpu_samples.tar.gz

aarch 64位  (如cv183x aarch64位平台)
""""""""""""""""""""""""""""""""""""""

TPU sdk准备：

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv183x.tar.gz
   export PATH=$PWD/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin:$PATH
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_C_FLAGS_RELEASE=-O3 \
       -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
       -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-aarch64-linux.cmake \
       -DTPU_SDK_PATH=$TPU_SDK_PATH \
       -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
       -DCMAKE_INSTALL_PREFIX=../install_samples \
       ..
   cmake --build . --target install


arm 32位  (如cv183x平台32位、cv182x平台)
""""""""""""""""""""""""""""""""""""""""""

TPU sdk准备：

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv182x.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

如果docker版本低于1.7，则需要更新32位系统库（只需一次）：

.. code-block:: shell

   dpkg --add-architecture i386
   apt-get update
   apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386


编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_C_FLAGS_RELEASE=-O3 \
       -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
       -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-linux-gnueabihf.cmake \
       -DTPU_SDK_PATH=$TPU_SDK_PATH \
       -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
       -DCMAKE_INSTALL_PREFIX=../install_samples \
       ..
   cmake --build . --target install


uclibc 32位平台 (cv182x uclibc平台)
""""""""""""""""""""""""""""""""""""""

TPU sdk准备：

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv182x_uclibc.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/arm-cvitek-linux-uclibcgnueabihf/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

如果docker版本低于1.7，则需要更新32位系统库（只需一次）：

.. code-block:: shell

   dpkg --add-architecture i386
   apt-get update
   apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386


编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_C_FLAGS_RELEASE=-O3 \
       -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
       -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-linux-uclibc.cmake \
       -DTPU_SDK_PATH=$TPU_SDK_PATH \
       -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
       -DCMAKE_INSTALL_PREFIX=../install_samples \
       ..
   cmake --build . --target install


riscv64位 musl平台 (如cv181x、cv180x riscv64位 musl平台)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

TPU sdk准备：

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv181x_musl_riscv64_rvv.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_C_FLAGS_RELEASE=-O3 \
       -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
       -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
       -DTPU_SDK_PATH=$TPU_SDK_PATH \
       -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
       -DCMAKE_INSTALL_PREFIX=../install_samples \
       ..
   cmake --build . --target install

riscv64位 glibc平台 (如cv181x、cv180x riscv64位glibc平台)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

TPU sdk准备：

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv181x_glibc_riscv64.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/riscv64-linux-x86_64/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_C_FLAGS_RELEASE=-O3 \
       -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
       -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-riscv64-linux-x86_64.cmake \
       -DTPU_SDK_PATH=$TPU_SDK_PATH \
       -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
       -DCMAKE_INSTALL_PREFIX=../install_samples \
       ..
   cmake --build . --target install


1) docker环境仿真运行的samples程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* cvimodel_samples_[cv182x|cv183x|cv181x|cv180x].tar.gz
* cvitek_tpu_samples.tar.gz


TPU sdk准备：

.. code-block:: shell

   tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
   source cvitek_mlir/cvitek_envs.sh

编译samples，安装至install_samples目录：

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build_soc
   cd build_soc
   cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_C_FLAGS_RELEASE=-O3 \
      -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
      -DTPU_SDK_PATH=$MLIR_PATH/tpuc \
      -DCNPY_PATH=$MLIR_PATH/cnpy \
      -DOPENCV_PATH=$MLIR_PATH/opencv \
      -DCMAKE_INSTALL_PREFIX=../install_samples \
      ..
   cmake --build . --target install

运行samples程序：

.. code-block:: shell

   # envs
   tar zxf cvimodel_samples_cv183x.tar.gz
   export MODEL_PATH=$PWD/cvimodel_samples
   source cvitek_mlir/cvitek_envs.sh

   # get cvimodel info
   cd ../install_samples
   ./bin/cvi_sample_model_info $MODEL_PATH/mobilenet_v2.cvimodel

**其他samples运行命令参照EVB运行命令**




