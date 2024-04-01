.. _onnx to cvimodel:

Appendix.02: CV18xx Guidance
=============================

CV18xx series processor currently supports ONNX and Caffe models but not TFLite models. In terms of quantization, CV18xx supports BF16 and symmetric INT8 format. This chapter takes the CV183X as an example to introduce the compilation and runtime sample of the CV18xx series processor.

Compile yolov5 model
--------------------

Install tpu_mlir
~~~~~~~~~~~~~~~~~~~~

Go to the Docker container and execute the following command to install tpu_mlir:

.. code-block:: shell

   $ pip install tpu_mlir[all]
   # or
   $ pip install tpu_mlir-*-py3-none-any.whl[all]


Prepare working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: get_resource.rst

Create a ``model_yolov5s`` directory, and put both model files and image files into the ``model_yolov5s`` directory.

The operation is as follows:

.. code-block:: shell
   :linenos:

   $ mkdir model_yolov5s && cd model_yolov5s
   $ wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
   $ cp -rf tpu_mlir_resource/dataset/COCO2017 .
   $ cp -rf tpu_mlir_resource/image .
   $ mkdir workspace && cd workspace


ONNX to MLIR
~~~~~~~~~~~~~~~~~~~~

If the input is an image, we need to learn the preprocessing of the model before conversion. If the model uses the preprocessed npz file as input, there is no need to consider preprocessing. The preprocessing process is expressed as follows ( :math:`x` stands for input):

.. math::

   y = (x - mean) \times scale


The input of yolov5 on the official website is rgb image, each value of it will be multiplied by ``1/255``, and converted into mean and scale corresponding to ``0.0,0.0,0.0`` and ``0.0039216,0.0039216,0.0039216``.

The model conversion command is as follows:

.. code-block:: shell

   $ model_transform \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 326,474,622 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

For the argument description of ``model_transform``, refer to the section :ref:`The main parameters of model_transform <model_transform param>` .

MLIR to BF16 Model
~~~~~~~~~~~~~~~~~~~~

Convert the mlir file to the cvimodel of bf16, the operation is as follows:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize BF16 \
       --processor cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --model yolov5s_cv183x_bf16.cvimodel

For the argument description of ``model_deploy``, refer to the section  :ref:`The main parameters of model_deploy <model_deploy param>` .

MLIR to INT8 Model
~~~~~~~~~~~~~~~~~~~~
Before converting to the INT8 model, you need to do calibration to get the calibration table. The number of input data depends on the situation but is normally around 100 to 1000. Then use the calibration table to generate INT8 symmetric cvimodel.

Here we use the 100 images from COCO2017 as an example to perform calibration:

.. code-block:: shell

   $ run_calibration yolov5s.mlir \
       --dataset ../COCO2017 \
       --input_num 100 \
       -o yolov5s_cali_table

After the operation is completed, a file named ``${model_name}_cali_table`` will be generated, which is used as the input of the following compilation work.

To convert to symmetric INT8 cvimodel model, execute the following command:

.. code-block:: shell

   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --processor cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --model yolov5s_cv183x_int8_sym.cvimodel

After compiling, a file named ``${model_name}_cv183x_int8_sym.cvimodel`` will be generated.


Result Comparison
~~~~~~~~~~~~~~~~~~~~

The onnx model is run as follows to get ``dog_onnx.jpg``:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model ../yolov5s.onnx \
       --output dog_onnx.jpg

The FP32 mlir model is run as follows to get ``dog_mlir.jpg``:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s.mlir \
       --output dog_mlir.jpg

The BF16 cvimodel is run as follows to get ``dog_bf16.jpg``:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_bf16.cvimodel \
       --output dog_bf16.jpg

The INT8 cvimodel is run as follows to get ``dog_int8.jpg``:

.. code-block:: shell

   $ detect_yolov5 \
       --input ../image/dog.jpg \
       --model yolov5s_cv183x_int8_sym.cvimodel \
       --output dog_int8.jpg


The comparison of the four images is shown in :numref:`yolov5s_result1`, due to the different operating environments, the final effect and accuracy will be slightly different from :numref:`yolov5s_result1`.

.. _yolov5s_result1:
.. figure:: ../assets/yolov5s_cvi.jpg
   :height: 13cm
   :align: center

   Comparing the results of different models



The above tutorial introduces the process of TPU-MLIR deploying the ONNX model to the CV18xx series processors. For the conversion process of the Caffe model, please refer to the chapter "Compiling the Caffe Model". You only need to replace the processors name with the specific CV18xx processors.

.. _merge weight:

Merge cvimodel Files
---------------------------
For the same model, independent cvimodel files can be generated according to the input batch size and resolution(different H and W). However, in order to save storage, you can merge these related cvimodel files into one cvimodel file and share its weight part. The steps are as follows:

Step 0: generate the cvimodel for batch 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the previous section to create a new workspace directory and convert yolov5s to the mlir fp32 model by model_transform

.. admonition:: Attention ：
  :class: attention

  1.Use the same workspace directory for the cvimodels that need to be merged, and do not share the workspace with other cvimodes that do not need to be merged.

  2.In Step 0, Step 1, --merge_weight is required


.. code-block:: shell

   $ model_transform \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 326,474,622 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s_bs1.mlir

Use the yolov5s_cali_table generated in preceding sections, or generate calibration table by run_calibration.

.. code-block:: shell

  # Add --merge_weight
   $ model_deploy \
       --mlir yolov5s_bs1.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --processor cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --merge_weight \
       --model yolov5s_cv183x_int8_sym_bs1.cvimodel

Step 1: generate the cvimodel for batch 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate mlir fp32 file in the same workspace:

.. code-block:: shell

   $ model_transform \
       --model_name yolov5s \
       --model_def ../yolov5s.onnx \
       --input_shapes [[2,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 326,474,622 \
       --test_input ../image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s_bs2.mlir

.. code-block:: shell

  # Add --merge_weight
   $ model_deploy \
       --mlir yolov5s_bs2.mlir \
       --quantize INT8 \
       --calibration_table yolov5s_cali_table \
       --processor cv183x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --tolerance 0.85,0.45 \
       --merge_weight \
       --model yolov5s_cv183x_int8_sym_bs2.cvimodel

Step 2: merge the cvimodel of batch 1 and batch 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use model_tool to mrege two cvimodel files:

.. code-block:: shell

  model_tool \
    --combine \
      yolov5s_cv183x_int8_sym_bs1.cvimodel \
      yolov5s_cv183x_int8_sym_bs2.cvimodel \
      -o yolov5s_cv183x_int8_sym_bs1_bs2.cvimodel

Step 3: use the cvimodel through the runtime interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use model_tool to check the program id of bs1 and bs2.:

.. code-block:: shell

  model_tool --info yolov5s_cv183x_int8_sym_bs1_bs2.cvimodel

At runtime, you can run different batch program in the following ways:

.. code-block:: c++

  CVI_MODEL_HANDEL bs1_handle;
  CVI_RC ret = CVI_NN_RegisterModel("yolov5s_cv183x_int8_sym_bs1_bs2.cvimodel", &bs1_handle);
  assert(ret == CVI_RC_SUCCESS);
  // choice batch 1 program
  CVI_NN_SetConfig(bs1_handle, OPTION_PROGRAM_INDEX, 0);
  CVI_NN_GetInputOutputTensors(bs1_handle, ...);
  ....


  CVI_MODEL_HANDLE bs2_handle;
  // Reuse loaded cvimodel
  CVI_RC ret = CVI_NN_CloneModel(bs1_handle, &bs2_handle);
  assert(ret == CVI_RC_SUCCESS);
  // choice batch 2 program
  CVI_NN_SetConfig(bs2_handle, OPTION_PROGRAM_INDEX, 1);
  CVI_NN_GetInputOutputTensors(bs2_handle, ...);
  ...

  // clean up bs1_handle and bs2_handle
  CVI_NN_CleanupModel(bs1_handle);
  CVI_NN_CleanupModel(bs2_handle);

Overview:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the above command, you can merge either the same models or different models

The main steps are:

1. When generating a cvimodel through model_deploy, add the --merge_weight parameter.
2. The work directory of the model to be merged must be the same, and do not clean up any intermediate files before merging the models(Reuse the previous model's weight is implemented through the intermediate file _weight_map.csv).
3. Use model_tool to merge cvimodels.




Compile and Run the Runtime Sample
-----------------------------------
This part introduces how to compile and run the runtime samples, include how to cross-compile samples for EVB board
and how to compile and run samples in docker. The following 4 samples are included:

* Sample-1 : classifier (mobilenet_v2)

* Sample-2 : classifier_bf16 (mobilenet_v2)

* Sample-3 : classifier fused preprocess (mobilenet_v2)

* Sample-4 : classifier multiple batch (mobilenet_v2)

1) Run the provided pre-build samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following files are required:

* cvitek_tpu_sdk_[cv183x|cv182x|cv182x_uclibc|cv181x_glibc32|cv181x_musl_riscv64_rvv|cv180x_musl_riscv64_rvv|cv181x_glibc_riscv64].tar.gz
* cvimodel_samples_[cv183x|cv182x|cv181x|cv180x].tar.gz

Select the required files according to the processor type and load them into the EVB file system.
Execute them on the Linux console of EVB. Here, we take CV183x as an example.

Unzip the model file (delivered in cvimodel format) and the TPU_SDK used by samples. Enter into the samples directory to execute the test.
The process is as follows:

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

At the same time, the script is provided as a reference, and the execution effect is the same as that of direct operation, as follows:

.. code-block:: shell

   ./run_classifier.sh
   ./run_classifier_bf16.sh
   ./run_classifier_fused_preprocess.sh
   ./run_classifier_multi_batch.sh

There are more samples can be refered in the ``cvitek_tpu_sdk/samples/samples_extra``：

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


2) Cross-compile samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The source code is given in the released packages. You can cross-compile the samples' source code in the docker environment and
run them on EVB board according to the following instructions.

The following files are required in this part:

* cvitek_tpu_sdk_[cv183x|cv182x|cv182x_uclibc|cv181x_glibc32|cv181x_musl_riscv64_rvv|cv180x_musl_riscv64_rvv].tar.gz
* cvitek_tpu_samples.tar.gz

aarch 64-bit  (such as cv183x aarch64-bit platform)
``````````````````````````````````````````````````````

Prepare TPU sdk:


.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv183x.tar.gz
   export PATH=$PWD/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin:$PATH
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

Compile samples and install them into "install_samples" directory:

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

arm 32-bit  (such as 32-bit cv183x/cv182x platform)
``````````````````````````````````````````````````````

Prepare TPU sdk:

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv182x.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

If docker version < 1.7, please update 32-bit system library(just once):

.. code-block:: shell

   dpkg --add-architecture i386
   apt-get update
   apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386

Compile samples and install them into ``install_samples`` directory:

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


uclibc 32-bit platform (such as cv182x uclibc platform)
`````````````````````````````````````````````````````````
Prepare TPU sdk:

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv182x_uclibc.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/arm-cvitek-linux-uclibcgnueabihf/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

If docker version < 1.7, please update 32-bit system library(just once):

.. code-block:: shell

   dpkg --add-architecture i386
   apt-get update
   apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386

Compile samples and install them into ``install_samples`` directory:

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

riscv 64-bit musl platform (such as cv180x/cv181x riscv 64-bit musl platform)
```````````````````````````````````````````````````````````````````````````````

Prepare TPU sdk:

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv181x_musl_riscv64_rvv.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

Compile samples and install them into ``install_samples`` directory:

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


riscv 64-bit glibc platform(such as cv180x/cv181x 64-bit glibc platform)
``````````````````````````````````````````````````````````````````````````````````````````

Prepare TPU sdk:

.. code-block:: shell

   tar zxf host-tools.tar.gz
   tar zxf cvitek_tpu_sdk_cv181x_glibc_riscv64.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   export PATH=$PWD/host-tools/gcc/riscv64-linux-x86_64/bin:$PATH
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

Compile samples and install them into ``install_samples`` directory:

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


3) Run samples in docker environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following files are required:

* cvitek_tpu_sdk_x86_64.tar.gz
* cvimodel_samples_[cv183x|cv182x|cv181x|cv180x].tar.gz
* cvitek_tpu_samples.tar.gz

Prepare TPU sdk:

.. code-block:: shell

   tar zxf cvitek_tpu_sdk_x86_64.tar.gz
   export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
   cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..

Compile samples and install them into ``install_samples`` directory:

.. code-block:: shell

   tar zxf cvitek_tpu_samples.tar.gz
   cd cvitek_tpu_samples
   mkdir build
   cd build
   cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_C_FLAGS_RELEASE=-O3 \
      -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
      -DTPU_SDK_PATH=$TPU_SDK_PATH \
      -DCNPY_PATH=$TPU_SDK_PATH/cnpy \
      -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
      -DCMAKE_INSTALL_PREFIX=../install_samples \
      ..
   cmake --build . --target install

Run samples:

.. code-block:: shell

   # envs
   tar zxf cvimodel_samples_cv183x.tar.gz
   export MODEL_PATH=$PWD/cvimodel_samples
   source cvitek_mlir/cvitek_envs.sh

   # get cvimodel info
   cd ../install_samples
   ./bin/cvi_sample_model_info $MODEL_PATH/mobilenet_v2.cvimodel

**Other samples are samely to the instructions of running on EVB board.**

FAQ
----

Model transformation FAQ
~~~~~~~~~~~~~~~~~~~~~~~~~~

1 Related to model transformation
`````````````````````````````````````

  1.1 Whether pytorch,tensorflow, etc. can be converted directly to cvimodel?

    pytorch: Supports the .pt model statically via ``jit.trace(torch_model.eval(), inputs).save('model_name.pt')``.

    tensorflow / others: It is not supported yet and can be supported indirectly through onnx.

  1.2 An error occurs when model_transform is executed

    ``model_transform`` This command convert the onnx,caffe model into the fp32 mlir. The high probability of error here is that there are unsupported operators or incompatible operator attributes, which can be fed back to the tpu team to solve.

  1.3 An error occurs when model_deploy is executed

    ``model_deploy`` This command quantizes fp32 mlir to int8/bf16mlir, and then converts int8/bf16mlir to cvimodel.
    In the process of conversion, two similarity comparisons will be involved: one is the quantitative comparison between fp32 mlir and int8/bf16mlir, and the other is the similarity comparison between int8/bf16mlir and the final converted cvimodel. If the similarity comparison fails, the following err will occur:

    .. figure:: ../assets/compare_failed.png
       :height: 13cm
       :align: center

    Solution: The tolerance parameter is incorrect. During the model conversion process, similarity will be calculated for the output of int8/bf16 mlir and fp32 mlir, and tolerance is to limit the minimum value of similarity. If the calculated minimum value of similarity is lower than the corresponding preset tolerance value, the program will stop execution. Consider making adjustments to tolerance. (If the minimum similarity value is too low, please report it to the tpu team.)

  1.4 What is the difference between the ``pixel_format parameter`` of ``model_transform`` and the ``customization_format`` parameter of ``model_deploy``?

    Channel_order is the input image type of the original model (only gray/rgb planar/bgr planar is supported),customization_format is the input image type of cvimodel, which is determined by the customer and must be used together with :ref:`fuse_preprocess <fuse preprocess>`. (If the input is a YUV image obtained through VPSS or VI, set customization_format to YUV format.) If pixel_format is inconsistent with customization_format,cvimodel will automatically converts the input to the type specified by pixel_format.

  1.5 Whether the multi-input model is supported and how to preprocess it?

    Models with multiple input images using different preprocessing methods are not supported.

2 Related to model quantization
````````````````````````````````````

  2.1 run run_calibration raise KeyError: 'images'

   Please check that the path of the data set is correct.

  2.2 How to deal with multiple input problems by running quantization?

    When running run_calibration, you can store multiple inputs using .npz, or using the --data_list argument, and the multiple inputs in each row of the data_list are separated by ",".

  2.3 Is the input preprocessed when quantization is performed?

    Yes, according to the preprocessing parameters stored in the mlir file, the quantization process is preprocessed by loading the preprocessing parameters.

  2.4 The program is killed by the system or the memory allocation fails when run calibration

    It is necessary to check whether the memory of the host is enough, and the common model requires about 8G memory. If memory is insufficient, try adding the following parameters when running run_calibration to reduce memory requirements.

     .. code-block:: shell

       --tune_num 2   			# default is 5

  2.5 Does the calibration table support manual modification?

    Supported, but it is not recommended.

3 Others
````````````````````

  3.1 Does the converted model support encryption?

    Not supported for now.

  3.2 What is the difference in inference speed between bf16 model and int8 model?

    The theoretical difference is about 3-4 times, and there will be differences for different models, which need to be verified in practice.

  3.3 Is dynamic shape supported?

    Cvimodel does not support dynamic shape. If several shapes are fixed, independent cvimodel files can be generated through the form of shared weights.
    See :ref:`Merge cvimodel Files <merge weight>` for details.

Model performance evaluation FAQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1 Evaluation process
```````````````````````

  First converted to bf16 model, through the ``model_tool --info xxxx.cvimodel`` command to obtain the ION memory and the storage space required by the model , and then execute ``model_runner`` on the EVB board to evaluate the performance, and then evaluate the accuracy in the business scenario according to the provided sample. After the accuracy of the model output meets the expectation, the same evaluation is performed on the int8 model.

2 After quantization, the accuracy does not match the original model, how to debug?
``````````````````````````````````````````````````````````````````````````````````````

  2.1 Ensure ``--test_input``, ``--test_reference``, ``--compare_all`` , ``--tolerance`` parameters are set up correctly.

  2.2 Compare the results of the original model and the bf16 model. If the error is large, check whether the pre-processing and post-processing are correct.

  2.3 If int8 model accuracy is poor:

    1) Verify that the data set used by run_calibration is the validation set used when training the model;

    2) A business scenario data set (typically 100-1000 images) can be added for run_calibration.

  2.4 Confirm the input type of cvimodel:

    1) If the ``--fuse_preprocess`` argument is specified, the input type of cvimodel is uint8;

    2) If ``--quant_input`` is specified,in general,bf16_cvimoel input type is fp32,int8_cvimodel input type is int8;

    3) The input type can also be obtained with ``model_tool --info xxx.cvimodel``

3 bf16 model speed is relatively slow,int8 model accuracy does not meet expectations how to do?
``````````````````````````````````````````````````````````````````````````````````````````````````

  Try using a mixed-precision quantization method. See :ref:`mix precision` for details.

Common problems of model deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1 The The CVI_NN_Forward interface encounters an error after being invoked for many times or is stuck for a long time
```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

  There may be driver or hardware issues that need to be reported to the tpu team for resolution.

2 Is the model preprocessing slow?
``````````````````````````````````````

  2.1 Add the ``--fuse_preprocess`` parameter when running model_deploy, which
  will put the preprocessing inside the Tensor Computing Processor for processing.

  2.2 If the image is obtained from vpss or vi, you can use ``--fuse_preprocess``, ``--aligned_input`` when converting to the model. Then use an interface such as CVI_NN_SetTensorPhysicalAddr to set the input tensor address directly to the physical address of the image, reducing the data copy time.

3 Are floating-point and fixed-point results the same when comparing the inference results of docker and evb ?
```````````````````````````````````````````````````````````````````````````````````````````````````````````````

  Fixed point has no difference, floating point has difference, but the difference can be ignored.

4 Support multi-model inference parallel?
````````````````````````````````````````````

  Multithreading is supported, but models are inferred on Tensor Computing Processor in serial.

5 Fill input tensor related interface
`````````````````````````````````````````

  ``CVI_NN_SetTensorPtr`` : Set the virtual address of input tensor, and the original tensor memory will not be freed. Inference **copies data** from a user-set virtual address to the original tensor memory.

  ``CVI_NN_SetTensorPhysicalAddr`` : Set the physical address of input tensor, and the original tensor memory will be freed. Inference directly reads data from the newly set physical address, **data copy is not required** . A Frame obtained from VPSS can call this interface by passing in the Frame's first address. Note that model_deploy must be set ``--fused_preprocess`` and ``--aligned_input`` .

  ``CVI_NN_SetTensorWithVideoFrame`` : Fill the Input Tensor with the VideoFrame structure. Note The address of VideoFrame is a physical address. If the model is fused preprocess and aligned_input, it is equivalent to CVI_NN_SetTensorPhysicalAddr, otherwise the VideoFrame data will be copied to the Input Tensor.

  ``CVI_NN_SetTensorWithAlignedFrames`` : Support multi-batch, similar to ``CVI_NN_SetTensorWithVideoFrame`` .

  ``CVI_NN_FeedTensorWithFrames`` : similar to ``CVI_NN_SetTensorWithVideoFrame`` .

6 How is ion memory allocated after model loading
`````````````````````````````````````````````````````

  6.1 Calling ``CVI_NN_RegisterModel`` allocates ion memory for weight and cmdbuf (you can see the weight and cmdbuf sizes by using model_tool).

  6.2 Calling ``CVI_NN_GetInputOutputTensors`` allocates ion memory for tensor(including private_gmem, shared_gmem, io_mem).

  6.3 Calling ``CVI_NN_CloneModel`` can share weight and cmdbuf memory.

  6.4 Other interfaces do not apply for ion memory.

  6.5 Shared_gmem of different models can be shared (including multithreading), so initializing shared_gmem of the largest model first will saves ion memory.

7 The model inference time becomes longer after loading the business program
`````````````````````````````````````````````````````````````````````````````````

  Generally, after services are loaded, the tdma_exe_ms becomes longer, but the tiu_exe_ms remains unchanged. This is because tdma_exe_ms takes time to carry data in memory. If the memory bandwidth is insufficient, the tdma time will increase.

  suggestion:

    1) vpss/venc optimize chn and reduce resolution

    2) Reduces memory copy

    3) Fill input tensor by using copy-free mode

Others
~~~~~~~~~~~~~~~~~~~~

1 In the cv182x/cv181x/cv180x on-board environment, the taz:invalid option --z decompression fails
```````````````````````````````````````````````````````````````````````````````````````````````````````````

  Decompress the sdk in other linux environments and then use it on the board. windows does not support soft links. Therefore, decompressing the SDK in Windows may cause the soft links to fail and an error may be reported

2 If tensorflow model is pb form of saved_model, how to convert it to pb form of frozen_model
```````````````````````````````````````````````````````````````````````````````````````````````````

  .. code-block:: shell

   import tensorflow as tf
   from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
   import numpy as np
   import tf2onnx
   import onnxruntime as rt

   img_path = "./cat.jpg"
   # pb model and variables should in model dir
   pb_file_path = "your model dir"
   img = image.load_img(img_path, target_size=(224, 224))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   # Or set your preprocess here
   x = preprocess_input(x)

   model = tf.keras.models.load_model(pb_file_path)
   preds = model.predict(x)

   # different model input shape and name will differently
   spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"), )
   output_path = model.name + ".onnx"

   model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
