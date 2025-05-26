.. _fuse preprocess:

使用智能深度学习处理器做前处理
==============================
目前TPU-MLIR支持的两个主要系列BM168x（除BM1684外）与CV18xx均支持将图像常见的预处理加入到模型中进行计算。开发者可以在模型编译阶段，通过编译选项传递相应预处理参数,由编译器直接在模型运算前插⼊相应前处理算⼦，⽣成的bmodel或cvimodel即可以直接以预处理前的图像作为输⼊，随模型推理过程使⽤深度学习处理器处理前处理运算。

.. list-table:: 预处理类型支持情况
   :align: center
   :widths: 22 15 15
   :header-rows: 1

   * - 预处理类型
     - BM168x
     - CV18xx
   * - 图像裁剪
     - True
     - True
   * - 归一化计算
     - True
     - True
   * - NHWC to NCHW
     - True
     - True
   * - BGR/RGB 转换
     - True
     - True

其中图像裁剪会先将图片按使用 ``model_transform`` 工具时输入的 “--resize_dims” 参数将图片调整为对应的大小，再裁剪成模型输入的尺寸。而归一化计算支持直接将未进行预处理的图像数据(即unsigned int8格式的数据)做归一化处理。

若要将预处理融入到模型中，则需要在使用 ``model_deploy`` 工具进行部署时使用 “--fuse_preprocess” 参数。如果要做验证，则传入的 ``test_input`` 需要是图像原始格式的输入(即jpg, jpeg和png格式)， 相应地会生成原始图像输入对应的npz文件，名称为 ``${model_name}_in_ori.npz``。

此外，当实际外部输入格式与模型的格式不相同时，用 “--customization_format” 指定实际的外部输入格式，支持的格式说明如下（以下支持情况不包含BM1684）：

.. list-table:: customization_format格式和说明
   :widths: 27 43 12 10
   :header-rows: 1

   * - customization_format
     - 说明
     - BM168X
     - CV18xx
   * - None
     - 与原始模型输入保持一致, 不做处理。默认
     - True
     - True
   * - RGB_PLANAR
     - rgb顺序,按照nchw摆放
     - True
     - True
   * - RGB_PACKED
     - rgb顺序,按照nhwc摆放
     - True
     - True
   * - BGR_PLANAR
     - bgr顺序,按照nchw摆放
     - True
     - True
   * - BGR_PACKED
     - bgr顺序,按照nhwc摆放
     - True
     - True
   * - GRAYSCALE
     - 仅有⼀个灰⾊通道,按nchw摆
     - True
     - True
   * - YUV420_PLANAR
     - yuv420 planner格式,来⾃vpss的输⼊
     - True
     - True
   * - YUV_NV21
     - yuv420的NV21格式,来⾃vpss的输⼊
     - True
     - True
   * - YUV_NV12
     - yuv420的NV12格式,来⾃vpss的输⼊
     - True
     - True
   * - RGBA_PLANAR
     - rgba格式,按照nchw摆放
     - False
     - True

注意，BM168X模型中 ``YUV`` 格式的输入数据形状是（n, resize_dim_h, resize_dim_w）， ``resize_dim_h,resize_dim_w`` 为 ``model_transform `` 阶段的 ``resize_dim`` 参数。

当 ``customization_format`` 中颜色通道的顺序与模型输入不同时，将会进行通道转换操作。若指令中未设置 ``customization_format`` 参数，则根据使用 ``model_transform `` 工具时定义的 ``pixel_format`` 和 ``channel_format`` 参数自动获取对应的 ``customization_format`` 。


模型部署样例
------------
以mobilenet_v2模型为例, 参考 “编译Caffe模型” 章节，使用 ``model_transform`` 工具生成原始mlir, 并通过 ``run_calibration`` 工具生成校准表。


BM1684X部署
~~~~~~~~~~~

生成融合预处理的INT8对称量化bmodel模型指令如下:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor bm1684x \
       --test_input ../image/cat.jpg \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --fuse_preprocess \
       --model mobilenet_v2_bm1684x_int8_sym_fuse_preprocess.bmodel


CV18xx部署
~~~~~~~~~~

生成融合预处理的INT8对称量化cvimodel模型的指令如下:

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor cv183x \
       --test_input ../image/cat.jpg \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --fuse_preprocess \
       --customization_format RGB_PLANAR \
       --model mobilenet_v2_cv183x_int8_sym_fuse_preprocess.cvimodel

VPSS作为输入
^^^^^^^^^^^^^
当输入数据是来自于CV18xx提供的视频后处理模块VPSS时(使⽤VPSS进⾏预处理的详细使⽤⽅法请参阅《CV18xx 媒体软件开发参考》,本⽂档不做介绍)，则会有数据对齐要求，⽐如w
按照32字节对齐，此时 ``fuse_preprocess, aligned_input`` 需要同时被设置，生成融合预处理的cvimodel模型的指令如下：

.. code-block:: shell

   $ model_deploy \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor cv183x \
       --test_input ../image/cat.jpg \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --fuse_preprocess \
       --customization_format YUV_NV21 \
       --aligned_input \
       --model mobilenet_v2_cv183x_int8_sym_fuse_preprocess_aligned.cvimodel

上述指令中， ``aligned_input`` 指定了模型需要做输入的对齐。

值得注意的是：vpss做输入，runtime可以使用 ``CVI_NN_SetTensorPhysicalAddr`` 减少数据的拷贝。
