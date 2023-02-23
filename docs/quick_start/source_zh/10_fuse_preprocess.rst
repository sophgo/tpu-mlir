使用TPU做前处理
==================
目前CV18XX系列芯片支持将图像常见的预处理加入到模型中进行计算,包括对图像的裁剪、将外部输入的数据格式转化为模型推理所需要的数据格式、归一化计算等。开编发者可以在模型编译阶段,通过译选项传递相应预
处理参数,由编译器直接在模型运输前插⼊相应前处理算⼦,⽣成的cvimodel即可以直接以预处理前的图像作为输⼊,随模型推理过程使⽤TPU处理前处理运算。

这里以mobilenet_v2模型为例,介绍在模型中融合预处理的使用方法。参考“编译Caffe模型”的章节,在model_transform工具生成原始的mlir之后,生成融合预处理的cvimodel模型的指令如下：

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --chip cv183x \
       --test_input ../image/cat.jpg \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --fuse_preprocess \
       --customization_format RGB_PLANAR \
       --model mobilenet_v2_cv183x_int8_sym_fuse_preprocess.cvimodel


上述指令中,fuse_preprocess指定了模型需要做预处理,customization_format是输入到模型的原始图像格式,支持的图像格式和说明如下：

.. list-table:: customization_format格式和说明
   :widths: 22 50
   :header-rows: 1

   * - customization_format
     - 说明
   * - RGB_PLANAR
     - rgb顺序,按照nchw摆放
   * - RGB_PACKED
     - rgb顺序,按照nhwc摆放
   * - BGR_PLANAR
     - bgr顺序,按照nchw摆放
   * - BGR_PACKED
     - bgr顺序,按照nhwc摆放
   * - GRAYSCALE
     - 仅有⼀个灰⾊通道,按nchw摆
   * - YUV420_PLANAR
     - yuv420 planner格式,来⾃vpss的输⼊
   * - YUV_NV21
     - yuv420的NV21格式,来⾃vpss的输⼊
   * - YUV_NV12
     - yuv420的NV12格式,来⾃vpss的输⼊
   * - RGBA_PLANAR
     - rgba格式,按照nchw摆放

若指令中未设置customization_format参数,则根据使用model_transform工具时定义的pixel_format和channel_format参数自动获取对应的customization_format。
需要注意的是,当需要把预处理融合到模型时,传入的test_input需要是图像原始格式的输入,相应地会生成原始图像输入对应的npz文件,名称为 ``${model_name}_in_ori.npz``.

当输入数据是来自于CV18xx提供的视频后处理模块VPSS时（使⽤VPSS进⾏预处理的详细使⽤⽅法请参阅《CV18xx 媒体软件开发参考》,本⽂档不做介绍）,则会有数据对齐要求,⽐如w
按照32字节对齐,此时生成融合预处理的cvimodel模型的指令如下：

.. code-block:: shell

   $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --chip cv183x \
       --test_input ../image/cat.jpg \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.96,0.70 \
       --fuse_preprocess \
       --customization_format RGB_PLANAR \
       --aligned_input \
       --model mobilenet_v2_cv183x_int8_sym_fuse_preprocess_aligned.cvimodel

上述指令中,aligned_input指定了模型需要做输入的对齐。需要注意的是,YUV格式的输入数据fuse_preprocess和aligned_input需要都做,其它格式的fuse_preprocess和aligned_input的操作可选择只做其中一个或两个都做,
若只做aligned_input操作,则需要设置test_input为做过预处理的 ``${model_name}_in_f32.npz`` 格式,和“编译Caffe模型”的章节设置是一致的。


