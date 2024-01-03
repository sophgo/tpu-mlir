.. _profile:

附录05：TPU Profile工具使用
==================================
本章节主要是介绍如何利用Profile数据及TPU Profile工具，可视化模型的完整运行流程，以便于进行模型性能分析。


编译bmodel
------------------

TPU Profile是将Profile数据转换为可视化网页的工具。首先先生成bmodel，下面以tpu-mlir工程中的yolov5s模型来演示。

由于Profile数据会把编译中的一些layer信息保存到bmodel中，导致bmodel体积变大，所以默认是关闭的。打开方式是在调用model_deploy.py加上--debug选项。
如果在编译时未开启该选项，运行时开启Profile得到的数据在可视化时，会有部分数据缺失。

.. code-block:: shell

   # 生成 top mlir
   $ model_transform \
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

.. code-block:: shell

   # 将top mlir转换成fp16精度的bmodel
   $ model_deploy \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --processor bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --model yolov5s_1684x_f16.bmodel \
       --debug # 记录profile数据

通过以上命令，将yolov5s.onnx编译成了yolov5s_bm1684x_f16.bmodel。

生成Profile原始数据
--------------------------

将生成的yolov5s_bm1684x_f16.bmodel拷贝到运行环境。同编译过程，运行时的Profile功能默认是关闭的，防止在做profile保存与传输时产生额外时间消耗。需要开启profile功能时，在运行编译好的应用前设置环境变量BMRUNTIME_ENABLE_PROFILE=1即可。然后用libsophon中提供的模型测试工具bmrt_test来作为应用，生成profile数据。

.. code-block:: shell

    export BMRUNTIME_ENABLE_PROFILE=1
    bmrt_test --bmodel yolov5s_1684x_f16.bmodel

下面是开启Profile后运行输出的日志:

    .. _profile_log:
    .. figure:: ../assets/profile_log_en.png
          :height: 13cm
          :align: center

          开启 Profile 后运行输出的日志

同时在当前目录生成bmprofile_data-1文件夹, 为全部的Profile数据。

可视化Profile数据
--------------------------

将bmprofile_data-1目录拷贝回tpu-mlir工程环境。tpu-mlir提供了tpu_profile.py脚本，来把生成的二进制profile数据转换成网页文件，来进行可视化。命令如下：

.. code-block:: shell

    # 将bmprofile_data_0目录的profile原始数据转换成网页放置到bmprofile_out目录
    # 如果有图形界面，会直接打开浏览器，直接看到结果
    tpu_profile.py bmprofile_data-1 bmprofile_out
    ls bmprofile_out
    # echarts.min.js  profile_data.js  result.html


用浏览器打开bmprofile_out/result.html可以看到profile的图表。此外，该工具还有其他用法，可以通过tpu_profile.py --help来查看。更多的Profile工具使用分析说明请参考https://tpumlir.org/zh-cn/2023/09/18/analyse-tpu-performance-with-tpu-profile.html
