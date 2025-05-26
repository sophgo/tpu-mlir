.. _fuse preprocess:

Use Tensor Computing Processor for Preprocessing
=================================================
At present, the two main series of processors supported by TPU-MLIR are BM168x (except BM1684) and CV18xx. Both of them support common image preprocessing fusion. The developer can pass the preprocessing arguments during the compilation process, and the compiler will directly insert the corresponding preprocessing operators into the generated model. The generated bmodel or cvimodel can directly use the unpreprocessed image as input and use TPU to do the preprocessing.

.. list-table:: Supported Preprocessing Type
   :align: center
   :widths: 22 15 15
   :header-rows: 1

   * - Preprocessing Type
     - BM168x
     - CV18xx
   * - Crop
     - True
     - True
   * - Normalization
     - True
     - True
   * - NHWC to NCHW
     - True
     - True
   * - BGR/ RGB Conversion
     - True
     - True

The image cropping will first adjust the image to the size specified by the "--resize_dims" argument of the ``model_transform`` tool, and then crop it to the size of the model input. The normalization supports directly converting unpreprocessed image data.

To integrate preprocessing into the model, you need to speficy the "--fuse_preprocess" argument when using the ``model_deploy`` tool, and the ``test_input`` should be an image of the original format (i.e., jpg, jpeg and png format). There will be a preprocessed npz file of input named ``${model_name}_in_ori.npz`` generated. In addition, there is a "--customization_format" argument to specify the original image format input to the model. The supported image formats are described as follows (BM1684 not included):

.. list-table:: Types of customization_format and Description
   :widths: 27 43 12 10
   :header-rows: 1

   * - customization_format
     - Description
     - BM168x
     - CV18xx
   * - None
     - same with model format, do nothing, as default
     - True
     - True
   * - RGB_PLANAR
     - rgb color order and nchw tensor format
     - True
     - True
   * - RGB_PACKED
     - rgb color order and nhwc tensor format
     - True
     - True
   * - BGR_PLANAR
     - bgr color order and nchw tensor format
     - True
     - True
   * - BGR_PACKED
     - bgr color order and nhwc tensor format
     - True
     - True
   * - GRAYSCALE
     - one color channel only and nchw tensor format
     - True
     - True
   * - YUV420_PLANAR
     - yuv420 planner format, from vpss input
     - True
     - True
   * - YUV_NV21
     - NV21 format of yuv420, from vpss input
     - True
     - True
   * - YUV_NV12
     - NV12 format of yuv420, from vpss input
     - True
     - True
   * - RGBA_PLANAR
     - rgba format and nchw tensor format
     - False
     - True

Note that the input data shape of the ``YUV`` format in the BM168X model is (n, resize_dim_h, resize_dim_w), ``resize_dim_h,resize_dim_w`` is the ``resize_dim`` argument of the ``model_transform`` stage.

When the order of the color channels in the ``customization_format`` is different from the model input, a channel conversion operation will be performed. If the customization_format argument is not specified, the corresponding ``customization_format`` will be automatically set according to the ``pixel_format`` and ``channel_format`` arguments defined when using the ``model_transform`` tool.


Model Deployment Example
-------------------------
Take the mobilenet_v2 model as an example, use the ``model_transform`` tool to generate the original mlir, and the ``run_calibration`` tool to generate the calibration table (refer to the chapter "Compiling the Caffe Model" for more details).


Deploy to BM168x
~~~~~~~~~~~~~~~~~~~

The command to generate the preprocess-fused symmetric INT8 quantized bmodel model is as follows:

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


Deploy to CV18xx
~~~~~~~~~~~~~~~~~

The command to generate the preprocess-fused symmetric INT8 quantized cvimodel model are as follows:

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

vpss input
^^^^^^^^^^^^^
When the input data comes from the video post-processing module VPSS provided by CV18xx (for details on how to use VPSS for preprocessing, please refer to "CV18xx Media Software Development Reference"), data alignment is required (e.g., 32-bit aligned width), fuse_preprocess and aligned_input need to be set at the same time. The command to generate the preprocessed-fused cvimodel model is as follows:

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
       --aligned_input \
       --model mobilenet_v2_cv183x_int8_sym_fuse_preprocess_aligned.cvimodel

In the above command, ``aligned_input`` specifies the alignment that the model input needs to do.

Note that with vpss as input, runtime can use ``CVI_NN_SetTensorPhysicalAddr`` to reduce memory data copy.
