.. figure:: ../assets/sophon.png
   :width: 400px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

| **Legal Notices**
| Copyright © SOPHGO 2025. All rights reserved.
| No part or all of the contents of this document may be copied, reproduced or transmitted in any form by any organization or individual without the written permission of the Company.

| **Attention**
| All products, services or features, etc. you purchased is subject to SOPHGO's business contracts and terms.
  All or part of the products, services or features described in this document may not be covered by your purchase or use.
  Unless otherwise agreed in the contract, SOPHGO makes no representations or warranties (including express and implied) regarding the contents of this document.
  The contents of this document may be updated from time to time due to product version upgrades or other reasons.
  Unless otherwise agreed, this document is intended as a guide only. All statements, information and recommendations in this document do not constitute any warranty, express or implied.

| **Technical Support**

:Address: Building 1, Zhongguancun Integrated Circuit Design Park (ICPARK), No. 9 Fenghao East Road, Haidian District, Beijing
:Postcode: 100094
:URL: https://www.sophgo.com/
:Email: sales@sophgo.com
:Tel: 010-57590723

| **Release Record**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Version
     - Release date
     - Explanation
   * - v1.20.0
     - 2025.06.30
     - Support IO_RELOC;
       Deconv3D INT8 bugfix;
       BatchNorm and Conv backward operators support 128 batch training
   * - v1.19.0
     - 2025.05.30
     - Support AWQ and GPTQ models;
       Deconv3D F16, F32 bugfix
   * - v1.18.0
     - 2025.05.01
     - YOLO series adds automatic mixed precision setting;
       Added SmoothQuant option for run_calibration;
       New one-click compilation script for LLM
   * - v1.17.0
     - 2025.04.03
     - Significant improvement in LLM model compilation speed;
       TPULang supports PPL operator integration;
       Fixed random error issue with Trilu bf16 on Mars3
   * - v1.16.0
     - 2025.03.03
     - TPULang ROI_Extractor support;
       Einsum supports abcde,abfge->abcdfg pattern;
       LLMC adds Vila model support
   * - v1.15.0
     - 2025.02.05
     - Added LLMC quantization support;
       Address boundary check in codegen;
       Fixed several comparison issues
   * - v1.14.0
     - 2025.01.02
     - Added post-processing fusion for yolov8/v11;
       Support for Conv3D stride > 15;
       Improved FAttention accuracy
   * - v1.13.0
     - 2024.12.02
     - Streamlined Release package;
       Performance optimization for MaxPoolWithMask training operator;
       Added support for large RoPE operators
   * - v1.12.0
     - 2024.11.06
     - tpuv7-runtime cmodel integration;
       BM1690 multi-core LayerGroup optimization;
       Support for PPL backend operator development
   * - v1.11.0
     - 2024.09.27
     - Added soc mode for BM1688 tdb;
       bmodel supports fine-grained merging;
       Fixed several performance degradation issues
   * - v1.10.0
     - 2024.08.15
     - Added yolov10 support;
       New quantization tuning section;
       Optimized tpu-perf log output
   * - v1.9.0
     - 2024.07.16
     - BM1690 added 40 model regression tests;
       New quantization algorithms: octav, aciq_guas and aciq_laplace
   * - v1.8.0
     - 2024.05.30
     - BM1690 supports multi-core MatMul operator;
       TPULang supports input/output order specification;
       tpuperf removes patchelf dependency
   * - v1.7.0
     - 2024.05.15
     - CV186X dual-core changed to single-core;
       BM1690 testing process aligned with BM1684X;
       Support for gemma/llama/qwen models
   * - v1.6.0
     - 2024.02.23
     - Added Pypi release format;
       Support for user-defined Global operators;
       Added CV186X processor platform support
   * - v1.5.0
     - 2023.11.03
     - More Global Layer support for multi-core parallel processing
   * - v1.4.0
     - 2023.09.27
     - System dependencies upgraded to Ubuntu22.04;
       Added BM1684 Winograd support
   * - v1.3.0
     - 2023.07.27
     - Added manual floating-point operation region specification;
       Added supported frontend framework operator list;
       Added NNTC vs TPU-MLIR quantization comparison
   * - v1.2.0
     - 2023.06.14
     - Adjusted mixed quantization examples
   * - v1.1.0
     - 2023.05.26
     - Added post-processing using intelligent deep learning processor
   * - v1.0.0
     - 2023.04.10
     - PyTorch support, added section for PyTorch model conversion
   * - v0.8.0
     - 2023.02.28
     - Added pre-processing using intelligent deep learning processor
   * - v0.6.0
     - 2022.11.05
     - Added section for mixed precision operation process
   * - v0.5.0
     - 2022.10.20
     - Added model-zoo specification to test all models within
   * - v0.4.0
     - 2022.09.20
     - Caffe support, added section for Caffe model conversion
   * - v0.3.0
     - 2022.08.24
     - TFLite support, added section for TFLite model conversion
   * - v0.2.0
     - 2022.08.02
     - Added chapter for running test samples in SDK
   * - v0.1.0
     - 2022.07.29
     - Initial release, supports ``resnet/mobilenet/vgg/ssd/yolov5s`` with yolov5s as example case
