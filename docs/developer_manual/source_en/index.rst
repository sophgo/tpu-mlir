TPU-MLIR Technical Reference Manual
=====================================

.. figure:: ../assets/sophon.png
   :width: 400px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

| **Legal Notice**
| Copyright © SOPHGO 2022. All rights reserved.
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
:Tel: +86-10-57590723
       +86-10-57590724

| **Release Record**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Version
     - Release Date
     - Explanation
   * - v0.1.0
     - 2022.08.10
     - Initial release, support ``resnet/mobilenet/vgg/ssd/yolov5s``，and use yolov5s as an example.


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Table of Contents
   :name: mastertoc
   :numbered:

   01_introduction
   02_environment
   03_user_interface
   04_over_design
   05_frontend
   06_quantization
   07_calibration
   08_lowering
   09_subnet
   10_layergroup
   11_memassign
   12_codegen
   13_mlir_define
   14_mlir_eval
