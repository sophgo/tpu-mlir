Accuracy Validation
=====================

Introduction
------------

Objects
~~~~~~~~~~~~
The accuracy validation in TPU-MLIR is mainly for the mlir model, fp32 uses the mlir model of the top layer while the int8 symmetric and asymmetric quantization uses the mlir model of the tpu layer.

Metrics
~~~~~~~~~~~~
Currently, the validation is mainly used for classification and object detection networks. The metrics for classification networks are Top-1 and Top-5 accuracy, while the object detection networks use 12 metrics of COCO, as shown below. Generally, we record the Average Precision when IoU=0.5 (i.e., PASCAL VOC metric).

.. math::

   \boldsymbol{Average Precision (AP):} & \\
   AP\quad & \text{\% AP at IoU=.50:.05:.95 (primary challenge metric)} \\
   AP^{IoU}=.50\quad & \text{\% AP at IoU=.50 (PASCAL VOC metric)} \\
   AP^{IoU}=.75\quad & \text{\% AP at IoU=.75 (strict metric)} \\
   \boldsymbol{AP Across Scales:} & \\
   AP^{small}\quad & \text{\% AP for small objects: $area < 32^2$} \\
   AP^{medium}\quad & \text{\% AP for medium objects: $32^2 < area < 96^2$} \\
   AP^{large}\quad & \text{\% AP for large objects: $area > 96^2$} \\
   \boldsymbol{Average Recall (AR):} & \\
   AR^{max=1}\quad & \text{\% AR given 1 detection per image} \\
   AR^{max=10}\quad & \text{\% AR given 10 detections per image} \\
   AR^{max=100}\quad & \text{\% AR given 100 detections per image} \\
   \boldsymbol{AP Across Scales:} & \\
   AP^{small}\quad & \text{\% AP for small objects: $area < 32^2$} \\
   AP^{medium}\quad & \text{\% AP for medium objects: $32^2 < area < 96^2$} \\
   AP^{large}\quad & \text{\% AP for large objects: $area > 96^2$}


Datasets
~~~~~~~~~~~~
In addition, the dataset used for validation needs to be downloaded by yourself. Classification networks use the validation set of ILSVRC2012 (50,000 images, https://www.image-net.org/challenges/LSVRC/2012/). There are two ways to place the images in the dataset. One is that there are 1000 subdirectories under the dataset directory, corresponding to 1000 classes, and each class has 50 images. In this case, no additional label file is required. The other way is that all images are in the same dataset directory, and there is an additional label file. According to the sequence of images' names, each line in the txt file uses a number from 1 to 1000 to indicate the class of each image.

Object detection networks use the COCO2017 validation set (5000 images, https://cocodataset.org/#download). All images are under the same dataset directory. The corresponding json label file needs to be downloaded as well.

Validation Interface
--------------------

TPU-MLIR provides the command for accuracy validation:

.. code-block:: shell

    $ model_eval.py \
        --model_file mobilenet_v2.mlir \
        --count 50 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

The supported parameters are shown below:

.. list-table:: Function of model_eval.py parameters
   :widths: 20 12 50
   :header-rows: 1

   * - Name
     - Required?
     - Explanation
   * - model_file
     - Y
     - Model file
   * - dataset
     - N
     - Directory of dataset
   * - dataset_type
     - N
     - Dataset type. Currently mainly supports imagenet, coco. The default is imagenet
   * - postprocess_type
     - Y
     - Metric. Currently supports topx and coco_mAP
   * - label_file
     - N
     - txt label file, which may be needed when validating the accuracy of classification networks
   * - coco_annotation
     - N
     - json label file, required when validating object detection networks
   * - count
     - N
     - The number of images used for validation. The default is to use the entire dataset.


Validation Example
------------------
In this section, mobilenet_v2 and yolov5s are used as the representative of the classification network and the object detection network for accuracy validation.

mobilenet_v2
~~~~~~~~~~~~~
1. Dataset Downloading

   Download the ILSVRC2012 validation set to the datasets/ILSVRC2012_img- _val_with_subdir directory. Images of the dataset are placed in subdirectories, so no additional label files are required.

2. Model Conversion

   Use the model_transform.py interface to convert the original model to the mobilenet_v2.mlir model, and obtain mobilenet_v2_cali_table through the run_calibration.py interface. Please refer to the "User Interface" chapter for specific usage. The INT8 model of the tpu layer is obtained through the command below. After running the command, an intermediate file named mobilenet_v2_bm1684x_int8_sym_tpu.mlir will be generated. We will use this intermediate file to validate the accuracy of the INT8 symmetric quantized model:

.. code-block:: shell

    # INT8 Sym Model
    $ model_deploy.py \
       --mlir mobilenet_v2.mlir \
       --quantize INT8 \
       --calibration_table mobilenet_v2_cali_table \
       --processor BM1684X \
       --test_input mobilenet_v2_in_f32.npz \
       --test_reference mobilenet_v2_top_outputs.npz \
       --tolerance 0.95,0.69 \
       --model mobilenet_v2_int8.bmodel

3. Accuracy Validation

   Use the model_eval.py interface to validate:

.. code-block:: shell

    # F32 model validation
    $ model_eval.py \
        --model_file mobilenet_v2.mlir \
        --count 50000 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

    # INT8 sym model validation
    $ model_eval.py \
        --model_file mobilenet_v2_bm1684x_int8_sym_tpu.mlir \
        --count 50000 \
        --dataset_type imagenet \
        --postprocess_type topx \
        --dataset datasets/ILSVRC2012_img_val_with_subdir

The accuracy validation results of the F32 model and the INT8 symmetric quantization model are as follows:

.. code-block:: shell

    # mobilenet_v2.mlir validation result
    2022/11/08 01:30:29 - INFO : idx:50000, top1:0.710, top5:0.899
    INFO:root:idx:50000, top1:0.710, top5:0.899

    # mobilenet_v2_bm1684x_int8_sym_tpu.mlir validation result
    2022/11/08 05:43:27 - INFO : idx:50000, top1:0.702, top5:0.895
    INFO:root:idx:50000, top1:0.702, top5:0.895

yolov5s
~~~~~~~~~~~~~

1. Dataset Downloading

   Download the COCO2017 validation set to the datasets/val2017 directory, which contains 5,000 images for validation. The corresponding label file instances_val2017.json is downloaded to the datasets directory.

2. Model Conversion

   The conversion process is similar to mobilenet_v2.

3. Accuracy Validation

   Use the model_eval.py interface to validate:

.. code-block:: shell

    # F32 model validation
    $ model_eval.py \
        --model_file yolov5s.mlir \
        --count 5000 \
        --dataset_type coco \
        --postprocess_type coco_mAP \
        --coco_annotation datasets/instances_val2017.json \
        --dataset datasets/val2017

    # INT8 sym model validation
    $ model_eval.py \
        --model_file yolov5s_bm1684x_int8_sym_tpu.mlir \
        --count 5000 \
        --dataset_type coco \
        --postprocess_type coco_mAP \
        --coco_annotation datasets/instances_val2017.json \
        --dataset datasets/val2017

The accuracy validation results of the F32 model and the INT8 symmetric quantization model are as follows:

.. code-block:: shell

    # yolov5s.mlir validation result
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.470
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.300
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.359
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.602
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

    # yolov5s_bm1684x_int8_sym_tpu.mlir validation result
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.544
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.382
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.432
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.473
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.514
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.337
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636




