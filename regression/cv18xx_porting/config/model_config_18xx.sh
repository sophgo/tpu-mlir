#!/bin/bash
set -e
######################################################
# when copy from tpu_compile need add option as below:
# 1.add INPUT_SHAPE
# 2.modify INPUT_SCALE (each channel)
# 3.modify CALI_TABLE path (don't include input num)
# 4.add CALI_IMAGES
# 5.add IMAGE_PATH (default cat.jpg)
# 6.modify TOLERANCE(remove duplicate cos sim)
######################################################

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export STD=1,1,1
export MODEL_CHANNEL_ORDER="bgr"
export EXCEPTS=-
export EXCEPTS_BF16=-
export CALIBRATION_IMAGE_COUNT=1000
export DO_QUANT_INT8=1
export DO_FUSED_PREPROCESS=1
export DO_FUSED_POSTPROCESS=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_ACCURACY_FUSED_PREPROCESS=0
export EVAL_MODEL_TYPE="imagenet"
export LABEL_FILE=$REGRESSION_PATH/data/synset_words.txt
export BGRAY=0
export RESIZE_KEEP_ASPECT_RATIO=0
export TOLERANCE_FP32=0.999,0.999,0.998
export TOLERANCE_INT8_CMDBUF=0.99,0.99,0.99
export DO_QUANT_BF16=1
export INT8_MODEL=0
export MIX_PRECISION_TABLE='-'1
export INPUT_NUM=100

# default inference and test image
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/cat.jpg

if [ $NET = "arcface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/Aaron_Eckhart_0001.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export INPUT_SHAPE=[[1,3,112,112]]
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=112,112
export IMAGE_RESIZE_DIMS=112,112
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export TOLERANCE_INT8=0.978,0.763
export EXCEPTS=data
export TOLERANCE_BF16=0.999,0.989
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/lfw/lfw
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
# replace $REGRESSION_PATH/data/cali_tables/ with $REGRESSION_PATH/cv18xx_porting/cali_tables/
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]  # new attr
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0      # value per-channel
export INPUT=data
export OUTPUTS=prob
export TOLERANCE_INT8=0.975,0.768     # 2 value
export TOLERANCE_BF16=0.998,0.990     # 2 value
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EXCEPTS=prob
export MODEL_CHANNEL_ORDER="bgr"      # set channel order
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012 # set calibration dateset
fi

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0,1.0,1.0
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc1000
export TOLERANCE_INT8=0.96,0.71
export TOLERANCE_MIX_PRECISION=0.96,0.95,0.73
export MIX_PRECISION_BF16_LAYER_NUM=10
export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "squeezenet_v1.0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.0.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,227,227]]
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
export OUTPUTS=pool10
export TOLERANCE_INT8=0.973,0.762
export TOLERANCE_BF16=0.999,0.987
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "mobilenet_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.96,0.73
export TOLERANCE_BF16=0.99,0.92
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.94,0.66
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,299,299]]
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.95,0.68
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.94
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,299,299]]
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.92,0.60
export TOLERANCE_BF16=0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.93
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=100
fi

if [ $NET = "densenet_201" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc6
export TOLERANCE_INT8=0.76,0.28
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=100
fi

if [ $NET = "resnext50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnext/caffe/resnext50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnext/caffe/resnext50.caffemodel
export LABEL_FILE=$MODEL_PATH/imagenet/resnext/caffe/corresp.txt
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
export OUTPUTS=prob
export EXCEPTS=data
export TOLERANCE_INT8=0.96,0.70
export TOLERANCE_BF16=0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.97
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=1000
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
export OUTPUTS=fc8
export TOLERANCE_INT8=0.997,0.928
export TOLERANCE_BF16=0.999,0.994
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export MODEL_CHANNEL_ORDER="bgr"
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "retinaface_mnet25_600" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,600,600]]
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1,1,1
export TOLERANCE_INT8=0.90,0.54
export TOLERANCE_BF16=0.99,0.88
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val/images
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export EXCEPTS=data
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600_with_detection.prototxt
fi

if [ $NET = "mobilenet_ssd" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/labelmap_voc.prototxt
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export INPUT_SHAPE=[[1,3,300,300]]
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.007843,0.007843,0.007843
export TOLERANCE_INT8=0.91,0.58   #0.96,0.67
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export EXCEPTS=detection_out
export EVAL_MODEL_TYPE="voc2012"
export EVAL_SCRIPT_VOC="eval_detector_voc.py"
export MODEL_CHANNEL_ORDER="bgr"
export CALI_IMAGES=$DATA_SET/VOCdevkit/VOC2012/JPEGImages
fi

if [ $NET = "yolo_v2_1080" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy_1080.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,1080,1920]]
export IMAGE_RESIZE_DIMS=1080,1920
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=1080,1920
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export TOLERANCE_INT8=0.90,0.50
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.93
export DO_QUANT_BF16=0
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,416,416]]
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export TOLERANCE_INT8=0.92,0.59
export EXCEPTS='output'
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export YOLO_V3=1
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "yolo_v3_spp" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,608,608]]
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export EXCEPTS=output
export TOLERANCE_INT8=0.86,0.32
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export YOLO_V3=1
export SPP_NET=1
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "icnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_bnnomerge.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_trainval_90k_bnnomerge.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/0.png
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,1025,2049]]
export NET_INPUT_DIMS=1025,2049
export IMAGE_RESIZE_DIMS=1025,2049
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
export OUTPUTS=conv5_3_pool1_interp
export CALIBRATION_IMAGE_COUNT=30
export TOLERANCE_INT8=0.85,0.41
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EVAL_MODEL_TYPE="isbi"
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val
fi

if [ $NET = "segnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/segnet/caffe/segnet_model_driving_webdemo_fix.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/segnet/caffe/segnet_weights_driving_webdemo.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/camvid.png
export COLOURS_LUT=$REGRESSION_PATH/data/camvid12_lut.png
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export OUTPUTS=conv1_1_D
export IMAGE_RESIZE_DIMS=360,480
export NET_INPUT_DIMS=360,480
export MEAN=0,0,0
export STD=1.0,1.0,1.0
export RAW_SCALE=255.0
export INPUT_SCALE=1.0
export TOLERANCE_FP32=0.999,0.999,0.977
export TOLERANCE_INT8=0.91,0.90,0.57
export EXCEPTS=upsample2,upsample1,pool1_mask,pool2_mask
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.98,0.87
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi
