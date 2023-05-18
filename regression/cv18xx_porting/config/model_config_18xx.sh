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
export RAW_SCALE=255.0
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
export LABEL_FILE=$REGRESSION_PATH/cv18xx_porting/data/synset_words.txt
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
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export TOLERANCE_INT8=0.96,0.70  # 0.97,0.74
export EXCEPTS=data
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EVAL_MODEL_TYPE="lfw"
export CALI_IMAGES=$DATA_SET/lfw/lfw
fi

if [ $NET = "fcn-8s" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/fcn-8s/caffe/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/fcn-8s/caffe/fcn-8s-pascalcontext.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,500,500]]
export NET_INPUT_DIMS=500,500
export IMAGE_RESIZE_DIMS=500,500
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=input
export TOLERANCE_INT8=0.99,0.86
export TOLERANCE_BF16=0.999,0.993
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
# replace $REGRESSION_PATH/cv18xx_porting/data/cali_tables/ with $REGRESSION_PATH/cv18xx_porting/cali_tables/
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=prob
export TOLERANCE_INT8=0.92,0.60  # 0.97,0.76
export TOLERANCE_BF16=0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
#export EXCEPTS=prob
export MODEL_CHANNEL_ORDER="bgr"
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
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0,1.0,1.0
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
#export OUTPUTS=fc1000
export TOLERANCE_INT8=0.96,0.74  # 0.96,0.75
export TOLERANCE_MIX_PRECISION=0.96,0.95,0.73
export MIX_PRECISION_BF16_LAYER_NUM=10
export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "squeezenet_v1.0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.0.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,227,227]]
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=pool10
export TOLERANCE_INT8=0.96,0.72  # 0.80,0.34
export TOLERANCE_BF16=0.99,0.94
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
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
#export OUTPUTS=fc7
export TOLERANCE_INT8=0.98,0.81 # 0.96,0.73
export TOLERANCE_BF16=0.98,0.85 #for yuv format, before is (0.99,0.92)
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
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
#export OUTPUTS=fc7
export TOLERANCE_INT8=0.97,0.75 #for yuv420
export TOLERANCE_BF16=0.98,0.82 #for yuv420
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
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
#export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.96,0.71  # 0.95,0.68
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
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
#export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.92,0.61  # 0.93,0.63
export TOLERANCE_BF16=0.99,0.93
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
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
#export OUTPUTS=fc6
export TOLERANCE_INT8=0.83,0.40  # 0.62,0.03
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
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=prob
export EXCEPTS=data
export TOLERANCE_INT8=0.96,0.72  # 0.91,0.56
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
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=input
#export OUTPUTS=fc8
export TOLERANCE_INT8=0.99,0.89
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
export MEAN=0,0,0
export INPUT_SCALE=1,1,1
export TOLERANCE_INT8=0.92,0.61  #0.90,0.54
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
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.007843,0.007843,0.007843
export TOLERANCE_INT8=0.97,0.79 #0.96,0.67
export TOLERANCE_BF16=0.99,0.95
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
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392 #1.0,1.0,1.0
export TOLERANCE_INT8=0.97,0.78  # 0.90,0.50
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.93
export DO_QUANT_BF16=0
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,416,416]]
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392 #1.0,1.0,1.0
export TOLERANCE_INT8=0.95,0.68  # 0.933,0.623
export EXCEPTS='output'
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416_with_detection.prototxt
export TOLERANCE_BF16=0.97,0.75  #set for yuv format, before is (0.99,0.93)
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export YOLO_V3=1
export CALI_IMAGES=$DATA_SET/coco/val2017/
export INPUT_NUM=200
fi

if [ $NET = "yolo_v3_spp" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,608,608]]
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392 #1.0,1.0,1.0
export EXCEPTS=output
export TOLERANCE_INT8=0.96,0.72  # 0.86,0.32
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
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=conv5_3_pool1_interp
export CALIBRATION_IMAGE_COUNT=30
export TOLERANCE_INT8=0.94,0.65  # 0.85,0.41
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EVAL_MODEL_TYPE="isbi"
export CALI_IMAGES=$DATA_SET/VOCdevkit/VOC2012/JPEGImages
export INPUT_NUM=30
fi

if [ $NET = "segnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/segnet/caffe/segnet_model_driving_webdemo_fix.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/segnet/caffe/segnet_weights_driving_webdemo.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/camvid.png
export COLOURS_LUT=$REGRESSION_PATH/cv18xx_porting/data/camvid12_lut.png
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
#export OUTPUTS=conv1_1_D
export INPUT_SHAPE=[[1,3,360,480]]
export IMAGE_RESIZE_DIMS=360,480
export NET_INPUT_DIMS=360,480
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export EXCEPTS=pool2_D,pool1_D
export DO_QUANT_BF16=0
export TOLERANCE_TOP=0.999,0.977
export TOLERANCE_INT8=0.89,0.53  # 0.90,0.55
export TOLERANCE_BF16=0.99,0.87
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/VOCdevkit/VOC2012/JPEGImages
fi

if [ $NET = "resnet_res_blstm" ]; then
export MODEL_DEF=$MODEL_PATH/rnn/resnet_res_blstm/caffe/deploy_fix.prototxt
export MODEL_DAT=$MODEL_PATH/rnn/resnet_res_blstm/caffe/model.caffemodel
export LABEL_MAP=$MODEL_PATH/rnn/resnet_res_blstm/caffe/label.txt
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/poem.jpg
export INPUT_SHAPE=[[1,3,32,280]]
export NET_INPUT_DIMS=32,280
export IMAGE_RESIZE_DIMS=32,280
export MEAN=152,152,152
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=input
#export OUTPUTS=fc1x
export TOLERANCE_INT8=0.98,0.80 # 0.99,0.79
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_POSTPROCESS=1
export POSTPROCESS_SCRIPT=$REGRESSION_PATH/cv18xx_porting/data/run_postprocess/ctc_greedy_decoder.sh
export CALI_IMAGES=$DATA_SET/OCR/images
#https://github.com/senlinuc/caffe_ocr
fi

if [ $NET = "blazeface" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/blazeface/onnx/blazeface.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export INPUT_SHAPE=[[1,3,128,128]]
export NET_INPUT_DIMS=128,128 # h,w
export IMAGE_RESIZE_DIMS=128,128
export CALIBRATION_IMAGE_COUNT=1
export MEAN=1,1,1
export INPUT_SCALE=0.0078,0.0078,0.0078  #1.0,1.0,1.0
export INPUT=input
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export TOLERANCE_INT8=0.960,0.717  # quant by double type
export TOLERANCE_INT8=0.95,0.68  # 0.955,0.699
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.96 #
export DO_PREPROCESS=0
export BGRAY=0
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val
# just compare last one
fi

if [ $NET = "faceboxes" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/faceboxes/onnx/faceboxes.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export INPUT_SHAPE=[[1,3,915,1347]]
export NET_INPUT_DIMS=915,1347
export IMAGE_RESIZE_DIMS=915,1347
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input.1
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export TOLERANCE_INT8=0.49,-0.26  #0.70,-0.10
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.96 #
export DO_PREPROCESS=0
export BGRAY=0
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val
# just compare last one
fi

if [ $NET = "unet" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/segmentation/unet/onnx/unet.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/0.png
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,1,256,256]]
export NET_INPUT_DIMS=256,256
export IMAGE_RESIZE_DIMS=256,256
export MEAN=0
export INPUT_SCALE=1.0
export INPUT=input
export MODEL_CHANNEL_ORDER="gray"
##export OUTPUTS=prob
export CALIBRATION_IMAGE_COUNT=30
export TOLERANCE_INT8=0.99,0.90 #0.99,0.91
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export EVAL_MODEL_TYPE="isbi"
export BGRAY=1
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val
fi

if [ $NET = "res2net50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/res2net/onnx/res2net50_48w_2s.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MEAN=123.675,116.28,103.53 #     # in RGB
export INPUT_SCALE=0.0171,0.0175,0.0174   # 1.0,1.0,1.0
export INPUT=input
# export OUTPUTS=output
export TOLERANCE_INT8=0.95,0.69  # 0.93,0.63
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012 # set calibration dateset
# export BATCH_SIZE=4
fi

if [ $NET = "resnetv2" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnetv2_tf_50_10.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=123.675,116.28,103.53 #   0.485,0.456,0.406  # in RGB
export INPUT_SCALE=0.0171,0.0175,0.0174   # 1.0,1.0,1.0
export INPUT=input
##export OUTPUTS=output
export EXCEPTS=resnet_v2_50/predictions/Reshape_1:0_Softmax
export TOLERANCE_INT8=0.86,0.45  # 0.83,0.40  use custom calibration table
export TOLERANCE_BF16=0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012 # set calibration dateset
fi

if [ $NET = "yolox_s" ]; then
# onnx: IoU 0.5:0.95 0.363, IoU 0.50 0.541, IoU 0.75 0.389
# int8: IoU 0.5:0.95 0.344, IoU 0.50 0.515, IoU 0.75 0.373
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolox/onnx/yolox_s.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="bgr"
export INPUT_SHAPE=[[1,3,640,640]]
export IMAGE_RESIZE_DIMS=640,640
export NET_INPUT_DIMS=640,640
export MEAN=0.,0.,0.
export INPUT_SCALE=1.0,1.0,1.0
export EXCEPTS="796_Sigmoid" # 0.873364, 0.873364, 0.347177
export TOLERANCE_FP32=0.99,0.99
export TOLERANCE_BF16=0.98,0.82
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_INT8=0.87,0.59 #set for yuv format, before is (0.87,0.6)
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_ONNX="eval_yolox.py"
export EVAL_SCRIPT_INT8="eval_yolox.py"
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "yolact" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/yolact/onnx/yolact_resnet50_coco_4outputs.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export INPUT_SHAPE=[[1,3,550,550]]
export NET_INPUT_DIMS=550,550
export IMAGE_RESIZE_DIMS=550,550
export MEAN=0,0,0
export INPUT_SCALE=0.0039215686,0.0039215686,0.0039215686
export INPUT=input
export DO_QUANT_BF16=0
export TOLERANCE_INT8=0.96,0.74
export TOLERANCE_BF16=0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export CALI_IMAGES=$DATA_SET/coco/val2017/
fi

if [ $NET = "squeezenet_v1.1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,227,227]]
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=pool10
export TOLERANCE_INT8=0.93,0.62  # 0.9,0.55
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.93
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "densenet_121" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_NUM=1000
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
#export OUTPUTS=fc6
export TOLERANCE_INT8=0.90,0.55  # 0.82,0.37
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "senet_res50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_NUM=1000
export INPUT_SHAPE=[[1,3,225,225]]
export NET_INPUT_DIMS=225,225
export IMAGE_RESIZE_DIMS=256,256
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
#export OUTPUTS=fc6
export TOLERANCE_INT8=0.97,0.77  # 0.96,0.73
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "retinaface_mnet25" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,320,320]]
export IMAGE_RESIZE_DIMS=320,320
export NET_INPUT_DIMS=320,320
export MEAN=0,0,0
export INPUT_SCALE=1,1,1
export TOLERANCE_INT8=0.94,0.65  # 0.90,0.54
# accuracy setting
export NET_INPUT_DIMS=320,320
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATA_SET/widerface/WIDER_val/images
export ANNOTATION=$DATA_SET/widerface/wider_face_split
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.92
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val/images
fi

if [ $NET = "retinaface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,600,600]]
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export MEAN=0,0,0
export INPUT_SCALE=1,1,1
export TOLERANCE_INT8=0.91,0.57 #0.86,0.49
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.87
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val/images
export INPUT_NUM=300
fi

if [ $NET = "ssd300" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="bgr"
export INPUT_SHAPE=[[1,3,300,300]]
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export MEAN=104.0,117.0,123.0
export INPUT_SCALE=1.0,1.0,1.0
export TOLERANCE_INT8=0.97,0.75  #0.91,0.52
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.91
export EXCEPTS=detection_out
export CALI_IMAGES=$DATA_SET/coco/val2017
export EVAL_MODEL_TYPE="coco"
# export EVAL_SCRIPT_CAFFE="eval_caffe_detector_ssd.py"
export EVAL_SCRIPT_INT8="eval_ssd.py"
fi

if [ $NET = "yolo_v1_448" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,448,448]]
export IMAGE_RESIZE_DIMS=448,448
export NET_INPUT_DIMS=448,448
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export TOLERANCE_INT8=0.98,0.81 #0.90,0.50
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v3_608" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_NUM=100
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,608,608]]
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export DO_QUANT_BF16=0
# export TOLERANCE_INT8=0.886,0.507 #0.92,0.55
export TOLERANCE_INT8=0.94,0.66 #0.92,0.55
export TOLERANCE_BF16=0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export YOLO_V3=1
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v3_320" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_NUM=100
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,320,320]]
export IMAGE_RESIZE_DIMS=320,320
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=320,320
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export TOLERANCE_INT8=0.95,0.68 # 0.92,0.60
export TOLERANCE_BF16=0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export YOLO_V3=1
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v3_tiny" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,416,416]]
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export EXCEPTS=output
export TOLERANCE_INT8=0.98,0.80  # 0.93,0.54
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export YOLO_V3=1
export TINY=1
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "faster_rcnn" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/frcn/caffe/faster_rcnn.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/frcn/caffe/faster_rcnn.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export INPUT_SHAPE=[[1,3,600,800]]
export IMAGE_RESIZE_DIMS=600,800
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=600,800
export MEAN=102.9801,115.9465,122.7717
export INPUT_SCALE=1.0,1.0,1.0
export EXCEPTS=rois,pool5,bbox_pred,output
export TOLERANCE_INT8=0.77,0.29 # TODO 0.83,0.40
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EXCEPTS_BF16=proposal,roi_pool5,roi_pool5_quant,fc6_reshape,relu6,relu7,cls_score,cls_score_dequant,bbox_pred,bbox_pred_dequant,cls_prob #output is euclidean_similarity   = 0.995603
export CALI_IMAGES=$DATA_SET/VOCdevkit/VOC2012/JPEGImages
fi

if [ $NET = "enet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/enet/caffe/enet_deploy_final.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/enet/caffe/cityscapes_weights.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/city.png
export COLOURS_LUT=$REGRESSION_PATH/cv18xx_porting/data/city_lut.png
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,512,1024]]
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
#export OUTPUTS=deconv6_0_0
export TOLERANCE_INT8=0.90,0.55  # 0.855,0.441
export TOLERANCE_BF16=0.96,0.74
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export EXCEPTS=pool1_0_4_mask_convert,pool2_0_4_mask_convert,conv2_7_1_a,prelu2_7_0,prelu2_7_1,prelu3_3_0,conv3_3_1_a,prelu3_3_1,prelu4_0_4,upsample4_0_4,upsample5_0_4
# export BATCH_SIZE=4
export CALI_IMAGES=$DATA_SET/cityscaps/val
fi

if [ $NET = "efficientnet_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
#export INPUT_NUM=1000
export INPUT_SHAPE=[[1,3,224,224]]
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export MODEL_CHANNEL_ORDER="rgb"
#export MEAN=0.485,0.456,0.406
export MEAN=123.675,116.28,103.53  # in RGB,
export INPUT_SCALE=0.01712,0.017507,0.017429
export INPUT=input
#export OUTPUTS=output
export EXCEPTS=424_Mul,422_Conv,388_Sigmoid
export TOLERANCE_INT8=0.79,0,32 # 0.68,0.13
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "efficientnet-lite_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-lite/b0/onnx/efficientnet_lite.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127,127,127
export INPUT_SCALE=0.0078125,0.0078125,0.0078125  # 1.0
export INPUT=input
##export OUTPUTS=output
export TOLERANCE_INT8=0.98,0.80 #0.989,0.831
export TOLERANCE_BF16=0.999,0.961
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "espcn_3x" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/super_resolution/espcn/onnx/espcn_3x.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,85,85]]
export IMAGE_RESIZE_DIMS=85,85
export NET_INPUT_DIMS=85,85
export MEAN=0,0,0
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=input
# cannot do fuse preprocessing, bcuz this model use
# PIL.Image to do resize in ANTIALIAS mode,
# which is not support in opencv
export DO_FUSED_PREPROCESS=0
export TOLERANCE_INT8=0.98,0.80
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "ecanet50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/ecanet/onnx/ecanet50.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
#export MEAN=0.485,0.456,0.406
export MEAN=123.675,116.28,103.53
export INPUT_SCALE=0.01712,0.017507,0.017429
export INPUT=input
#export OUTPUTS=output
export TOLERANCE_INT8=0.96,0.74 # 0.96,0.71
export TOLERANCE_BF16=0.999,0.977 # 0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "efficientdet_d0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/efficientdet-d0/onnx/efficientdet-d0.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,512,512]]
export IMAGE_RESIZE_DIMS=512,512
export NET_INPUT_DIMS=512,512
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=123.675,116.28,103.53
export INPUT_SCALE=0.01712,0.017507,0.017429
export INPUT=input
#export OUTPUTS=2249,3945,5305
export DO_QUANT_BF16=0
export EXCEPTS=2367_Mul,2365_Conv
export TOLERANCE_INT8=0.82,0.28  # 0.78,0.19
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v2_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,416,416]]
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export MEAN=0,0,0
export INPUT_SCALE=0.0039215686,0.0039215686,0.0039215686
export TOLERANCE_INT8=0.97,0.76 # 0.89,0.48
export TOLERANCE_BF16=0.999,0.987 # 0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.99
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v4" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.caffemodel
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,608,608]]
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export MEAN=0,0,0
export INPUT_SCALE=0.0039215686,0.0039215686,0.0039215686
export DO_QUANT_BF16=0
# mish layer
export EXCEPTS="layer137-conv,layer138-conv/1,layer138-conv,layer142-conv,layer149-conv,layer149-conv/1"
#export OUTPUTS="layer139-conv,layer150-conv,layer161-conv"
export TOLERANCE_INT8=0.92,0.60  # 0.88,0.46
export TOLERANCE_BF16=0.99,0.94 # 0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export CALI_IMAGES=$DATA_SET/coco/val2017
export YOLO_V4=1
fi

if [ $NET = "yolo_v4_s" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/onnx/yolov4-csp-s-leaky.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export EVAL_SCRIPT=accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,640,640]]
export NET_INPUT_DIMS=640,640
export IMAGE_RESIZE_DIMS=640,640
export RESIZE_KEEP_ASPECT_RATIO=1
export MEAN=0,0,0
export INPUT_SCALE=0.0039215686,0.0039215686,0.0039215686
export TOLERANCE_INT8=0.98,0.81
export TOLERANCE_BF16=0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export CALI_IMAGES=$DATA_SET/coco/val2017
export YOLO_V4=1
fi

if [ $NET = "yolo_v5_s" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v5/onnx/yolov5s.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export INPUT_SHAPE=[[1,3,640,640]]
export NET_INPUT_DIMS=640,640 # h,w
export IMAGE_RESIZE_DIMS=640,640
export RESIZE_KEEP_ASPECT_RATIO=1
export CALIBRATION_IMAGE_COUNT=1000
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=input
export TOLERANCE_INT8=0.96,0.71 #0.97,0.76 #set for yuv format, before is (0.96,0.74)
export TOLERANCE_BF16=0.98,0.84 #set for yuv format, before is (0.99,0.96)
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export TOLERANCE_FP32=0.99,0.99
export DO_PREPROCESS=0
export BGRAY=0
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_ONNX="eval_yolo_v5.py"
export EVAL_SCRIPT_INT8="eval_yolo_v5.py"
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolo_v5_m" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v5/onnx/yolov5m.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export INPUT_SHAPE=[[1,3,640,640]]
export NET_INPUT_DIMS=640,640 # h,w
export IMAGE_RESIZE_DIMS=640,640
export RESIZE_KEEP_ASPECT_RATIO=1
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=input
export TOLERANCE_INT8=0.96,0.71 #0.97,0.76 #set for yuv format, before is (0.96,0.74)
export TOLERANCE_BF16=0.98,0.84 #set for yuv format, before is (0.99,0.96)
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export TOLERANCE_FP32=0.99,0.99
export DO_PREPROCESS=0
export BGRAY=0
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_ONNX="eval_yolo_v5.py"
export EVAL_SCRIPT_INT8="eval_yolo_v5.py"
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "yolov5s-face" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/yolov5-face/onnx/yolov5s-face.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/parade.jpg
export INPUT_SHAPE=[[1,3,640,640]]
export NET_INPUT_DIMS=640,640 # h,w
export IMAGE_RESIZE_DIMS=640,640
export RESIZE_KEEP_ASPECT_RATIO=1
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=input
export TOLERANCE_INT8=0.97,0.76
export TOLERANCE_BF16=0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export TOLERANCE_FP32=0.99,0.99
export DO_PREPROCESS=0
export BGRAY=0
export CALI_IMAGES=$DATA_SET/widerface/WIDER_val/images
fi

if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=data
#export OUTPUTS=fc
export TOLERANCE_INT8=0.96,0.73  # 0.92,0.57
export TOLERANCE_BF16=0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EXCEPTS=data
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "yolo_v7" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v7/yolov7.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SHAPE=[[1,3,640,640]]
export IMAGE_RESIZE_DIMS=640,640
export NET_INPUT_DIMS=640,640
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export TOLERANCE_INT8=0.97,0.79  # 0.93,0.6
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_ONNX="eval_yolo_v7.py"
export EVAL_SCRIPT_INT8="eval_yolo_v7.py"
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "ppyolo_tiny" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/ppyolo/ppyolo_tiny.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
#export MIX_PRECISION_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/ppyolo_tiny_mix_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/dog.jpg
export RESIZE_KEEP_ASPECT_RATIO=0
export INPUT_SHAPE=[[1,3,320,320]]
export NET_INPUT_DIMS=320,320
export IMAGE_RESIZE_DIMS=320,320
export MEAN=123.675,116.28,103.53 #     # in RGB
export INPUT_SCALE=0.0171,0.0175,0.0174   # 1.0,1.0,1.0
export INPUT=input
export EXCEPTS=relu_17.tmp_0_Relu  # almost all zeros
export MODEL_CHANNEL_ORDER="rgb"
export TOLERANCE_INT8=0.78,0.31 #0.78,0.29
export TOLERANCE_BF16=0.98,0.82 #set for yuv format, before is 0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.98
export CALI_IMAGES=$DATA_SET/coco/val2017
fi

if [ $NET = "gaitset" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/gaitset/onnx/gaitset.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/002-bg-02-018-124.png
export INPUT_SHAPE=[[1,1,64,64]]
export IMAGE_RESIZE_DIMS=64,64
export NET_INPUT_DIMS=64,64
export MEAN=0,0,0
export INPUT_SCALE=0.00392,0.00392,0.00392 #1.0,1.0,1.0
export INPUT=input
export TOLERANCE_INT8=0.94,0.59  #0.954,0.636 # 0.95,0.7
export TOLERANCE_BF16=0.99,0.96
export EXCEPTS=97_Conv
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export MODEL_CHANNEL_ORDER="gray"
export BGRAY=1
export CALI_IMAGES=$DATA_SET/GaitDatasetB-silh
export INPUT_NUM=100
fi

if [ $NET = "alphapose" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/pose_256_192.jpg
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,256,192]]
export IMAGE_RESIZE_DIMS=256,192
export NET_INPUT_DIMS=256,192
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=103.53,116.535,122.399  #0.406,0.457,0.48 # in RGB
export INPUT_SCALE=0.00392,0.00392,0.00392
export INPUT=input
export EXCEPTS=404_Relu
export TOLERANCE_INT8=0.96,0.68  # 0.964,0.706 # 0.91,0.50
export TOLERANCE_BF16=0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export CALI_IMAGES=$DATA_SET/hico_20160224_det/images/test2015/
fi

if [ $NET = "erfnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_deploy_mergebn.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_cityscapes_mergebn.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/city.png
export COLOURS_LUT=$REGRESSION_PATH/cv18xx_porting/data/city_lut.png
export INPUT_SHAPE=[[1,3,512,1024]]
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export MEAN=0,0,0
export INPUT_SCALE=1.0,1.0,1.0
export CALIBRATION_IMAGE_COUNT=60
export INPUT=data
#export OUTPUTS=Deconvolution23_deconv
export TOLERANCE_INT8=0.88,0.47 # 0.78,0.25
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.91
export EXCEPTS=NBD19_add_conv1_3x1,NBD19_add_conv1_1x3,NBD19_add_conv2_3x1,NBD19_add_conv2_1x3
# export BATCH_SIZE=4
export CALI_IMAGES=$DATA_SET/cityscaps/val
export INPUT_NUM=60
fi

if [ $NET = "nasnet_mobile" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/nasnet_mobile/onnx/nasnet_mobile.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export MEAN=127.5,127.5,127.5  #0.5,0.5,0.5  # in BGR, pytorch mean=[0.5, 0.5, 0.5]
export INPUT_SCALE=0.007843,0.007843,0.007843   # 1.0
export IMAGE_RESIZE_DIMS=256,256
export INPUT=input
export TOLERANCE_INT8=0.96,0.71 # 0.77,0.277
export CALIBRATION_IMAGE_COUNT=2000
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "resnet18" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=123.675,116.28,103.53
export INPUT_SCALE=0.01712,0.017507,0.017429
export INPUT=input
#export OUTPUTS=output
export TOLERANCE_INT8=0.99,0.86
export TOLERANCE_BF16=0.999,0.987 # 0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi
