## yolov5s

`yolov5s.onnx`
From: <https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx>

`yolov5s.pt`
From: <https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt>

# mobilenet_v2

`mobilenet_v2_deploy.prototxt`,`mobilenet_v2.caffemodel`
From: <https://github.com/shicai/MobileNet-Caffe>

- Accuracy: Top-1 71.90, Top-5 90.49
- Preprocess
  - scale: 0.017
  - mirror: true
  - resize: 256,256
  - crop_size: 224,224
  - pixel: bgr
  - mean_value: [103.94, 116.78, 123.68]
