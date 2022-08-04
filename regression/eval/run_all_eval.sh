model_eval.py --model_file $REGRESSION_PATH/regression_out/resnet18/resnet18.mlir --dataset_type imagenet --postprocess_type topx --dataset /data/ILSVRC2012_img_val_with_subdir/ --count 1000
model_eval.py --model_file $REGRESSION_PATH/regression_out/yolov5s/yolov5s_tpu_int8_asym.mlir --dataset_type coco --postprocess_type coco_mAP --dataset /data/coco/val2017 --coco_annotation /data/coco/instances_val2017.json
model_eval.py --model_file $REGRESSION_PATH/regression_out/yolov5s/yolov5s_opt.onnx --dataset_type coco  --count 1000 --postprocess_type coco_mAP --dataset /data/coco/val2017/ --coco_annotation /data/coco/instances_val2017.json
model_eval.py --model_file $REGRESSION_PATH/regression_out/yolov5s/yolov5s_tpu_int8_sym_tuned.mlir --dataset_type coco    --count 1000 --postprocess_type coco_mAP --dataset /data/coco/val2017/ --coco_annotation /data/coco/instances_val2017.json

