#!/bin/bash
for model in "mobilenet_v2_cf" "resnet18_cf" "segnet_cf" "mobilenet_v2_cf_bs4" "retinaface_mnet_with_det" "arcface_res50" "yolov3_416_with_det" "enet_cf" "erfnet_cf" "googlenet_cf" "icnet_cf" "inception_v4_cf" "ssd300_cf" "yolov3_spp_cf" "yolov4_cf";
do
	for chip_id in "bm1684x" "bm1686":
	do
		python3 run_model.py  $model --chip ${chip_id} >>/workspace/model_regression_log/${model}_1684x.log
	done
done
