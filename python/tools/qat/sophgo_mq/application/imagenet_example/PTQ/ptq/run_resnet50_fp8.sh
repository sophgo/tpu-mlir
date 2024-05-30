#!/bin/bash
# run FP8(E4M3/E5M2)FakeQuantize resnet50
python3 ptq_main.py \
	--arch=resnet50 \
	--batch-size=64 \
	--cali-batch-num=16 \
	--data_path=/data/imagenet/for_train_val \
	--backend=sophgo_tpu \
	--seed=1005 \
	--pretrained \
	--quantize_type=naive_ptq \
	--fp8
