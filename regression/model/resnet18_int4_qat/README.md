1,modify bm1686_support_int4_sym=1 in regression/chip.cfg

2,convert int4 qat fp32 model to int4:
cd tpu-mlir/regression
./run_model.sh resnet18_int4_qat bm1686 basic

3,run eval:
cd regression_out/resnet18_int4_qat_bm1686
$REGRESSION_PATH/../python/tools/model_eval.py --model_file resnet18_int4_qat_bm1686_tpu_int4_sym.mlir --count 10 --dataset_type imagenet --postprocess_type topx --dataset /workspace/datasets/ILSVRC2012_img_val_with_subdir/
notice: you must modify dataset path
