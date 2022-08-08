import os
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--debug', type=str, default='', help='debug')
opt = parser.parse_args()

postprocess_type_all=[
  "--count 100 --dataset_type imagenet --postprocess_type topx --dataset /data/ILSVRC2012_img_val_with_subdir/",
  "--count 100 --dataset_type coco --postprocess_type coco_mAP --dataset /data/coco/val2017 --coco_annotation /data/coco/instances_val2017.json"
]

model_list_all={
# classification
  #'网络名':[]
  "resnet18":[0],
  # "resnet50_v2":[0],
#   "mobilenet_v2":[0],
#   "squeezenet1.0":[0],
#   "vgg16":[0],
# # object detection
#   "resnet34_ssd1200":[1],
  "yolov5s":[1]
}

def worker(i, cmd_line):
    print('idx:{}, cmd_line is runing:'.format(i), cmd_line)
    os.system(cmd_line)
    print('idx:{}, cmd_line end:'.format(i), cmd_line)


if __name__ == "__main__":
  po = Pool(4)

  for model in model_list_all:
    cmd_line = 'rm -rf {model};$REGRESSION_PATH/eval/gen_model.sh {model}'.format(model=model)
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

  i = 0
  for model in model_list_all:
    cmd_str = postprocess_type_all[model_list_all[model][0]]
    # cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_opt.onnx {cmd_str} >log_{model}_opt.onnx 2>&1'.format(model=model, cmd_str=cmd_str)
    # po.apply_async(worker, (i, cmd_line,))

    cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}.mlir {cmd_str} >log_{model}.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
    po.apply_async(worker, (i, cmd_line,))
    i += 1

    cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_sym.mlir {cmd_str} >log_{model}_bm1684x_tpu_int8_sym.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
    po.apply_async(worker, (i, cmd_line,))
    i += 1

    cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_asym.mlir {cmd_str} >log_{model}_bm1684x_tpu_int8_asym.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
    po.apply_async(worker, (i, cmd_line,))
    i += 1

  print('before join')
  po.close()
  po.join()
  print('all end')
