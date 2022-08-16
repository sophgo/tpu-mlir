import os
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--exclude', type=str, default='onnx', help='exclude')
opt = parser.parse_args()

postprocess_type_all=[
  "--count 50 --dataset_type imagenet --postprocess_type topx --dataset /data/ILSVRC2012_img_val_with_subdir/",
  "--count 0 --dataset_type coco --postprocess_type coco_mAP --dataset /data/coco/val2017 --coco_annotation /data/coco/instances_val2017.json"
]

model_list_all={
# classification
  #'网络名':[]
  "resnet18":[0],
  # "mobilenet_v2":[0],
  # "squeezenet1.0":[0],
  # "vgg16":[0],
  # "resnet50_v2":[0],
# object detection
  # "resnet34_ssd1200":[1],
  #"yolov5s":[1]
}

def worker(i, cmd_line):
    print('idx:{}, cmd_line is runing:'.format(i), cmd_line)
    os.system(cmd_line)
    print('idx:{}, cmd_line end:'.format(i), cmd_line)

if __name__ == "__main__":
  po = Pool(5)

  for model in model_list_all:
    cmd_line = 'rm -rf {model};$REGRESSION_PATH/eval/gen_model.sh {model}'.format(model=model)
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

  i = 0
  exclude = [i.strip() for i in opt.exclude.split(',')]
  for model in model_list_all:
    cmd_str = postprocess_type_all[model_list_all[model][0]]
    if 'onnx' not in exclude:
      preprocess_str = ''
      reg_path = os.getenv('REGRESSION_PATH')
      lines = open('{}/config/{}.cfg'.format(reg_path, model)).readlines()
      lines_dict = {line.strip().split('=')[0]:line.strip().split('=')[1] for line in lines if '=' in line}
      if 'input_shapes' in lines_dict:
        preprocess_str += ' --net_input_dims {}'.format(lines_dict['input_shapes'])
      if 'resize_dims' in lines_dict:
        preprocess_str += ' --resize_dims {}'.format(lines_dict['resize_dims'])
      if 'mean' in lines_dict:
        preprocess_str += ' --mean {}'.format(lines_dict['mean'])
      if 'scale' in lines_dict:
        preprocess_str += ' --scale {}'.format(lines_dict['scale'])
      if 'channel_format' in lines_dict:
        preprocess_str += ' --channel_format {}'.format(lines_dict['channel_format'])
      if 'pixel_format' in lines_dict:
        preprocess_str += ' --pixel_format {}'.format(lines_dict['pixel_format'])
      print('preprocess_str:', preprocess_str)

      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_opt.onnx {cmd_str} {preprocess} >log_{model}_opt.onnx 2>&1'.format(model=model, cmd_str=cmd_str, preprocess = preprocess_str)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'top' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}.mlir {cmd_str} >log_{model}.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'sym' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_sym.mlir {cmd_str} >log_{model}_bm1684x_tpu_int8_sym.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'asym' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_asym.mlir {cmd_str} >log_{model}_bm1684x_tpu_int8_asym.mlir 2>&1'.format(model=model, cmd_str=cmd_str)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

  po.close()
  po.join()
  print('all end')
