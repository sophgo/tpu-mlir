import os
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--exclude', type=str, default='onnx', help='exclude')
parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
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
  "yolov5s_qat":[1,'/workspace/yolov5/qat_output_ep15/yolov5s/yolov5s_mqmoble_cali_table_from_mqbench_sophgo_tpu']
}
def str2bool(v):
  return v.lower() in ("yes", "true", "1")

def worker(i, cmd_line):
    print('idx:{}, cmd_line is runing:'.format(i), cmd_line)
    os.system(cmd_line)
    print('idx:{}, cmd_line end:'.format(i), cmd_line)

if __name__ == "__main__":
  po = Pool(5)

  for model in model_list_all:
    cmd_line = 'rm -rf {};$REGRESSION_PATH/eval/gen_model.sh {} {}'.format(model, model, model_list_all[model][1] if len(model_list_all[model]) > 1 else '')
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

  debug_cmd = ''
  if opt.debug_cmd != '':
    debug_cmd = '--debug_cmd {}'.format(opt.debug_cmd)
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
        preprocess_str += ' --input_shapes {}'.format(lines_dict['input_shapes'])
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
      if 'pad_value' in lines_dict:
        preprocess_str += ' --pad_value {}'.format(lines_dict['pad_value'])
      if 'pad_type' in lines_dict:
        preprocess_str += ' --pad_type {}'.format(lines_dict['pad_type'])
      if 'keep_aspect_ratio' in lines_dict and str2bool(lines_dict['keep_aspect_ratio']):
        preprocess_str += ' --keep_aspect_ratio'
      print('preprocess_str:', preprocess_str)

      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_opt.onnx {cmd_str} {preprocess} {debug_cmd} >log_{model}_opt.onnx 2>&1'\
                .format(model=model, cmd_str=cmd_str, preprocess = preprocess_str, debug_cmd=debug_cmd)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'top' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}.mlir {cmd_str} {debug_cmd} >log_{model}.mlir 2>&1'\
                .format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'sym' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_sym.mlir {cmd_str} {debug_cmd} >log_{model}_bm1684x_tpu_int8_sym.mlir 2>&1'\
                .format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

    if 'asym' not in exclude:
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_asym.mlir {cmd_str} {debug_cmd} >log_{model}_bm1684x_tpu_int8_asym.mlir 2>&1'.\
                format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      po.apply_async(worker, (i, cmd_line,))
      i += 1

  po.close()
  po.join()
  print('all end')
