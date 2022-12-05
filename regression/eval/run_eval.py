import os
import ast
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--exclude', type=str, default='onnx', help='exclude')
parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
parser.add_argument('--pool_size', type=int, default=10, help='pool size')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--qat_config_path', type=str, default='./qat_config', help='qat_config_path')
parser.add_argument('--qat_model_path', type=str,
                      default='/workspace/classify_models/,/workspace/yolov5/qat_models/',
                      help='list of diff qat_model_path,use douhao to seperate')
parser.add_argument('--qat_eval', action='store_true')
parser.add_argument('--fast_test', action='store_true')
opt = parser.parse_args()

postprocess_type_all=[
  "--count 50 --dataset_type imagenet --postprocess_type topx --dataset /data/ILSVRC2012_img_val_with_subdir/",
  "--count 0 --dataset_type coco --postprocess_type coco_mAP --dataset /data/coco/val2017 --coco_annotation /data/coco/instances_val2017.json"
]

model_list_all={
# object detection
  # "yolov5s_qat_ori":[1,1],
  "yolov5s_qat":[1,1],
# classification
  # "resnet18_qat_ori":[0,0],
  # "resnet18_qat":[0,0],
  # "resnet50_qat_ori":[0,0],
  # "resnet50_qat":[0,0],
  # "mobilenet_v2_qat_ori":[0,0],
  # "mobilenet_v2_qat":[0,0],
  # "squeezenet1_1_qat_ori":[0,0],
  # # "squeezenet1_1_qat":[0,2],
  # "vgg11_bn_qat_ori":[0,0],
  # "vgg11_bn_qat":[0,0],
}

def writeInfoToFile(log_file, info):
    fh = open(log_file, 'w', encoding='utf-8')
    fh.write(info+'\r\n')
    fh.flush()
    fh.close()

def str2bool(v):
  return v.lower() in ("yes", "true", "1")

task_i = 0
main_process_exit = False
def worker(n, cmd_line):
    if cmd_line == 'monitor':
      count = 0
      log_file_lit = []
      while True:
        if count % 120 == 0:
          line = open('/tmp/run_eval_log_file_list').readlines()[0]
          log_file_lit = eval(line)
        if count % 60 == 0:
          for file_path in log_file_lit:
            model_name = file_path.split('/')[0]
            dis_num = 5 if model_list_all[model_name][0] == 1 else 1
            lines = os.popen(f'tail -n {dis_num} {file_path}').readlines()
            if len(lines) > dis_num:
              for i in range(dis_num, -1, -1):
                print(file_path, 'result:', lines[len(lines)-1-i].strip('\r\n')[10:])
        if main_process_exit:
          break
        time.sleep(1)
        count += 1
      return
    print('idx:{}, cmd_line is runing:'.format(n), cmd_line)
    os.system(cmd_line)
    print('idx:{}, cmd_line end'.format(n))

if __name__ == "__main__":
  os.system('cat /dev/null > /tmp/run_eval_log_file_list')
  po = Pool(opt.pool_size)
  if opt.fast_test:
    for n,process_str in enumerate(postprocess_type_all):
      tmp = [i for i in process_str.split(' ') if len(i.strip()) > 0]
      tmp[tmp.index('--count')+1] = str(21)
      postprocess_type_all[n] = ' '.join(tmp)
  if opt.qat_eval and opt.qat_model_path == '':
    print('must set qat_model_path when qat_eval is enable')
    exit(0)
  qat_config_path = os.path.realpath(opt.qat_config_path)
  paths = [os.path.realpath(i.strip()) for i in opt.qat_model_path.split(',') if len(i.strip()) > 0]
  for model in model_list_all:
    if opt.qat_eval:
      qat_model_path = paths[0]
      if len(model_list_all[model]) > 1:
        qat_model_path = paths[model_list_all[model][1]]
      if model.endswith('_qat'):
        pytorch_arch_name = model[0:-4]
        cfg_file = os.path.join(qat_config_path, f'{model}.cfg')
        onnx_suffix = '_mqmoble_deploy_model.onnx'
      elif model.endswith('_qat_ori'):
        pytorch_arch_name = model[0:-8]
        cfg_file = os.path.join(qat_config_path, f'{model}.cfg')
        onnx_suffix = '_ori.onnx'
      else:
        continue
      lines = open(cfg_file).readlines()
      str1 = os.path.join(pytorch_arch_name, f'{pytorch_arch_name}{onnx_suffix}')
      str2 = os.path.join(qat_model_path, str1)
      model_path_str = f'model_path={str2}\n'
      model_path_exist = False
      if model.endswith('_qat'):
        str1 = os.path.join(pytorch_arch_name, f'{pytorch_arch_name}_mqmoble_cali_table_from_mqbench_sophgo_tpu')
        str2 = os.path.join(qat_model_path, str1)
        specified_cali_table_str = f'specified_cali_table={str2}\n'
        specified_cali_table_exist = False
      for i, line in enumerate(lines):
        if 'model_path' in line:
          lines[i] = model_path_str
          model_path_exist = True
        if 'input_shapes' in line:
          input_shapes = ast.literal_eval(line.strip().split('=')[1])
          for shape in input_shapes:
            shape[0] = opt.batch_size
          input_shapes = str(input_shapes)
          input_shapes = input_shapes.replace(' ','')
          lines[i] = f'input_shapes={input_shapes}\n'
          print(lines[i], 'wxc11')
        if model.endswith('_qat') and 'specified_cali_table' in line:
          lines[i] = specified_cali_table_str
          specified_cali_table_exist = True
      if not model_path_exist:
        lines.append(model_path_str)
      if model.endswith('_qat') and not specified_cali_table_exist:
        lines.append(specified_cali_table_str)
      fh = open(cfg_file, 'w', encoding='utf-8')
      for line in lines:
        fh.write(line)
      fh.flush()
      cmd_line = 'rm -rf {};bash $REGRESSION_PATH/eval/gen_model.sh {} {}'.format(model, model, cfg_file)
    else:
      cmd_line = 'rm -rf {};bash $REGRESSION_PATH/eval/gen_model.sh {}'.format(model, model)
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

  debug_cmd = ''
  if opt.debug_cmd != '':
    debug_cmd = '--debug_cmd {}'.format(opt.debug_cmd)

  po.apply_async(worker, (task_i, 'monitor'))
  log_file_lit=[]
  task_i += 1
  exclude = [ex.strip() for ex in opt.exclude.split(',')]
  for model in model_list_all:
    cmd_str = postprocess_type_all[model_list_all[model][0]]
    if 'onnx' not in exclude:
      if opt.qat_eval:
        if model.endswith('_qat'):
          pytorch_arch_name = model[0:-4]
          cfg_file = os.path.join(qat_config_path, f'{model}.cfg')
        elif model.endswith('_qat_ori'):
          pytorch_arch_name = model[0:-8]
          cfg_file = os.path.join(qat_config_path, f'{model}.cfg')
        else:
          continue
      else:
        cfg_file = '{}/config/{}.cfg'.format(os.getenv('REGRESSION_PATH'), model)
      preprocess_str = ''
      lines = open(cfg_file).readlines()
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

      log_file_lit.append(f'{model}/log_{model}_opt.onnx')
      writeInfoToFile('/tmp/run_eval_log_file_list', str(log_file_lit))
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_opt.onnx {cmd_str} {preprocess} {debug_cmd} >>log_{model}_opt.onnx 2>&1'\
                .format(model=model, cmd_str=cmd_str, preprocess = preprocess_str, debug_cmd=debug_cmd)
      writeInfoToFile(log_file_lit[-1], cmd_line)
      po.apply_async(worker, (task_i, cmd_line,))
      task_i += 1

    if 'top' not in exclude:
      log_file_lit.append(f'{model}/log_{model}.mlir')
      writeInfoToFile('/tmp/run_eval_log_file_list', str(log_file_lit))
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}.mlir {cmd_str} {debug_cmd} >>log_{model}.mlir 2>&1'\
                .format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      writeInfoToFile(log_file_lit[-1], cmd_line)
      po.apply_async(worker, (task_i, cmd_line,))
      task_i += 1

    if 'sym' not in exclude:
      log_file_lit.append(f'{model}/log_{model}_bm1684x_tpu_int8_sym.mlir')
      writeInfoToFile('/tmp/run_eval_log_file_list', str(log_file_lit))
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_sym.mlir {cmd_str} {debug_cmd} >>log_{model}_bm1684x_tpu_int8_sym.mlir 2>&1'\
                .format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      writeInfoToFile(log_file_lit[-1], cmd_line)
      po.apply_async(worker, (task_i, cmd_line,))
      task_i += 1

    if 'asym' not in exclude:
      log_file_lit.append(f'{model}/log_{model}_bm1684x_tpu_int8_asym.mlir')
      writeInfoToFile('/tmp/run_eval_log_file_list', str(log_file_lit))
      cmd_line = 'cd $REGRESSION_PATH/eval/{model}&&$REGRESSION_PATH/../python/tools/model_eval.py --model_file {model}_bm1684x_tpu_int8_asym.mlir {cmd_str} {debug_cmd} >>log_{model}_bm1684x_tpu_int8_asym.mlir 2>&1'.\
                format(model=model, cmd_str=cmd_str, debug_cmd=debug_cmd)
      writeInfoToFile(log_file_lit[-1], cmd_line)
      po.apply_async(worker, (task_i, cmd_line,))
      task_i += 1

  po.close()
  po.join()
  print('all end')
  main_process_exit = True
