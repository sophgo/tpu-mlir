import os
import time
import argparse
from multiprocessing import Pool

model_list_all=[
  # "--arch=mobilenet_v2 --batch-size=64 --cali-batch-num=16",
  # "--arch=resnet50 --batch-size=64 --cali-batch-num=16",
  # "--arch=vgg11_bn --batch-size=64 --cali-batch-num=16",
  "--arch=resnet18 --batch-size=64 --cali-batch-num=16",
  # "--arch=shufflenet_v2_x0_5 --batch-size=64 --cali-batch-num=16",
  # "--arch=squeezenet1_1 --batch-size=64 --cali-batch-num=16",
  # "--arch=mobilenet_v3_small  --batch-size=64 --cali-batch-num=16"
]

# output_path='/path-of-your-dir/'
output_path='./ptq_test_before_push'

cmd_str = f"--data_path=/sea/data/imagenet/for_train_val --chip=BM1690 --quantmode=weight_activation --seed=1005 --pretrained --quantize_type=naive_ptq --deploy\
           --output_path={output_path}"

def worker(cmd_line):
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

if __name__ == "__main__":

  time_start = time.time()
  wp = os.getcwd()
  po = Pool(1)
  for i,model in enumerate(model_list_all):
    arch = model.split(' ')[0].split('=')[1].strip()
    os.system(f'mkdir -p {output_path}/{arch}')

    import torch
    if torch.cuda.is_available():
        cmd_line = f'cd {wp};CUDA_VISIBLE_DEVICES=0 python3 application/imagenet_example/PTQ/ptq/ptq_main.py  {model} {cmd_str} > {output_path}/{arch}/{i}_log_train_{arch} 2>&1'
    else:
        cmd_line = f'cd {wp};python3 application/imagenet_example/PTQ/ptq/ptq_main.py  {model} {cmd_str} --cpu > {output_path}/{arch}/{i}_log_train_{arch} 2>&1'
    # cmd_line = f'cd {wp};CUDA_VISIBLE_DEVICES=0 python3 application/imagenet_example/PTQ/ptq/ptq_main.py  {model} {cmd_str}'

    print('start', model)
    po.apply_async(worker, (cmd_line,))

  po.close()
  po.join()
  print('all end')

  time_end = time.time()
  print('totally time is ', time_end-time_start)
