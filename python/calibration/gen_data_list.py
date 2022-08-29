#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import sys, os
from pathlib import Path
import random

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', help='dataset dir')
  parser.add_argument('num', help='data number')
  parser.add_argument('output', help='data list file')
  args = parser.parse_args()

  full_list = []

  for file_path in Path(args.dataset).glob('**/*'):
    if file_path.is_file():
      full_list.append(str(file_path))
  random.shuffle(full_list)
  #print(full_list)
  #print(len(full_list))
  num = int(args.num) if int(args.num) > 0 else len(full_list)
  print(num)
  with open(args.output, 'w') as fp:
    for i in range(num):
      print(full_list[i])
      fp.write(full_list[i])
      fp.write("\n")
