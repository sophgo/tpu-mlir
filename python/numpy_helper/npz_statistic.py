# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def npz_statistic(args):
  npzfile = np.load(args[0])
  if len(args) == 1 or args[1] == "--list":
    print("\n".join(npzfile.files))
    exit(0)

  if args[1] in npzfile.files:
    d = npzfile[args[1]]
  else:
    raise ValueError("No {} in {} npz file".format(args[1], args[0]))

  shape = d.shape
  print('data shape:', shape)
  ax = 1 if len(args) < 3 else int(args[2])
  np.set_printoptions(precision=6)
  np.set_printoptions(suppress=True)
  new_shape = [ax]
  for i in range(len(shape)):
    if i != ax:
      new_shape.append(i)
  d = d.transpose(new_shape)
  print('new_shape:', new_shape)
  d = d.reshape([shape[ax],-1])

  amax = np.amax(d, axis=-1)
  amin = np.amin(d, axis=-1)
  print('mean', np.mean(d, axis=-1))
  print('var', np.var(d, axis=-1))
  print('max', amax)
  print('min', amin)
  r = amax - amin
  print('data range', r)
  norm = normalize(r)
  print('data range norm', norm)
  norm_odr = np.argsort(norm)
  print('norm_odr', norm_odr)
  for i in range(len(r)):
    print(f'idx:{i}, range:{r[i]}, range norm:{norm[i]}, norm order:{norm_odr[i]}')

  d2 = npzfile[args[1]]
  print('mean', np.mean(d2, axis=(1,2,3)))
  print('var', np.var(d2, axis=(1,2,3)))
