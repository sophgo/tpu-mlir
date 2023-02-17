# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def npz_statistic(args):
  os.system('pip3 install visdom')
  from visdom import Visdom
  viz = Visdom()
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

  viz.boxplot(X=d)
  # amax = np.amax(d, axis=-1)
  # amin = np.amin(d, axis=-1)
  # print('mean', np.mean(d, axis=-1))
  # print('var', np.var(d, axis=-1))
  # print('max', amax)
  # print('min', amin)
  # r = amax - amin
  # print('data range', r)
  # norm = normalize(r)
  # norm = [(i,data)for i,data in enumerate(norm.tolist())]
  # norm = sorted(norm, key=lambda x: x[1], reverse=False)
  # print('norm', norm)
