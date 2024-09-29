import pickle as pkl
from tpu_perf.infer import SGInfer
import numpy as np
from tqdm import tqdm
import pandas as pd
from time import time


class Pipeline:
    def __init__(self, ):
        self.model = SGInfer("./cosface_r34.bmodel", devices=[0])

    def forward(self, imgs):
        ipt_img = np.array(imgs, dtype=np.float32)
        opt = self.model.infer_one(ipt_img)
        return opt


print(1111)
lst_img = np.load('./data_img_ipt.npy')

print(2222)
time_sat = time()

lst_img_opt = []
pipe_line = Pipeline()
for img in lst_img:
    img_opt = pipe_line.forward(img)
    lst_img_opt.append(img_opt)

print(3333)
print(time() - time_sat)

lst_img_opt_pred = []
for img_opt in lst_img_opt:
    img_opt_pred = np.argmax(img_opt[0][0])
    lst_img_opt_pred.append(img_opt_pred)

print(4444)
print(len(lst_img_opt_pred))

sub = pd.DataFrame()
sub['pred'] = lst_img_opt_pred
sub.to_csv('sub.csv', index=False)