import joblib
from tpu_perf.infer import SGInfer
import numpy as np
from tqdm import tqdm
import pandas as pd
from time import time

image_ids = joblib.load('image_ids.joblib')
test_npy  = joblib.load('test_npy.joblib')


class Pipeline:
    def __init__(
            self,
    ):
        # models
        self.model = SGInfer("cosface_r34.bmodel", devices=[0] )
    def forward(self, line):
        input_ids = np.array(line, dtype=np.float32)

        results = self.model.infer_one(input_ids)
        return results

print('loading model...')
pipeline = Pipeline()

print('infering test data...')
begin_time = time()

result_list = []
for image in test_npy:
    tmp_res = pipeline.forward(image)
    result_list.append(tmp_res)
print('time cost: ',time() - begin_time)

preds_list = []
for res in tqdm(result_list):
    tmp_pred = np.argmax(res[0][0])
    preds_list.append(tmp_pred)

print(preds_list[:5],len(preds_list))

subtmit_df = pd.DataFrame()
subtmit_df['img'] = image_ids
subtmit_df['label'] = preds_list
subtmit_df.to_csv('submission.csv', index=False)