Team AP0200008
=====================

Sophgo TPU programming competition Top1 code sample.

Note: This sample is for future entrants’ reference only. Origin model, test image and calibration should be prepared by yourselves.

run.sh
```bash
cd /workspace/tpu-mlir

cd /workspace/tpu-mlir/python/samples/unet

bash run.sh
```

Use search.py instead if the above command doesn't work.

```bash
cd /workspace/tpu-mlir

source envsetup.sh

cd /workspace/tpu-mlir/python/samples/unet

python search.py # Full workflow, the same as run_model script
```
