Team AP0200008
=====================

run.sh
```bash
cd /workspace/tpu-mlir
source envsetup.sh

cd /workspace/competition/unet/submit
bash run.sh
```

如果上面的命令有问题，可以换用search.py

```bash
cd /workspace/tpu-mlir
source envsetup.sh

cd /workspace/competition/unet/submit
python search.py # 全流程，与run.sh逻辑一致
```