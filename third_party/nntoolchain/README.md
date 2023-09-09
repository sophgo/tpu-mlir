## TPU1684 2023-07-21
sha256: ebfe12c4683ff2345f94faf58e68743d8a8bc606

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-08-29
sha256: 212abba393e027cbbc8b707276d62fe49a0d69fc

## TPU1686(sg2260) 2023-09-04
sha256: 25f66fdeaff481614cbc479b7fcceba62846a828

``` bash
pushd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684x_backend_cmodel
rebuild_bm1686_backend_cmodel
cp out/install/lib/libcmodel_1684x.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684x.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp out/install/lib/libcmodel_1686.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1686.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
popd
pushd nntoolchain/TPU1686/
source scripts/envsetup.sh
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
popd
```

## tpu-runtime 2023-08-31
build from tpu-runtime 37226235356cd162c872b14bce25b45d3a1424d8
``` bash
pushd nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bmruntime
cp out/install_bmruntime/lib/libbmrt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp out/install/lib/libbmlib.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
popd
```
