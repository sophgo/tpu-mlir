## TPU1684 2023-07-21
sha256: ebfe12c4683ff2345f94faf58e68743d8a8bc606

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-09-08
sha256: 31ba5c7bf6c25348dd1c48b288817b03649ff0d5

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

## tpu-runtime 2023-09-18
build from tpu-runtime 969fb471519b9c2cf34398d914e69f27786e8f52
``` bash
pushd libsophon
mkdir -p build && cd build
cmake -G Ninja -DPLATFORM=cmode ..
ninja
cp tpu-runtime/libbmrt.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmlib/libbmlib.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
popd
```
