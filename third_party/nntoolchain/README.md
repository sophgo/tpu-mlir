## TPU1684 2023-07-21
sha256: ebfe12c4683ff2345f94faf58e68743d8a8bc606

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-10-24
sha256: 3e8e483db93e36fbb9ac476f9f42290d5435210f

``` bash
#bm1684x
cd TPU1686
source  scripts/envsetup.sh bm1684x
rebuild_backend_lib_cmodel
#sg2260
source  scripts/envsetup.sh sg2260
rebuild_backend_lib_cmodel
# cp out/install/lib/libcmodel_1684x.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
# cp bmcompiler/libbackend/libbackend_1684x.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
# cp out/install/lib/libcmodel_1686.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
cp build/backend_api/libbackend_sg2260.so ../../tpu-mlir/third_party/nntoolchain/lib/
cp build/backend_api/libbackend_sg2260.so ../../tpu-mlir/install/lib/
cp build_runtime/firmware_core/libcmodel_firmware.so ../../tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2260.so
cp build_runtime/firmware_core/libcmodel_firmware.so ../../tpu-mlir/install/lib/libcmodel_sg2260.so
popd
# pushd nntoolchain/TPU1686/
# source scripts/envsetup.sh
# rebuild_firmware
# cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
# popd
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


## tpu-cpuop 2023-10-10

```bash
pushd /workspace/nntoolchain/net_compiler
source new_scripts/envsetup.sh
rebuild_cpuop
cp /workspace/nntoolchain/net_compiler/out/lib/libcpuop.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/

# libbmcpu.so/libusercpu.so are deprecated
```
