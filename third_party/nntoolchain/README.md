**注意：所有提交到代码库的第三方依赖库都必须是release版本的；nntoolchain和libsophon需要与tpu-mlir同级目录**

## TPU1684 2024-05-24
sha256: f1f90abab13d217408fcf555bc6f820cad3b31dd

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
**debug: rebuild_bm1684_backend_cmodel**
**release: rebuild_bm1684_backend_release_cmodel**
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690 2024-6-26
``` bash

#bm1684x sha256: 8a9624e43098e2ed72c51eb353addda72af231ec
cd TPU1686
source  scripts/envsetup.sh bm1684x
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a

#bm1688 sha256: 8a9624e43098e2ed72c51eb353addda72af231ec
cd TPU1686
source  scripts/envsetup.sh bm1686
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a

#bm1690 sha256: 8a9624e43098e2ed72c51eb353addda72af231ec
# bm1690 backend has problem, tmp revert:
# revert: 4c302a3b5b533b859c4a3f635824ebfd523ac863, 1a432de967b2b52db341924d895525a93d83c35d
cd TPU1686
source  scripts/envsetup.sh sg2260
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_bm1690.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a

#sg2380 sha256: 26946e72212d5984deede74e7e6a60669400d4d4
cd TPU1686
source  scripts/envsetup.sh sg2380
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so
```


## tpu-runtime 2023-5-9
build from tpu-runtime 85e9e78b9d47f497f819374de110e1d9e362db01
``` bash
pushd libsophon
mkdir -p build && cd build
cmake -G Ninja -DPLATFORM=cmodel -DCMAKE_BUILD_TYPE=DEBUG ../ # release version has problem
ninja
cp -P tpu-runtime/libbmrt.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp -P bmlib/libbmlib.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
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
