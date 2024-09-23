**注意：所有提交到代码库的第三方依赖库都必须是release版本的；nntoolchain和libsophon需要与tpu-mlir同级目录**

## TPU1684 2024-08-23
sha256: 4d655f3187e5e918e7ecfe7411bad0cc04d0166a

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
**debug: rebuild_bm1684_backend_cmodel**
**release: rebuild_bm1684_backend_release_cmodel**
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690/SG2380/MARS3 2024-8-30
``` bash

#bm1684x sha256: a72fbed51c6506c9e7f010c2ae6d0a7710903d6c
cd TPU1686
source  scripts/envsetup.sh bm1684x
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a

#bm1688 sha256: 253d8e58e7b2764d625bfcdd9d049d0184a0f602
# reverts for segnet perf
# -ca9ae233dbde82307c816b88c8563f6fa1e5197a
# -44a3e41e8109aebb43aeed6c86ec4fcc49c3b4a5
# -33f2a9a014f098804acf72578327077b49936539
# -01bf54a7745897a76abcfb906d478d8707b6abcc
# -dd1bb51da1d74ce1316b26d4c67a2bc000fc9aa7git
# -7bf05363836f6fadc057f661b38d8355cf1352ee
# -7c1b7e990cfc56ea80187b3b430408945996de66
# +5fd5cfea836a56e91faf59ae669b1865959019a4
cd TPU1686
source  scripts/envsetup.sh bm1686
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a

#bm1690 sha256: 253d8e58e7b2764d625bfcdd9d049d0184a0f602
cd TPU1686
source  scripts/envsetup.sh sg2260
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_bm1690.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a

#sg2380 sha256: 18586a9b5167f6f3959978032d6acc4371e3c116
cd TPU1686
source  scripts/envsetup.sh sg2380
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so

#mars3 sha256: 988ba086a60c4e9721ea3760281d09cef5a543e2
cd TPU1686
source  scripts/envsetup.sh mars3
**debug: rebuild_backend_lib_cmodel**
**release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel**
cp build/backend_api/libbackend_mars3.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_mars3.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_mars3.so
```


## tpu-runtime 2023-09-12
build from tpu-runtime a896999bb85550c769190dec203d32ecf5516ac2
``` bash
pushd libsophon
mkdir -p build && cd build
cmake -G Ninja -DPLATFORM=cmodel -DCMAKE_BUILD_TYPE=DEBUG ../ # release version has problem
ninja
cp -P tpu-runtime/libbmrt.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp -P bmlib/libbmlib.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp -P tpu-bmodel/libmodel_combine.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
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
