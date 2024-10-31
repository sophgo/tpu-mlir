**注意：所有提交到代码库的第三方依赖库都必须是release版本的；nntoolchain和libsophon需要与tpu-mlir同级目录**

## TPU1684 2024-08-23
sha256: 4d655f3187e5e918e7ecfe7411bad0cc04d0166a

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
debug: rebuild_bm1684_backend_cmodel
release: rebuild_bm1684_backend_release_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690/SG2380/MARS3 2024-8-30
``` bash

#bm1684x sha256: 906153e8252d30b141a2c09e23cf6b2d2f59d542
cd TPU1686
source  scripts/envsetup.sh bm1684x
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a

#bm1688 sha256: 906153e8252d30b141a2c09e23cf6b2d2f59d542
cd TPU1686
source  scripts/envsetup.sh bm1686
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a

#bm1690 sha256: 929bac5a6c1b3af424934c2b050075127d9e9ea4
cd TPU1686
source  scripts/envsetup.sh sg2260
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_bm1690.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a
cp build/firmware_core/libtpuv7_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/

#sg2380 sha256: 10b5a2378f9b67489f1ab9a56d697f5bb36466bf
cd TPU1686
source  scripts/envsetup.sh sg2380
debug: rebuild_backend_lib_cmodel
elease: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so

#mars3 sha256: e5b864414e3a4feb8a343f7c3417601addb6cc82
cd TPU1686
source  scripts/envsetup.sh mars3
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
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

## tpuv7-runtime 2024-08-21
build from tpuv7-runtime 1d1c1fbc4c876be4bcce8d7a79afafae447270e9
```bash
mkdir -p -p build/emulator
cd -p build/emulator
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DUSING_DEBUG=ON -DUSING_TP_DEBUG=ON ../..
make -j$(nproc)
cp model-runtime/runtime/libtpuv7_modelrt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/host/cdm_runtime/libtpuv7_rt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/ap/daemon/cdm_daemon/libcdm_daemon_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp model-runtime/runtime/include/tpuv7_modelrt.h /workspace/tpu-mlir/third_party/nntoolchain/include
cp cdmlib/host/cdm_runtime/include/tpuv7_rt.h /workspace/tpu-mlir/third_party/nntoolchain/include
```
