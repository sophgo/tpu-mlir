**注意：所有提交到代码库的第三方依赖库都必须是release版本的；nntoolchain和libsophon需要与tpu-mlir同级目录**

## TPU1684 2024-11-20
sha256: I36917a067fc3902743c8e95764f6ece32e206476

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
debug: rebuild_bm1684_backend_cmodel
release: rebuild_bm1684_backend_release_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690/SG2380/MARS3 2024-11-6
``` bash

#bm1684x sha256:  b9653fd714815e33546701730698c4f14114b5dd
# - yolov5s perf revive: 5fd5cfea836a56e91faf59ae669b1865959019a4
cd TPU1686
source  scripts/envsetup.sh bm1684x
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a

#bm1688 sha256: 8453b87d6b1ad8858692f8ca4cd2441e7b677f1e
# - MDSR_x3/x4 perf revive: 253d8e58e7b2764d625bfcdd9d049d0184a0f602
cd TPU1686
source  scripts/envsetup.sh bm1686
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a

#bm1690 sha256: 90428e6d47c53c73fab8f846a1f579af456862aa
# + MaxPoolingWithMask optimize:
cd TPU1686
source  scripts/envsetup.sh sg2260
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
export EXTRA_CONFIG="-DDEBUG=ON -DUSING_FW_DEBUG=ON" && rebuild_test sgdnn
cp build/firmware_core/libtpuv7_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
unset EXTRA_CONFIG && rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a

#sg2380 sha256: 8453b87d6b1ad8858692f8ca4cd2441e7b677f1e
cd TPU1686
source  scripts/envsetup.sh sg2380
debug: rebuild_backend_lib_cmodel
elease: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so

#mars3 sha256:  68a4781b8f0ab6bb38eab38e4d1fd9c6f8a426b4
cd TPU1686
source  scripts/envsetup.sh mars3
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_mars3.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_mars3.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_mars3.so
```


## tpu-runtime 2023-10-31
build from tpu-runtime e42778db10b903b87f8f83637fdfc4b46e18ebd4
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

## tpuv7-runtime 2024-11-08
build from tpuv7-runtime 007069c8ec10823a60fedb7a1be8dafd49aae15e
```bash
mkdir -p -p build/emulator
cd -p build/emulator
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DUSING_DEBUG=OFF -DUSING_TP_DEBUG=OFF ../..
make -j$(nproc)
cp model-runtime/runtime/libtpuv7_modelrt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/host/cdm_runtime/libtpuv7_rt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/ap/daemon/cdm_daemon/libcdm_daemon_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp model-runtime/runtime/include/tpuv7_modelrt.h /workspace/tpu-mlir/third_party/nntoolchain/include
cp cdmlib/host/cdm_runtime/include/tpuv7_rt.h /workspace/tpu-mlir/third_party/nntoolchain/include
```
