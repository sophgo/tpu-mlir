**注意：所有提交到代码库的第三方依赖库都必须是 release 版本的；nntoolchain 和 libsophon 需要与 tpu-mlir 同级目录**

## TPU1684 2024-11-20

sha256: 6ca642e822618af4cffcf531d2cc9e81edc8e03e

```bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
debug: rebuild_bm1684_backend_cmodel
release: rebuild_bm1684_backend_release_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690/SG2380/CV184X/SGTPUV8 2025-12-24
```bash
#bm1684x sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh bm1684x
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.a  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.a
rebuild_firmware
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a
/workspace/tpu-mlir/lib/PplBackend/build.sh

#bm1688 sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh bm1686
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.a  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.a
rebuild_firmware
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a
/workspace/tpu-mlir/lib/PplBackend/build.sh

#bm1690 sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh sg2260
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
export EXTRA_CONFIG="-DDEBUG=OFF -DUSING_FW_DEBUG=OFF" && rebuild_test sgdnn
cp build/firmware_core/libcmodel_firmware.a  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_bm1690.a
unset EXTRA_CONFIG && rebuild_firmware
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a

#bm1690e sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh sg2260e
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2260e.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690e.so
export EXTRA_CONFIG="-DDEBUG=OFF -DUSING_FW_DEBUG=OFF" && rebuild_test sgdnn
cp build/firmware_core/libcmodel_firmware.a  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_bm1690e.a
unset EXTRA_CONFIG && rebuild_firmware
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690e_kernel_module.a

#cv184x sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh mars3
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_mars3.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_cv184x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_cv184x.so

#sg2380 sha256: 4498fe9ae93429e3c7b79df24db596dc2e42aef4
cd TPU1686
source  scripts/envsetup.sh sg2380
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so

#SGTPUV8 sha256: 22988e8d42a591fadb69da28959884ede5984e5f
cd TPU1686
source  scripts/envsetup.sh sgtpuv8
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sgtpuv8.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sgtpuv8.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sgtpuv8.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libsgtpuv8_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libsgtpuv8_kernel_module.a
```

## tpu-runtime 2025-12-9

build from tpu-runtime 5a67ee3c1ba0beee128812f7a55b70120882b14d

```bash
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

## tpuv7-runtime 2025-11-18

build from tpuv7-runtime ff61f7ed6bc0d15ea77f0c0f746acabd6cec8255

```bash
mkdir -p build/emulator
pushd build/emulator
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DUSING_DEBUG=OFF -DUSING_TP_DEBUG=OFF ../..
make -j$(nproc)
cp model-runtime/runtime/libtpuv7_modelrt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/host/cdm_runtime/libtpuv7_rt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/fw/ap/daemon/libcdm_daemon_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/fw/tp/daemon/libtpuv7_scalar_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
popd
cp model-runtime/runtime/include/tpuv7_modelrt.h /workspace/tpu-mlir/third_party/nntoolchain/include
cp cdmlib/host/cdm_runtime/include/tpuv7_rt.h /workspace/tpu-mlir/third_party/nntoolchain/include
```

