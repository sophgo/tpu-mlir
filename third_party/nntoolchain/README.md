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

## TPU1684X/1688/BM1690/SG2380/MARS3/SGTPUV8 2025-07-02

```bash

#bm1684x sha256: 915790b32a3008c60b518bdd05e53d02f452880c
cd TPU1686
source  scripts/envsetup.sh bm1684x
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1684x.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1684x.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1684x.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1684x_kernel_module.a

#bm1688 sha256: 915790b32a3008c60b518bdd05e53d02f452880c
cd TPU1686
source  scripts/envsetup.sh bm1686
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_bm1686.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_1688.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_1688.so
rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmtpulv60_kernel_module.a

#bm1690 sha256: c4d407397c7163ab5ae3a9e89e3fe2004867c013
cd TPU1686
source  scripts/envsetup.sh sg2260
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2260.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_bm1690.so
export EXTRA_CONFIG="-DDEBUG=OFF -DUSING_FW_DEBUG=OFF" && rebuild_test sgdnn
cp build/firmware_core/libtpuv7_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
unset EXTRA_CONFIG && rebuild_firmware
cp build/firmware_core/libfirmware_core.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.so
cp build/firmware_core/libfirmware_core.a /workspace/tpu-mlir/third_party/nntoolchain/lib/libbm1690_kernel_module.a

#sg2380 sha256: c2cdcdc5617320b9c5bbf3da05339362dde2cae3
cd TPU1686
source  scripts/envsetup.sh sg2380
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_sg2380.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_sg2380.so
cp build_runtime/firmware_core/libcmodel_firmware.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_sg2380.so

#mars3 sha256: 800f08bd7df50ddbc4dcf7baef08fbd70efe67b0
cd TPU1686
source  scripts/envsetup.sh mars3
debug: rebuild_backend_lib_cmodel
release: unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
cp build/backend_api/libbackend_mars3.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libbackend_mars3.so
cp build_runtime/firmware_core/libcmodel_firmware.so  /workspace/tpu-mlir/third_party/nntoolchain/lib/libcmodel_mars3.so

#SGTPUV8 sha256: 682107b2e2513e51f2fceccd6d9db5cd4610fec9
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

## tpu-runtime 2025-1-17

build from tpu-runtime 2a1ecf74d667b26b86d094ef073b286141cdcfb1

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

## tpuv7-runtime 2024-12-20

build from tpuv7-runtime e06660ead76443d9dc6cabd59d9d003a1d5ccfd6

```bash
mkdir -p -p build/emulator
cd build/emulator
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DUSING_DEBUG=OFF -DUSING_TP_DEBUG=OFF ../..
make -j$(nproc)
cp model-runtime/runtime/libtpuv7_modelrt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/host/cdm_runtime/libtpuv7_rt.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp cdmlib/ap/daemon/cdm_daemon/libcdm_daemon_emulator.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp model-runtime/runtime/include/tpuv7_modelrt.h /workspace/tpu-mlir/third_party/nntoolchain/include
cp cdmlib/host/cdm_runtime/include/tpuv7_rt.h /workspace/tpu-mlir/third_party/nntoolchain/include
```
