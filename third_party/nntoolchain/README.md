**注意：所有提交到代码库的第三方依赖库都必须是 release 版本的；nntoolchain 和 libsophon 需要与 tpu-mlir 同级目录**

## TPU1684 2024-11-20

sha256: 6ca642e822618af4cffcf531d2cc9e81edc8e03e

```bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
debug: rebuild_bm1684_backend_cmodel
release: rebuild_bm1684_backend_release_cmodel
cp out/install/lib/libcmodel_bm1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_bm1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1684X/1688/BM1690/SG2380/CV184X/SGTPUV8

``` bash
# update all
./build_tpu1686.sh -t /workspace/TPU1686
# update one, (bm1684x2 for example)
./build_tpu1686.sh -t /workspace/TPU1686 -c bm1684x2
# if debug, add -d
```

## tpu-runtime 2026-05-22

build from tpu-runtime 43a13605f667a2ceae45c8a3dda1ab7dc2192556

```bash
pushd libsophon
mkdir -p build && cd build
cmake -G Ninja -DPLATFORM=cmodel -DCMAKE_BUILD_TYPE=Debug ../ # release version has problem
ninja
cp -P tpu-runtime/libbmrt.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp -P bmlib/libbmlib.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp -P tpu-bmodel/libmodel_combine.so* /workspace/tpu-mlir/third_party/nntoolchain/lib/
# libsopn need branch BM1684X2
cp -P bmlib/libbmlib.so /workspace/tpu-mlir/third_party/nntoolchain/lib/libbmlib_bm1684x2.so.0
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

## tpuv7-runtime

build from tpuv7-runtime ff61f7ed6bc0d15ea77f0c0f746acabd6cec8255

```bash
# RELEASE build (default)
./build_tpuv7.sh -t /workspace/tpuv7-runtime
# DEBUG build
./build_tpuv7.sh -d -t /workspace/tpuv7-runtime
```

