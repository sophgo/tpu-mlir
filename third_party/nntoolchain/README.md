## TPU1684 2023-06-16
sha256: 9540a9b1bff45c9ffb11507c6b2388f95f21994a

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-06-24
sha256: 76e2acf711c012fb523eccde0df9a80eebe98fed

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

2023-06-121
build from tpu-runtime a609d2b722dbd6aa924bee0e39ed4eaea47fa548
2023-03-03
build from nntoolchain 8308714678f92122c2b3bb1989cfa978feb13c71
2023-05-16
build from libsophon   721d98b749901481d5f1fbc4594d6d588ac165ea
2023-04-26
build from tpu-cpuop   a158817d0260990d7a7aa6486cfacfab552b7ced
