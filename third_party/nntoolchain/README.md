## TPU1684 2023-05-09
sha256: 0bddf4692746ee7cead54cd3b4ceb807ed24df8c

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-05-19
sha256: 6df9af24553a5dcd60e58b1349c13d3392d178bc

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

2023-04-12
build from tpu-runtime 9e9d9a7d983305cb15c36b5774f9dd3259d6b970
2023-03-03
build from nntoolchain e0309c47ba4957dec9e9a7c108946f7eee943128
2023-05-16
build from libsophon   721d98b749901481d5f1fbc4594d6d588ac165ea
2023-04-26
build from tpu-cpuop   a158817d0260990d7a7aa6486cfacfab552b7ced
