## TPU1684 2023-06-16
sha256: fd4142dc6b874f6a72db72008fbefd32cec04599

``` bash
cd  nntoolchain/net_compiler/
source  scripts/envsetup.sh
rebuild_bm1684_backend_cmodel
cp out/install/lib/libcmodel_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
cp bmcompiler/libbackend/libbackend_1684.so /workspace/tpu-mlir/third_party/nntoolchain/lib/
```

## TPU1686 2023-07-20
sha256: 2f65d7ecacc19299a346c062044cd1b414087f55

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

2023-07-20
build from tpu-runtime 8dfe19265a34602bfbdb38303a98d93fcabe92a6
2023-03-03
build from nntoolchain 8308714678f92122c2b3bb1989cfa978feb13c71
2023-07-20
build from libsophon   4c14fcc26f7f0df454205d832a7d7fbf21196811
2023-04-26
build from tpu-cpuop   a158817d0260990d7a7aa6486cfacfab552b7ced
