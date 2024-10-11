# Step 1: Install ppl release

``` shell
cp ppl_v1.0.3-gbdca73a-20240522.tar.gz tpu-mlir/third-party/
cd tpu-mlir/third-party
tar -xvhf ppl_v1.0.3-gbdca73a-20240522.tar.gz
cd ppl_v1.0.3-gbdca73a-20240522
source envsetup.sh
```
# Step 2: Compile ppl backend so

``` shell
cd tpu-mlir/ppl_backend
sh build.sh
```

This step will install libppl_host.so to tpu-mlir/install/lib

# Step3: Modify Fattention's codegen
open tpu-mlir/lib/Dialect/Tpu/Interfaces/BM1684X/FAttention.cpp
``` cpp
#if 0
  BM168x::call_global_func("backend_api_flash_attention_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
#else
  BM168x::call_ppl_func("api_fattention_global", &param, sizeof(param),
                        input_spec->data(), output_spec->data());
#endif
```

modify "#if 1" to "#if 0", and recompile tpu-mlir

# Step4: Test FAttention op
open tpu-mlirpython/test/test_torch.py, find "test_FAttention" function
``` python
        # _test_flash_attention(1, 128, 28, 4, 1, 8193,   True)
        # _test_flash_attention(1, 128, 28, 4, 256, 513,  True)
        # _test_flash_attention(1, 128, 32, 32, 1, 8193,  True)
        # _test_flash_attention(1, 128, 32, 32, 256, 513, True)
        # _test_flash_attention(1, 128, 64, 8, 1, 8193,   True)
        # _test_flash_attention(1, 128, 64, 8, 256, 513,  True)
        # _test_flash_attention(1, 128, 10, 1024, 1024, False)
        # _test_flash_attention(1, 128, 10, 2048, 2048, True)
        # _test_flash_attention(1, 128, 10, 2048, 2048, False)
        # _test_flash_attention(2, 128, 10, 1024, 1024, True)
        # _test_flash_attention(1, 128, 4, 4, 128, 128, True)
        # _test_flash_attention(1, 128, 4, 1, 128, True)
        # _test_flash_attention(1, 128, 10, 1, 2048, True)
        # _test_flash_attention(1, 128, 10, 1, 4096, True)
        # _test_flash_attention(1, 128, 10, 1, 8192, True)
```
Uncomment
``` python
        # _test_flash_attention(1, 128, 28, 4, 1, 8193,   True)
        # _test_flash_attention(1, 128, 28, 4, 256, 513,  True)
        # _test_flash_attention(1, 128, 32, 32, 1, 8193,  True)
        # _test_flash_attention(1, 128, 32, 32, 256, 513, True)
        # _test_flash_attention(1, 128, 64, 8, 1, 8193,   True)
        # _test_flash_attention(1, 128, 64, 8, 256, 513,  True)
        # _test_flash_attention(1, 128, 10, 1024, 1024, False)
        # _test_flash_attention(1, 128, 10, 2048, 2048, True)
        # _test_flash_attention(1, 128, 10, 2048, 2048, False)
        # _test_flash_attention(2, 128, 10, 1024, 1024, True)
        _test_flash_attention(1, 128, 4, 4, 128, 128, True)
        # _test_flash_attention(1, 128, 4, 1, 128, True)
        # _test_flash_attention(1, 128, 10, 1, 2048, True)
        # _test_flash_attention(1, 128, 10, 1, 4096, True)
        # _test_flash_attention(1, 128, 10, 1, 8192, True)
```

run

``` shell
test_torch.py --case FAttention --chip bm1684x
```



