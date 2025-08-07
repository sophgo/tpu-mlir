Here, both libcmodel_xxx.so and libxxx_kernel_module.so embed the code from lib/Pplbackend on top of the nntc static library. The libcmodel_xxx.so library serves as the inference engine for the C model, and libxxx_kernel_module.so is injected into the bmodel to act as the device module.

nntoolchain related：
libcmodel_xxx.so is built with nntoolchain/lib/libcmodel_xxx.a
libxxx_kernel_module.so is built with nntoolchain/lib/libxxx_kernel_module

tpu-mlir code gen related：
libppl_host.so is for tpu-mlir staic mode
libppl_dyn_host_xxx.so is for tpu-mlir dynamic mode

build script: /workspace/tpu-mlir/lib/PplBackend/build.sh [RELEASE|DEBUG] [force|conditional]
    param：
        RELEASE：（default）Build type for all above mentiond libs
        DEBUG：（option）Build type for all above mentiond libs
        force：（default）Compile regardless of whether the library has been updated or not
        conditional（option）：Compile when the nntc or ppl libraries are updated

***NOTICE***
make sure build release type by using:
/workspace/tpu-mlir/lib/PplBackend/build.sh