add_subdirectory(cnpy)

install(FILES nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbackend_1688.so
              nntoolchain/lib/libbackend_mars3.so
              nntoolchain/lib/libbackend_bm1690.so
              nntoolchain/lib/libbackend_sg2380.so
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libbmlib.so.0
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_1684x.so
              nntoolchain/lib/libcmodel_1688.so
              nntoolchain/lib/libcmodel_mars3.so
              nntoolchain/lib/libcmodel_bm1690.so
              nntoolchain/lib/libcmodel_sg2380.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libbmrt.so.1.0
              nntoolchain/lib/libcpuop.so
              nntoolchain/lib/libcpuop.so.1.0.0
              nntoolchain/lib/libbm1684x_kernel_module.so
              nntoolchain/lib/libbm1684x_kernel_module.a
              nntoolchain/lib/libbmtpulv60_kernel_module.so
              nntoolchain/lib/libbmtpulv60_kernel_module.a
              nntoolchain/lib/liblibtorch_plugin.so
              or-tools/lib/libortools.so
              or-tools/lib/libortools.so.9
              or-tools/lib/libortools.so.9.8.3296
              tpu-train/lib/libsgdnn.so

              # cvitek
              CV18xx/lib/libcvikernel.so
              CV18xx/lib/libcviruntime.so
              CV18xx/lib/libcvicmodel.so
              CV18xx/lib/libunwind.so.8
              atomic_exec/libbm1684x_atomic_kernel.so
        DESTINATION lib)
