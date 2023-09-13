add_subdirectory(cnpy)

install(FILES nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbackend_1686.so
              nntoolchain/lib/libbackend_mars3.so
              nntoolchain/lib/libbackend_sg2260.so
              nntoolchain/lib/libbmlib.so.0
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_1684x.so
              nntoolchain/lib/libcmodel_1686.so
              nntoolchain/lib/libcmodel_mars3.so
              nntoolchain/lib/libcmodel_sg2260.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libcpuop.so
              nntoolchain/lib/libbm1684x_kernel_module.so
              # cvitek
              CV18xx/lib/libcvikernel.so
              CV18xx/lib/libcviruntime.so
              CV18xx/lib/libcvicmodel.so
              CV18xx/lib/libunwind.so.8
        DESTINATION lib)
