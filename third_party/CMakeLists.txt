add_subdirectory(cnpy)

install(FILES oneDNN/lib/libdnnl.so
              oneDNN/lib/libdnnl.so.2
              oneDNN/lib/libdnnl.so.2.6
              nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_1684x.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libbmcpu.so
              nntoolchain/lib/libusercpu.so
        DESTINATION lib)
install(DIRECTORY llvm/python_packages/mlir_core/mlir DESTINATION python)
