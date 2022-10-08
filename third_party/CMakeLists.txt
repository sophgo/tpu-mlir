add_subdirectory(cnpy)

install(FILES oneDNN/lib/libdnnl.so
              oneDNN/lib/libdnnl.so.2
              oneDNN/lib/libdnnl.so.2.6
              caffe/lib/libboost_filesystem.so.1.65.1
              caffe/lib/libboost_python3-py36.so.1.65.1
              caffe/lib/libboost_regex.so.1.65.1
              caffe/lib/libboost_system.so.1.65.1
              caffe/lib/libboost_thread.so.1.65.1
              caffe/lib/libgflags.so.2.2
              caffe/lib/libglog.so.0
              caffe/lib/libprotobuf.so.10
              nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_1684x.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libbmcpu.so
              nntoolchain/lib/libusercpu.so
              # cvitek
              CV18xx/lib/libcvikernel.so
              CV18xx/lib/libcviruntime.so
              CV18xx/lib/libcvicmodel.so
              CV18xx/lib/libunwind.so.8

        DESTINATION lib)
install(DIRECTORY llvm/python_packages/mlir_core/mlir DESTINATION python)
install(DIRECTORY caffe/python/caffe DESTINATION python)
