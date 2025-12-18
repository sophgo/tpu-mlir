add_subdirectory(cnpy)

install(FILES nntoolchain/lib/libbackend_1684.so
              nntoolchain/lib/libbackend_1684x.so
              nntoolchain/lib/libbackend_1688.so
              nntoolchain/lib/libbackend_cv184x.so
              nntoolchain/lib/libbackend_bm1690.so
              nntoolchain/lib/libbackend_bm1690e.so
              nntoolchain/lib/libbackend_sg2380.so
              nntoolchain/lib/libbackend_sgtpuv8.so
              nntoolchain/lib/libbmlib.so
              nntoolchain/lib/libbmlib.so.0
              nntoolchain/lib/libcmodel_1684.so
              nntoolchain/lib/libcmodel_cv184x.so
              nntoolchain/lib/libcmodel_sg2380.so
              nntoolchain/lib/libcmodel_sgtpuv8.so
              nntoolchain/lib/libcmodel.so
              nntoolchain/lib/libbmrt.so
              nntoolchain/lib/libbmrt.so.1.0
              nntoolchain/lib/libcpuop.so
              nntoolchain/lib/libcpuop.so.1.0.0
              nntoolchain/lib/libbm1684x_kernel_module.a
              nntoolchain/lib/libbmtpulv60_kernel_module.a
              # nntoolchain/lib/libbm1690_kernel_module.so
              nntoolchain/lib/libbm1690_kernel_module.a
              # nntoolchain/lib/libcv184x_kernel_module.so
              # nntoolchain/lib/libcv184x_kernel_module.a
              nntoolchain/lib/libbm1690e_kernel_module.so
              nntoolchain/lib/libbm1690e_kernel_module.a
              nntoolchain/lib/libsgtpuv8_kernel_module.so
              nntoolchain/lib/libsgtpuv8_kernel_module.a
              nntoolchain/lib/liblibtorch_plugin.so
              nntoolchain/lib/libmodel_combine.so
              nntoolchain/lib/libtpuv7_rt.so
              nntoolchain/lib/libtpuv7_modelrt.so
              # nntoolchain/lib/libtpuv7_emulator.so
              nntoolchain/lib/libcdm_daemon_emulator.so
              nntoolchain/lib/libtpuv7_scalar_emulator.so
              or-tools/lib/libortools.so
              or-tools/lib/libortools.so.9
              or-tools/lib/libortools.so.9.8.3296
              # tpu-train/lib/libsgdnn.so

              # cvitek
              CV18xx/lib/libcvikernel.so
              CV18xx/lib/libcviruntime.so
              CV18xx/lib/libcvicmodel.so
              CV18xx/lib/libunwind.so.8
              atomic_exec/libbm1684x_atomic_kernel.so
              # ppl (nntoolchain/ppl/README.md for details)
              # cmodel libs
              nntoolchain/ppl/libcmodel_1684x.so
              nntoolchain/ppl/libcmodel_1688.so
              nntoolchain/ppl/libtpuv7_emulator.so
              nntoolchain/ppl/libtpuv7.1_emulator.so
              # device libs
              nntoolchain/ppl/libbm1684x_kernel_module.so
              nntoolchain/ppl/libbmtpulv60_kernel_module.so
              nntoolchain/ppl/libbm1690_kernel_module.so
              # dyn host libs
              nntoolchain/ppl/libppl_dyn_host_bm1684x.so
              nntoolchain/ppl/libppl_dyn_host_bm1688.so
              nntoolchain/ppl/libppl_dyn_host_bm1690.so
              # static host lib
              nntoolchain/ppl/libppl_host.so
        DESTINATION lib)

# install(DIRECTORY ppl
#         DESTINATION ${CMAKE_INSTALL_PREFIX}
#         PATTERN "*.sh" PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
#         PATTERN "ppl-compile" PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
# if using install(DIRECTORY), the files will miss access priority
install(CODE "
        execute_process(COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/ppl ${CMAKE_INSTALL_PREFIX}/)
")
