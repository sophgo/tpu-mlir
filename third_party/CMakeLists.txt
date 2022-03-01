add_subdirectory(cnpy)

install(FILES oneDNN/lib/libdnnl.so
              oneDNN/lib/libdnnl.so.2
              oneDNN/lib/libdnnl.so.2.5
        DESTINATION lib)
install(DIRECTORY llvm/python_packages/mlir_core/mlir/ DESTINATION python)
