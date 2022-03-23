set(LLVM_OPTIONAL_SOURCES
  InferenceInterface.cpp
  QuantizeInterface.cpp
  WeightReorderInterface.cpp
  CodegenInterface.cpp
  )

function(add_mlir_interface_library name)
  add_mlir_library(MLIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
endfunction(add_mlir_interface_library)

add_mlir_interface_library(InferenceInterface)
add_mlir_interface_library(QuantizeInterface)
add_mlir_interface_library(WeightReorderInterface)
add_mlir_interface_library(CodegenInterface)
