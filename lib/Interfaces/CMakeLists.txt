set(LLVM_OPTIONAL_SOURCES
  InferenceInterface.cpp
  WeightReorderInterface.cpp
  GlobalGenInterface.cpp
  FlopsInterface.cpp
  LocalGenInterface.cpp
  TypeInterface.cpp
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
add_mlir_interface_library(GlobalGenInterface)
add_mlir_interface_library(FlopsInterface)
add_mlir_interface_library(LocalGenInterface)
add_mlir_interface_library(TypeInterface)
