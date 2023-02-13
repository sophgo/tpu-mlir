file(GLOB _sources
          *.cpp
          BM1684X/*.cpp
          BM1684/*.cpp
          CV18xx/*.cpp
    )

add_mlir_conversion_library(TPUMLIRTopToTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

  DEPENDS
  TpuMlirConversionPassIncGen

  LINK_LIBS PUBLIC
  TPUMLIRTop
  TPUMLIRTpu
  )
