add_mlir_conversion_library(
  TPUMLIRTopTFLiteToTpu
  TopTFLiteToTpu.cpp


  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include

  DEPENDS
  TPUMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRInferenceInterface
  TPUMLIRTop
  TPUMLIRTpu
)
