add_mlir_conversion_library(
  SophgoTopTFLiteToTpu
  TopTFLiteToTpu.cpp


  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include

  DEPENDS
  SophgoConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRInferenceInterface
  SophgoTop
  SophgoTpu
)
