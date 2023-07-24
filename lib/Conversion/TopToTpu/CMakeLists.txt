file(GLOB_RECURSE _sources *.cpp)
tpu_add_library(ConversionTopToTpu ${_sources}
  DEPENDS
  TpuPassConversionIncGen
  LIBS
	TpuLibDialectTop
	TpuLibDialectTpu
  )
# ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu
