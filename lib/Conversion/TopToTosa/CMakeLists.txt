file(GLOB_RECURSE _sources *.cpp)
tpu_add_library(ConversionTopToTosa ${_sources}
  DEPENDS
  TpuPassConversionIncGen
  LIBS
	TpuLibDialectTop
	MLIRTosaDialect
  )
