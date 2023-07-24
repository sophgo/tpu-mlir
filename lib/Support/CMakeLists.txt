file(GLOB_RECURSE _sources *.cpp)
tpu_add_library(Support ${_sources}
	DEPENDS 
	TpuEnumModuleIncGen
	TpuDialectTopOpsIncGen
	LIBS
  MLIRIR
  MLIRSupport
  MLIRQuantDialect
  MLIRQuantUtils
	TpuLibInterfaceInferenceInterface
	cnpy
	dnnl
	z
)
