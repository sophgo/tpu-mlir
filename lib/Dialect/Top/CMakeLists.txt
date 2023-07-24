file(GLOB_RECURSE _sources *.cpp)

tpu_add_library(DialectTop ${_sources}
	DEPENDS
	BuilderHeader
	TpuDialectTopOpsIncGen
	TpuPassTopIncGen
	LIBS
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
	TpuLibInterfaceInferenceInterface
	TpuLibInterfaceFlopsInterface
	TpuLibInterfaceShapeInterface
	TpuLibTrait
	TpuLibSupport
)
