file(GLOB_RECURSE _sources *.cpp)

tpu_add_library(DialectTpu ${_sources}
	DEPENDS 
	BuilderHeader
	TpuDialectTopOpsIncGen
	TpuDialectTpuOpsIncGen
	TpuPassTopIncGen
	TpuPassTpuIncGen
	LIBS
	TpuLibInterfaceInferenceInterface
	TpuLibInterfaceGlobalGenInterface
	TpuLibInterfaceLocalGenInterface
	TpuLibInterfaceDynGlobalGenInterface
	TpuLibInterfaceDynLocalGenInterface
	TpuLibInterfaceTypeInterface
	TpuLibInterfaceIndexingMapsInterface
	TpuLibBackend
	TpuLibBuilder
	TpuLibTrait
	TpuLibSupport
	crypto
)
