file(GLOB_RECURSE _sources *.cpp)
tpu_add_library(Backend ${_sources}
	LIBS
	TpuLibInterfaceLocalGenInterface
	TpuLibInterfaceDynLocalGenInterface
	)
