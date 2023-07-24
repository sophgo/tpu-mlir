file(GLOB_RECURSE _sources *.cpp)
tpu_add_executable(cvimodel_debug ${_sources}
	LIBS
	TpuLibBuilder
	TpuLibInitAll
	TpuLibSupport
	cviruntime
)
install(TARGETS cvimodel_debug DESTINATION bin)
