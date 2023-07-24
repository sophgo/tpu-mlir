file(GLOB_RECURSE _sources *.cpp)
tpu_add_executable(model_tool ${_sources}
	LIBS
	TpuLibBuilder
	flatbuffers
	stdc++fs
)
install(TARGETS model_tool DESTINATION bin)
