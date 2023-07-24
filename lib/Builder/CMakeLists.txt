file(GLOB _sources *.cpp)
tpu_add_library(Builder ${_sources}
	DEPENDS BuilderHeader)
