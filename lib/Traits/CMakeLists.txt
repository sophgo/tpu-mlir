file(GLOB _sources *.cpp)
tpu_add_library(Trait ${_sources} LIBS MLIRIR MLIRQuantDialect)
