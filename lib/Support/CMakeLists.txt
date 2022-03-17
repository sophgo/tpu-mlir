# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources *.cpp Dnnl/*.cpp Helper/*.cpp)

add_llvm_library(SophgoSupport
  ${_sources}
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRQuant
  MLIRQuantUtils
  MLIRInferenceInterface
  cnpy
  dnnl
  z
)
