file(GLOB _sources *.cpp)

add_llvm_library(cnpy
  ${_sources}
)
