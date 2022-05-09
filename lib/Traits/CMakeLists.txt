file(GLOB _sources *.cpp)

add_llvm_library(SophgoTrait
  ${_sources}
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuant
)
