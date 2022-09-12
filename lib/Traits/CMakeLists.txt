file(GLOB _sources *.cpp)

add_llvm_library(TPUMLIRTrait
  ${_sources}
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuantDialect
)
