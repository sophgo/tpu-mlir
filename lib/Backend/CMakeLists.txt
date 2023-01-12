# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources
     *.cpp
     BM168x/*.cpp
     CV18xx/*.cpp
     CV18xx/Kernel/*.cpp)

add_llvm_library(TPUMLIRBackend
  ${_sources}
  LINK_LIBS PUBLIC
  TPUMLIRSupport
  MLIRLocalGenInterface
  MLIRDynLocalGenInterface
)
