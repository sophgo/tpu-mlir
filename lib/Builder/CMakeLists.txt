# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources *.cpp)

add_llvm_library(TPUMLIRBuilder
  ${_sources}

  DEPENDS
  Builder
)
