# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources BM168x/*.cpp)

add_llvm_library(SophgoBackend
  ${_sources}
  LINK_LIBS PUBLIC
  SophgoSupport
  MLIRLocalGenInterface
)

