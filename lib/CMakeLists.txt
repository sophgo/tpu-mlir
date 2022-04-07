add_subdirectory(Interfaces)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Backend)
add_subdirectory(Builder)

add_mlir_library(SophgoInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunc
  MLIRQuant
  MLIRInferenceInterface
  SophgoTop
  SophgoTpu
  SophgoSupport
)

