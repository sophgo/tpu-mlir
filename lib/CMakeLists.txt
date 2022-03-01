add_subdirectory(Interfaces)
add_subdirectory(Support)
add_subdirectory(Dialect)

add_mlir_library(SophgoInitAll
  InitAll.cpp
  ModuleInterpreter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferenceInterface
  SophgoTops
  SophgoSupport
)

