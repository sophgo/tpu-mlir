add_subdirectory(Interfaces)
add_subdirectory(Traits)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Backend)
add_subdirectory(Builder)
add_subdirectory(CAPI)

add_mlir_library(TPUMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRInferenceInterface
  TPUMLIRTopToTpu
  TPUMLIRTopToTosa
  TPUMLIRTopToLinalg
)

# install(DIRECTORY PplBackend
#         DESTINATION ${CMAKE_INSTALL_PREFIX}
#         PATTERN "*.sh" PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
install(CODE "
  execute_process(COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/PplBackend ${CMAKE_INSTALL_PREFIX}/)
")
