add_subdirectory(Interfaces)
add_subdirectory(Traits)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Backend)
add_subdirectory(Builder)
add_subdirectory(Support)

tpu_add_library(InitAll InitAll.cpp
	LIBS
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRTosaDialect
	TpuLibInterfaceInferenceInterface
	TpuLibConversionTopToTpu
	TpuLibConversionTopToTosa
)
