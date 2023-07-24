file(GLOB TPU_LIBS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(fname ${TPU_LIBS})
	string(REPLACE ".cpp" "" name ${fname})
		#INCLUDES ${MLIR_INCLUDE_DIRS}
	tpu_add_library(Interface${name} ${fname}
		DEPENDS TpuInterface${name}IncGen
		LIBS MLIRIR
		)
endforeach()
