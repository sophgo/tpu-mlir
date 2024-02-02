enable_testing()

file(GLOB SOURCES $ENV{CUSTOM_LAYER_UNITTEST_DIR}/test_*.py)
foreach(source ${SOURCES})
	get_filename_component(name ${source} NAME_WE)
	add_test(
		NAME ${name}
		COMMAND python3 $ENV{CUSTOM_LAYER_UNITTEST_DIR}/${name}.py
		WORKING_DIRECTORY $ENV{CUSTOM_LAYER_UNITTEST_DIR}
	)
endforeach()
