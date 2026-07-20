if ("$ENV{CUSTOM_LAYER_CHIP_ARCH}" STREQUAL "bm1684x")
  set(CMODEL_CUSTOM_1684X_OUTPUT_FILE cmodel_custom_1684x)
  add_library(${CMODEL_CUSTOM_1684X_OUTPUT_FILE} SHARED ${SRC_FILES})
  # This plugin is loaded by an existing cmodel/backend instance. Linking the
  # cmodel shared object here creates a second cmodel instance in tpuc-opt.
  target_link_libraries(${CMODEL_CUSTOM_1684X_OUTPUT_FILE} m)
  install(TARGETS ${CMODEL_CUSTOM_1684X_OUTPUT_FILE} LIBRARY DESTINATION lib)
elseif ("$ENV{CUSTOM_LAYER_CHIP_ARCH}" STREQUAL "bm1688")
  set(CMODEL_CUSTOM_1688_OUTPUT_FILE cmodel_custom_1688)
  add_library(${CMODEL_CUSTOM_1688_OUTPUT_FILE} SHARED ${SRC_FILES})
  # See the BM1684X target above: resolve cmodel APIs from the host process.
  target_link_libraries(${CMODEL_CUSTOM_1688_OUTPUT_FILE} m)
  install(TARGETS ${CMODEL_CUSTOM_1688_OUTPUT_FILE} LIBRARY DESTINATION lib)
else()
  message(FATAL_ERROR "Unknown chip arch: " $ENV{CUSTOM_LAYER_CHIP_ARCH})
endif()
