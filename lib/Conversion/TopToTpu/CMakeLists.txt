file(GLOB _sources
          *.cpp
          BM1684X/*.cpp
          BM1684/*.cpp
          CV18xx/*.cpp
    )

add_mlir_conversion_library(TPUMLIRTopToTpu
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

  DEPENDS
  TpuMlirConversionPassIncGen

  LINK_LIBS PUBLIC
  TPUMLIRTop
  TPUMLIRTpu
  )

if(TPUMLIR_USE_PCH)
    # add_mlir_conversion_library splits into obj.TPUMLIRTopToTpu (OBJECT lib,
    # does compilation) + TPUMLIRTopToTpu (link wrapper). PCH targets the OBJECT lib.
    if(TARGET obj.TPUMLIRTopToTpu)
        target_precompile_headers(obj.TPUMLIRTopToTpu PRIVATE ${TPUMLIR_PCH_HEADERS})
    else()
        target_precompile_headers(TPUMLIRTopToTpu PRIVATE ${TPUMLIR_PCH_HEADERS})
    endif()
endif()
