function(compile_fbs)
  foreach(fbs_file IN LISTS ARGN)
    get_filename_component(name ${fbs_file} NAME_WE)
    set(target_header ${name}_generated.h)
    set(fbs_header ${name}_fbs.h)
    message(STATUS "To compile fbs files ${fbs_file} for ${target_header}")
    add_custom_command(OUTPUT ${target_header}
      DEPENDS ${fbs_file}
      COMMAND ${FLATBUFFERS_CMD} -o .  --gen-mutable --gen-object-api  --cpp ${fbs_file}
    )
    add_custom_target(${target_header}_t DEPENDS ${target_header})
    add_dependencies(BuilderHeader ${target_header}_t)
    add_custom_command(OUTPUT ${fbs_header}
      DEPENDS ${fbs_file}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_header.sh ${fbs_file} ${name}
    )
    add_custom_target(${fbs_header}_t DEPENDS ${fbs_header})
    add_dependencies(BuilderHeader ${fbs_header}_t)
  endforeach()
endfunction()

compile_fbs(
  ${CMAKE_CURRENT_SOURCE_DIR}/bmodel.fbs
)
