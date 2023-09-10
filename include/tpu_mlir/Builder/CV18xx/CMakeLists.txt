cmake_minimum_required(VERSION 3.1.0)


function(compile_fbs)
  foreach(fbs_file IN LISTS ARGN)
    get_filename_component(name ${fbs_file} NAME_WE)
    set(target_header ${name}_generated.h)
    message(STATUS "To compile fbs files ${fbs_file} for ${target_header}")
    add_custom_command(OUTPUT ${target_header}
      DEPENDS ${fbs_file}
      COMMAND flatc -o . --cpp ${fbs_file}
    )
    set(targets ${targets} ${target_header})
  endforeach()
  add_custom_target(CV18xx_Builder ALL DEPENDS ${targets})
endfunction()

compile_fbs(
  ${CMAKE_CURRENT_SOURCE_DIR}/proto/cvimodel.fbs
  ${CMAKE_CURRENT_SOURCE_DIR}/proto/parameter.fbs
)
