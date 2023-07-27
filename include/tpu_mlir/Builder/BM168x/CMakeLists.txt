function(compile_fbs)
  get_filename_component(CMAKE_INSTALL_PREFIX_ABS "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
  foreach(fbs_file IN LISTS ARGN)
    get_filename_component(name ${fbs_file} NAME_WE)
    set(target_header ${name}_generated.h)
		set(fbs_header ${name}_fbs.h)
    message(STATUS "To compile fbs files ${fbs_file} for ${target_header}")
    add_custom_command(OUTPUT ${target_header}
      DEPENDS ${fbs_file}
      COMMAND flatc -o .  --gen-mutable --gen-object-api  --cpp ${fbs_file}
    )
		set(targets ${targets} ${target_header})
		add_custom_command(OUTPUT ${fbs_header}
      DEPENDS ${fbs_file}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_header.sh ${fbs_file} ${name}
    )
    set(python_targets ${CMAKE_INSTALL_PREFIX_ABS}/python/debugger/bmodel_fbs.py)
    add_custom_command(
      OUTPUT ${python_targets}
      DEPENDS ${fbs_file}
      COMMAND flatc -o ${CMAKE_CURRENT_BINARY_DIR} --python ${fbs_file}
      COMMAND ${CMAKE_SOURCE_DIR}/python/utils/merge_pyfbs.py ${CMAKE_CURRENT_BINARY_DIR}/bmodel ${python_targets}
    )
    set(targets ${targets} ${fbs_header} ${python_targets})
  endforeach()
  add_custom_target(Builder ALL DEPENDS ${targets})
endfunction()

compile_fbs(
  ${CMAKE_CURRENT_SOURCE_DIR}/bmodel.fbs
)
