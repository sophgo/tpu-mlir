set(PPL_INCLUDE "")
set(PPL_DEF "")
set(PPL_NO_GEN_DIR "")

function(add_ppl_include name)
	set(PPL_INCLUDE "${PPL_INCLUDE}-I ${name} " PARENT_SCOPE)
endfunction()

function(add_ppl_def name)
	set(PPL_DEF "${PPL_DEF}${name} " PARENT_SCOPE)
endfunction()

function(set_ppl_no_gen_dir)
	set(PPL_NO_GEN_DIR "--no-gen-dir" PARENT_SCOPE)
endfunction()

function(ppl_gen input chip output opt_level)
  message(NOTICE "run ppl_gen:")
  message(NOTICE "${PPL_INCLUDE}")

# Check the environment variables
  if(NOT EXISTS ${output})
    file(MAKE_DIRECTORY ${output})
  endif()

  if(NOT DEFINED ENV{PPL_PROJECT_ROOT})
    message(FATAL_ERROR "PPL_PROJECT_ROOT is not defined. Please source envsetup.sh firstly.")
  endif()
  if(NOT DEFINED chip)
    set(chip bm1684x)
  message(NOTICE "CHIP is not defined, it's set to bm1684x by default.")
  endif()
  set(working_dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
  if(NOT EXISTS ${working_dir})
    file(MAKE_DIRECTORY ${working_dir})
  endif()
  message(STATUS "CHIP: ${chip}")
  message(STATUS "LANGUAGE: c++")
	add_definitions(-D__${chip}__)

# Organize include search paths
  # set(inc_path $ENV{PPL_PROJECT_ROOT}/inc)

  # foreach(arg IN LISTS ARGN)
  #   if(arg MATCHES "^-D([^=]+)=(.*)$")
  #     set(key ${CMAKE_MATCH_1})
  #     set(value ${CMAKE_MATCH_2})

  #     string(FIND "${key}" "CORENUM" position)
  #     if(NOT position EQUAL -1)
  #       list(APPEND corenums "-D${key}=${value}")
  #     endif()
  #   else()
  #     list(APPEND inc_path "--I" "${arg}")
  #   endif()
  # endforeach()
  # message(STATUS "INC_PATH: ${inc_path};")

# Execute commands
  set(ppl_compile_cmd
      "ppl-compile"
      "${input}"
      "--function=*"
			"${PPL_INCLUDE}"
      "${inc_path}"
      "-chip" "${CHIP}"
      "-O2"
      "-o"
      "${output}"
			"${PPL_DEF}"
			"${PPL_NO_GEN_DIR}"
  )

  message(STATUS "COMMAND: ${ppl_compile_cmd}")
  set(log_path ${output})

  execute_process(
    COMMAND ${ppl_compile_cmd}
    RESULT_VARIABLE exec_result
    OUTPUT_VARIABLE exec_output
    ERROR_VARIABLE exec_error
    OUTPUT_FILE "${log_path}/out.log"
    ERROR_FILE "${log_path}/error.log"
    WORKING_DIRECTORY ${working_dir}
  )

# Output error messages
  message(STATUS "INPUT: ${input}")

  file(READ "${log_path}/error.log" error_log_content)
  string(TOUPPER "${error_log_content}" upper_error_log_content)
  string(FIND "${upper_error_log_content}" "ERROR" error_pos)
  if (NOT "${error_pos}" STREQUAL "-1")
    message(FATAL_ERROR "An error was found in the error log:\n${error_log_content}")
  endif()
  string(FIND "${upper_error_log_content}" "FAIL" fail_pos)
  if (NOT "${fail_pos}" STREQUAL "-1")
    message(FATAL_ERROR "A failure was found in the error log:\n${error_log_content}")
  endif()
  if(NOT exec_result EQUAL "0")
    message(FATAL_ERROR "ppl-compile failed Ret:${exec_result} : ${exec_error} :${exec_output}")
  endif()

  message(STATUS "OUTPUT: ${output}\n")

endfunction()
