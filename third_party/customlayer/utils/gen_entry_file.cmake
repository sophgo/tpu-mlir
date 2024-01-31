set(ENTRY_FILE "${CMAKE_BINARY_DIR}/src/entry.c")
file(APPEND ${ENTRY_FILE}
    "\#include <string.h>\n"
    "\#include \"interface_custom_ops.h\"\n"
    "\#include \"api_common.h\"\n"
    "int tpu_global_calculate_entry\(const char* name, const void* param, const int param_size,\n"
    "                                const tensor_spec_t* in_tensors, const int in_num,\n"
    "                                tensor_spec_t* out_tensors, const int out_num\) {\n"
)
set(PREFIX_IF "  if")
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        ${PREFIX_IF} " \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    api_" ${op_name} "_global\(in_tensors, out_tensors, param\);\n"
    )
    set(PREFIX_IF "  } else if")
endforeach()
file(APPEND ${ENTRY_FILE}
    "  }\n"
    "  return 0;\n"
    "}\n"
)
file(APPEND ${ENTRY_FILE}
    "int tpu_local_calculate_entry\(const char* name, const void* param, const int param_size,\n"
    "                               const tensor_spec_t* in_tensors, const int in_num,\n"
    "                               const local_sec_info_t* sec_info,\n"
    "                               tensor_spec_t* out_tensors, const int out_num\) {\n"
)
set(PREFIX_IF "  if")
foreach(op_name ${REGSTERED_LOCAL_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        ${PREFIX_IF} " \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    api_" ${op_name} "_local\(sec_info, in_tensors, out_tensors, param\);\n"
    )
    set(PREFIX_IF "  } else if")
endforeach()
file(APPEND ${ENTRY_FILE}
    "  }\n"
    "  return 0;\n"
    "}\n"
)
file(APPEND ${ENTRY_FILE}
    "void default_type_infer\(const tensor_spec_t* in_tensors, tensor_spec_t* out_tensors, const void* param\) {\n"
    "  out_tensors->dtype = in_tensors->dtype;\n"
    "  out_tensors->dims = in_tensors->dims;\n"
    "  memcpy(out_tensors->shape, in_tensors->shape, out_tensors->dims * sizeof(int));\n"
    "  out_tensors->elem_num = in_tensors->elem_num;\n"
    "}\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        "void __attribute__((weak)) type_infer_" ${op_name} "\(const tensor_spec_t* in_tensors, tensor_spec_t* out_tensors, const void* param\) {\n"
        "  default_type_infer\(in_tensors, out_tensors, param\);\n"
        "}\n"
    )
endforeach()
file(APPEND ${ENTRY_FILE}
    "int tpu_shape_infer_entry\(const char* name, const void* param, const int param_size,\n"
    "                           const tensor_spec_t* in_tensors, const int in_num,\n"
    "                           tensor_spec_t* out_tensors, const int out_num\) {\n"
)
set(PREFIX_IF "  if")
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        ${PREFIX_IF} " \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    type_infer_" ${op_name} "\(in_tensors, out_tensors, param\);\n"
    )
    set(PREFIX_IF "  } else if")
endforeach()
file(APPEND ${ENTRY_FILE}
    "  } else {\n"
    "    default_type_infer(in_tensors, out_tensors, param);\n"
    "  }\n"
    "  return 0;\n"
    "}\n"
)
file(APPEND ${ENTRY_FILE}
    "void default_slice_infer\(const local_sec_info_t* sec_info, const tensor_spec_t* in_tensors,\n"
    "                          tensor_slice_t* out_slices, const void* param\) {\n"
    "  out_slices->h_idx = sec_info->h_idx;\n"
    "  out_slices->h_slice = sec_info->h_slice;\n"
    "}\n"
)
foreach(op_name ${REGSTERED_LOCAL_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        "void __attribute__((weak)) slice_infer_" ${op_name} "\(const local_sec_info_t* sec_info, const tensor_spec_t* in_tensors, tensor_slice_t* out_slices, const void* param\) {\n"
        "  default_slice_infer\(sec_info, in_tensors, out_slices, param\);\n"
        "}\n"
    )
endforeach()
file(APPEND ${ENTRY_FILE}
    "int tpu_slice_infer_entry\(const char* name, const void* param, const int param_size,\n"
    "                           const tensor_spec_t* in_tensors, const int in_num,\n"
    "                           const local_sec_info_t* sec_info,\n"
    "                           tensor_slice_t* out_slices, const int out_num\) {\n"
)
set(PREFIX_IF "  if")
foreach(op_name ${REGSTERED_LOCAL_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        ${PREFIX_IF} " \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    slice_infer_" ${op_name} "\(sec_info, in_tensors, out_slices, param\);\n"
    )
    set(PREFIX_IF "  } else if")
endforeach()
file(APPEND ${ENTRY_FILE}
    "  } else {\n"
    "    default_slice_infer(sec_info, in_tensors, out_slices, param);\n"
    "  }\n"
    "  return 0;\n"
    "}\n"
)
