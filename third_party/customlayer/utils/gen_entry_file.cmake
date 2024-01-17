set(ENTRY_FILE "${CMAKE_BINARY_DIR}/src/entry.c")
file(APPEND ${ENTRY_FILE}
    "\#include <string.h>\n"
    "\#include \"interface_custom_ops.h\"\n"
    "\#include \"api_common.h\"\n"
    "int tpu_global_calculate_entry\(const char* name, const void* param, const int param_size,\n"
    "                               const tensor_spec_t* in_tensors, const int in_num,\n"
    "                               const unsigned long long buffer_addr, const unsigned long long buffer_size,\n"
    "                               tensor_spec_t* out_tensors, const int out_num\) {\n"
    "  if (false) {\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        "  } else if \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    api_" ${op_name} "_global\(in_tensors, out_tensors, param\);\n"
    )
endforeach()
file(APPEND ${ENTRY_FILE}
    "  }\n"
    "  return 0;\n"
    "}\n"
    "int tpu_shape_infer_entry\(const char* name, const void* param, const int param_size,\n"
    "                          const tensor_spec_t* in_tensors, const int in_num,\n"
    "                          tensor_spec_t* out_tensors, const int out_num\) {\n"
    "  if (false) {\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${ENTRY_FILE}
        "  } else if \(strcmp\(name, \"" ${op_name} "\"\) == 0\) {\n"
        "    shape_infer_" ${op_name} "\(in_tensors, out_tensors, param\);\n"
    )
endforeach()
file(APPEND ${ENTRY_FILE}
    "  }\n"
    "  return 0;\n"
    "}\n"
)
