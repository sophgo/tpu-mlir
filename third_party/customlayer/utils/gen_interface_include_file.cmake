set(API_INCLUDE_FILE "${CMAKE_BINARY_DIR}/include/interface_custom_ops.h")
file(APPEND ${API_INCLUDE_FILE}
    "\#pragma once\n"
    "\#include \"api_common.h\"\n"
    "\#include \"tpu_impl_custom_ops.h\"\n"
    "\n"
    "// shape infer function\n"
    "\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "void shape_infer_" ${op_name} "\(\n"
        "    const global_tensor_spec_t *input,\n"
        "    global_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
file(APPEND ${API_INCLUDE_FILE}
    "\n"
    "// global api function\n"
    "\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "void api_" ${op_name} "_global\(\n"
        "    const global_tensor_spec_t *input,\n"
        "    global_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
