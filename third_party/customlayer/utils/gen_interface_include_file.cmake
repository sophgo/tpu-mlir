set(API_INCLUDE_FILE "${CMAKE_BINARY_DIR}/include/interface_custom_ops.h")
file(APPEND ${API_INCLUDE_FILE}
    "\#pragma once\n"
    "\#include \"api_common.h\"\n"
    "\#include \"tpu_impl_custom_ops.h\"\n"
    "\n"
)
file(APPEND ${API_INCLUDE_FILE}
    "\n"
    "// global api function\n"
    "\n"
)
foreach(op_name ${REGSTERED_GLOBAL_BFSZ_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "int64_t api_" ${op_name} "_global_bfsz\(\n"
        "    const global_tensor_spec_t *input,\n"
        "    global_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "void api_" ${op_name} "_global\(\n"
        "    const global_tensor_spec_t *input,\n"
        "    global_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
file(APPEND ${API_INCLUDE_FILE}
    "\n"
    "// local api function\n"
    "\n"
)
foreach(op_name ${REGSTERED_LOCAL_BFSZ_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "int api_" ${op_name} "_local_bfsz\(\n"
        "    const local_sec_info_t *sec_info,\n"
        "    const local_tensor_spec_t *input,\n"
        "    local_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
foreach(op_name ${REGSTERED_LOCAL_OP_NAMES})
    file(APPEND ${API_INCLUDE_FILE}
        "void api_" ${op_name} "_local\(\n"
        "    const local_sec_info_t *sec_info,\n"
        "    const local_tensor_spec_t *input,\n"
        "    local_tensor_spec_t *output,\n"
        "    const void *param\);\n"
    )
endforeach()
if (NOT "$ENV{CUSTOM_LAYER_DEV_MODE}" STREQUAL "backend")
    file(APPEND ${API_INCLUDE_FILE}
        "// type infer function\n"
        "\n"
    )
    foreach(op_name ${REGSTERED_OP_NAMES})
        file(APPEND ${API_INCLUDE_FILE}
            "void type_infer_" ${op_name} "\(\n"
            "    const global_tensor_spec_t *input,\n"
            "    global_tensor_spec_t *output,\n"
            "    const void *param\);\n"
        )
    endforeach()
    file(APPEND ${API_INCLUDE_FILE}
        "// slice infer function\n"
        "\n"
    )
    foreach(op_name ${REGSTERED_OP_NAMES})
        file(APPEND ${API_INCLUDE_FILE}
            "void slice_infer_" ${op_name} "\(\n"
            "    const local_sec_info_t* sec_info,\n"
            "    const local_tensor_spec_t *input,\n"
            "    tensor_slice_t *output,\n"
            "    const void *param\);\n"
        )
    endforeach()
endif()
