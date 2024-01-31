set(BACKEND_API_FILE "${CMAKE_BINARY_DIR}/src/backend_custom_api.c")
file(APPEND ${BACKEND_API_FILE}
    "\#include \"api_common.h\"\n"
    "\#include \"backend_helper.h\"\n"
    "\#include \"common_def.h\"\n"
    "\#include \"interface_custom_ops.h\"\n"
    "\n"
    "// global backend api functions\n"
)
foreach(op_name ${REGSTERED_OP_NAMES})
    file(APPEND ${BACKEND_API_FILE}
        "IMPL_CUSTOM_API_GLB\(" ${op_name} "\)\n"
    )
endforeach()
foreach(op_name ${REGSTERED_GLOBAL_BFSZ_NAMES})
    file(APPEND ${BACKEND_API_FILE}
        "IMPL_CUSTOM_API_GLB_BFSZ\(" ${op_name} "\)\n"
    )
endforeach()
file(APPEND ${BACKEND_API_FILE}
    "\n"
    "// local backend api functions\n"
)
foreach(op_name ${REGSTERED_LOCAL_OP_NAMES})
    file(APPEND ${BACKEND_API_FILE}
        "IMPL_CUSTOM_API_LOC\(" ${op_name} "\)\n"
    )
endforeach()
foreach(op_name ${REGSTERED_LOCAL_BFSZ_NAMES})
    file(APPEND ${BACKEND_API_FILE}
        "IMPL_CUSTOM_API_LOC_BFSZ\(" ${op_name} "\)\n"
    )
endforeach()
