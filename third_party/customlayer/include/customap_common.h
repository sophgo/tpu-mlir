#ifndef _CUSTOMAP_COMMON_H_
#define _CUSTOMAP_COMMON_H_
#include <string.h>

#define CPU_CUSTOM_LAYER_TYPE_T AP_CUSTOM_LAYER_TYPE_T
typedef enum {
    AP_CUSTOM                                 = 10001,
    AP_CUSTOM_TOPK                            = 10002,
    AP_CUSTOM_LAYER_NUM                          ,
    AP_CUSTOM_LAYER_UNKNOW = AP_CUSTOM_LAYER_NUM,
} AP_CUSTOM_LAYER_TYPE_T;


#ifndef _CPU_COMMON_H_
typedef enum {
    CPU_DTYPE_FP32 = 0,
    CPU_DTYPE_FP16 = 1,
    CPU_DTYPE_INT8 = 2,
    CPU_DTYPE_UINT8 = 3,
    CPU_DTYPE_INT16 = 4,
    CPU_DTYPE_UINT16 = 5,
    CPU_DTYPE_INT32 = 6,
    CPU_DTYPE_UINT32 = 7,
    CPU_DTYPE_BFP16 = 8,
    CPU_DTYPE_UNKNOWN = -1,
} CPU_DATA_TYPE_T;
#endif

typedef struct {
    const char* name;
    CPU_CUSTOM_LAYER_TYPE_T value;
} NameToEnum;

#define NAME_TO_ENUM_MAP(ENUM) { #ENUM, AP_CUSTOM_##ENUM }

static const NameToEnum custom_layer_map[] = {
    NAME_TO_ENUM_MAP(TOPK),
    // map new name to optype here
    {NULL, AP_CUSTOM_LAYER_UNKNOW}
};

static CPU_CUSTOM_LAYER_TYPE_T GetCustomLayerType(const char* name) {
    const NameToEnum* current = custom_layer_map;
    while (current->name != NULL) {
        if (strcmp(current->name, name) == 0) {
            return current->value;
        }
        ++current;
    }
    return AP_CUSTOM_LAYER_UNKNOW;
}
#endif /* _CUSTOMAP_COMMON_H_ */
