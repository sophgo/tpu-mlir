#ifndef TPU_UTILS_H
#define TPU_UTILS_H

#include "common_def.h"
#include "tpu_kernel.h"

static inline data_type_t tpu_type_convert(sg_data_type_t data_type) {
    data_type_t dtype = DT_FP32;
    switch (data_type) {
    case SG_DTYPE_FP32:    dtype = DT_FP32;    break;
    case SG_DTYPE_UINT32:  dtype = DT_UINT32;  break;
    case SG_DTYPE_INT32:   dtype = DT_INT32;   break;
    case SG_DTYPE_FP16:    dtype = DT_FP16;    break;
    case SG_DTYPE_BFP16:   dtype = DT_BFP16;   break;
    case SG_DTYPE_INT16:   dtype = DT_INT16;   break;
    case SG_DTYPE_UINT16:  dtype = DT_UINT16;  break;
    case SG_DTYPE_INT8:    dtype = DT_INT8;    break;
    case SG_DTYPE_UINT8:   dtype = DT_UINT8;   break;
    case SG_DTYPE_INT4:    dtype = DT_INT4;    break;
    case SG_DTYPE_UINT4:   dtype = DT_UINT4;   break;
    default:
        TPUKERNEL_ASSERT(0);
        break;
    }
    return dtype;
}

static inline rounding_mode_t tpu_round_mode_convert(sg_round_mode_t sg_mode) {
    rounding_mode_t mode = RM_HALF_TO_EVEN;
    switch (sg_mode) {
        case SG_ROUND_EVEN: // = 3,  // 1.5 -> 2    2.5 -> 2
            mode = RM_HALF_TO_EVEN;/* =0 */
            break;
        case SG_ROUND_INF: // = 0,   // 1.5 -> 2   -1.5 -> -2
            mode = RM_HALF_AWAY_FROM_ZERO;/* =1 */
            break;
        case SG_TRIM_ZERO: // = 6,   // 1.6 -> 1   -1.6 -> -1
            mode = RM_TOWARDS_ZERO;/* =2 */
            break;
        case SG_TRIM_DOWN: // = 9,   // 1.6 -> 1   -1.4 -> -2
            mode = RM_DOWN;/* =3 */
            break;
        case SG_TRIM_UP: // = 8,     // 1.4 -> 2   -1.6 -> -1
            mode = RM_UP;/* =4 */
            break;
        case SG_ROUND_UP: // = 1,    // 1.5 -> 2   -1.5 -> -1
            mode = RM_HALF_UP;/* =5 */
            break;
        case SG_ROUND_DOWN: // = 2,  // 1.5 -> 1   -1.5 -> -2
            mode = RM_HALF_DOWN;/* =6 */
            break;
        case SG_ROUND_ODD: // = 4,   // 1.5 -> 1    0.5 -> 1
        case SG_ROUND_ZERO: // = 5,  // 1.5 -> 1   -1.5 -> -1
        case SG_TRIM_INF: // = 7,    // 1.4 -> 2   -1.4 -> -2
        default:
            TPUKERNEL_ASSERT(0);
            break;
    }
    return mode;
}

#endif // TPU_UTILS_H
