#ifndef COMMON_UTIL_H_
#define COMMON_UTIL_H_
#include "common.h"
#include "tpu_fp16.h"

inline float cast_to_float32(float value) {
    return value;
}
inline float cast_to_float32(fp16 value) {
    return fp16_to_fp32(value).fval;
}
inline float cast_to_float32(bf16 value) {
    return bf16_to_fp32(value).fval;
}

inline int cast_to_int(float value) {
    DataUnion tmp = {.f32val = value};
    return tmp.i32val;
}
inline int cast_to_int(fp16 value) {
    return value.bits;
}
inline int cast_to_int(bf16 value) {
    return value.bits;
}

// array compare for int32/uint32, int16/uint16, int8/uint8
template<typename T>
static int sg_array_cmp_int(
    T *p_exp,
    T *p_got,
    int len,
    const char *info_label,
    int delta) {
    int idx = 0;
    int first_error_idx = -1;
    int max_error_idx   = -1;
    int max_error_value = 0;
    int mismatch_cnt = 0;
    T first_expect_value = 0;
    T first_got_value = 0;
    T max_expect_value = 0;
    T max_got_value = 0;

    T exp_int = 0;
    T got_int = 0;
    for (idx = 0; idx < len; idx++) {
        int error   = 0;
        exp_int = *(p_exp + idx);
        got_int = *(p_got + idx);

        error = abs(exp_int - got_int);
        if (error > 0) {
            if (first_error_idx == -1) {
                first_error_idx = idx;
                first_expect_value = exp_int;
                first_got_value = got_int;
            }
            if(error > max_error_value) {
                max_error_idx = idx;
                max_error_value = error;
                max_expect_value = exp_int;
                max_got_value = got_int;
            }
            mismatch_cnt ++;
        }
        if (error > delta) {
            printf("%s     error      at index %d exp %d got %d\n", info_label, idx, exp_int, got_int);
            printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
            printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
            fflush(stdout);
            return -1;
        }
    }
    if(max_error_idx != -1) {
        printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
        printf("%s  max  mismatch at index %d exp %d got %d (delta %d)\n", info_label, max_error_idx, max_expect_value, max_got_value, delta);
        printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
        fflush(stdout);
    }

    return 0;
}

// array compare for float/fp16/bfp16
template<typename T>
int sg_array_cmp_float(
    T *p_exp,
    T *p_got,
    int len,
    const char *info_label,
    float delta) {
    int max_error_count = 128;
    int idx = 0;
    int total = 0;
    bool only_warning = false;
    if (1e4 <= delta) {
        delta = 1e-2;
        only_warning = true;
    }
    for (idx = 0; idx < len; idx++) {
        float exp_fp32 = cast_to_float32(p_exp[idx]);
        float got_fp32 = cast_to_float32(p_got[idx]);
        int exp_hex = cast_to_int(p_exp[idx]);
        int got_hex = cast_to_int(p_got[idx]);
        if (sg_max(fabs(exp_fp32), fabs(got_fp32)) > 1.0) {
            // compare rel
            if (sg_min(fabs(exp_fp32), fabs(got_fp32)) < 1e-20) {
                printf("%s:%s(): %s rel warning at index %d exp %.20f got %.20f\n", __FILE__, __FUNCTION__, info_label, idx, exp_fp32, got_fp32);
                total++;
                if (max_error_count < total && !only_warning) {
                    return -1;
                }
            }
            if (fabs(exp_fp32 - got_fp32) / sg_min(fabs(exp_fp32), fabs(got_fp32)) > delta) {
                printf(
                    "%s:%s(): %s rel warning at index %d exp %.20f(0x%08X) got %.20f(0x%08X), diff=%.20f\n",
                    __FILE__, __FUNCTION__,
                    info_label, idx,
                    exp_fp32, exp_hex,
                    got_fp32, got_hex,
                    exp_fp32 - got_fp32);
                total++;
                if (max_error_count < total && !only_warning) {
                    return -1;
                }
            }
        } else {
            // compare abs
            if (fabs(exp_fp32 - got_fp32) > delta) {
                printf(
                    "%s:%s(): %s abs warning at index %d exp %.20f(0x%08X) got %.20f(0x%08X), diff=%.20f\n",
                    __FILE__, __FUNCTION__,
                    info_label, idx,
                    exp_fp32, exp_hex,
                    got_fp32, got_hex,
                    exp_fp32 - got_fp32);
                total++;
                if (max_error_count < total && !only_warning) {
                    return -1;
                }
            }
        }

        DataUnion if_val_exp, if_val_got;
        if_val_exp.f32val = exp_fp32;
        if_val_got.f32val = got_fp32;
        if (SG_IS_NAN(if_val_got.i32val) && !SG_IS_NAN(if_val_exp.i32val)) {
            printf("There are nans in %s idx %d\n", info_label, idx);
            printf("floating form exp %.10f got %.10f\n", if_val_exp.f32val, if_val_got.f32val);
            printf("hex form exp %8.8x got %8.8x\n", if_val_exp.i32val, if_val_got.i32val);
            return -2;
        }
    }
    if (0 < total && !only_warning) {
        return -1;
    }
    return 0;
}


#endif
