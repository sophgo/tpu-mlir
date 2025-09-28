#ifndef __ATOMIC_DEF_H__
#define __ATOMIC_DEF_H__

#define SG_TV_GEN 1
#define SIGN(dtype) ((dtype) & 0x1)
#define PRECISION(dtype) (((dtype) >> 1) & 0xf)
#define FP8TYPE(dtype) ((dtype) >> 5)
#define WIDTH(dtype) tpu_data_type_bits(dtype)
#define DSIZE(dtype) tpu_data_type_size(dtype)
#define ALIGNED_OR_USER(stride) ((stride) == NULL ? 0 : 3)
#define MAX_TPU_CORE_NUM 4

// =============================================
// The following is allocation for static memory
// For lookup table
// Align with 128byte
#define SFU_TAYLOR_TABLE_SIZE 32
#define SFU_TAYLOR_L_TABLE_SIZE 64
#define ERF_TAYLOR_SIZE 16
#define STATIC_MEM_OFFSET 0
#define SERIAL_NUMBER_SIZE 64
#define SIN_TAYLOR_SIZE 32
#define COS_TAYLOR_SIZE 32
#define ARCSIN_TAYLOR_SIZE 64
#define TAN_TAYLOR_SIZE 32
#define EXP_TAYLOR_OFFSET (STATIC_MEM_OFFSET)
#define LOG_TAYLOR_OFFSET                                                      \
  (EXP_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float))
#define ERF_TAYLOR_OFFSET                                                      \
  (LOG_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(float))
#define SERIAL_NUMBER_OFFSET                                                   \
  (ERF_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float)) // align 128 byte
#define SIN_TAYLOR_OFFSET                                                      \
  (SERIAL_NUMBER_OFFSET + SERIAL_NUMBER_SIZE * sizeof(float))
#define COS_TAYLOR_OFFSET (SIN_TAYLOR_OFFSET + SIN_TAYLOR_SIZE * sizeof(float))
#define ARCSIN_TAYLOR_OFFSET                                                   \
  (COS_TAYLOR_OFFSET + COS_TAYLOR_SIZE * sizeof(float))
#define TAN_TAYLOR_OFFSET                                                      \
  (ARCSIN_TAYLOR_OFFSET + ARCSIN_TAYLOR_SIZE * sizeof(float))
#define EXP_FP16_TAYLOR_OFFSET                                                 \
  (TAN_TAYLOR_OFFSET + TAN_TAYLOR_SIZE * sizeof(float))
#define EXP_BF16_TAYLOR_OFFSET                                                 \
  (EXP_FP16_TAYLOR_OFFSET +                                                    \
   SFU_TAYLOR_L_TABLE_SIZE * sizeof(short)) // align 128 byte
#define ERF_FP16_TAYLOR_OFFSET                                                 \
  (EXP_BF16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define ERF_BF16_TAYLOR_OFFSET                                                 \
  (ERF_FP16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define LOG_FP16_TAYLOR_OFFSET                                                 \
  (ERF_BF16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define LOG_BF16_TAYLOR_OFFSET                                                 \
  (LOG_FP16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define SIN_FP16_TAYLOR_OFFSET                                                 \
  (LOG_BF16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_BFP16_TAYLOR_OFFSET                                                \
  (SIN_FP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_FP16_TAYLOR_OFFSET                                                 \
  (SIN_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_BFP16_TAYLOR_OFFSET                                                \
  (COS_FP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SMEM_STATIC_END_OFFSET                                                 \
  (COS_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))

typedef enum {
  TEEW_E8 = 0,
  TEEW_E16 = 1,
  TEEW_E32 = 2,
  TEEW_E4 = 5,
} TEEW_E;

#define RVT_CFGGR(ga, is_int, teew, subtype, layout, addr)                     \
  rvt_cfggr(ga, ((uint64_t)is_int << 63 | (uint64_t)teew << 60 |               \
                 (uint64_t)subtype << 59 | (uint64_t)layout << 52 |            \
                 ((uint64_t)addr & 0xffffffffffff)))

#define RVT_CFGGR_SHAPE(ga, n, c, h, w)                                        \
  rvt_cfggr_shape(ga, (((uint64_t)n & 0xffff) << 48 |                          \
                       (((uint64_t)c & 0xffff) << 32) |                        \
                       (((uint64_t)h & 0xffff) << 16) | (uint64_t)w & 0xffff))

#define RVT_CFGGR_HWDIM(ga, h, w)                                              \
  rvt_cfggr_hwdim(ga, (((uint64_t)h & 0xffffffff) << 32) |                     \
                          (((uint64_t)w & 0xffffffff)))

#define RVT_CFGGR_NCDIM(ga, n, c)                                              \
  rvt_cfggr_ncdim(ga, (((uint64_t)n & 0xffff) << 16) | (((uint64_t)c & 0xffff)))

#define RVT_CFGGR_REG_SHAPE(ga, n, c, h, w)                                    \
  if (h < (1 << 16) && w < (1 << 16)) {                                        \
    RVT_CFGGR_SHAPE(ga, n, c, h, w);                                           \
  } else {                                                                     \
    RVT_CFGGR_HWDIM(ga, h, w);                                                 \
    RVT_CFGGR_NCDIM(ga, n, c);                                                 \
  }

#define RVT_CFGTR(ga, is_int, teew, subtype, layout, addr)                     \
  rvt_cfgtr(ga, ((uint64_t)is_int << 63 | (uint64_t)teew << 60 |               \
                 (uint64_t)subtype << 59 | (uint64_t)layout << 52 |            \
                 ((uint64_t)addr)))

#define RVT_CFGTR_SHAPE(ga, n, c, h, w)                                        \
  rvt_cfgtr_shape(ga, (((uint64_t)n & 0xffff) << 48 |                          \
                       ((uint64_t)c & 0xffff) << 32 |                          \
                       ((uint64_t)h & 0xffff) << 16 | (uint64_t)w & 0xffff))

#define RVT_GR_STRIDE(ga, layout, n, c, h, w)                                  \
  if (layout == FREE_LAYOUT) {                                                 \
    rvt_cfggr_ncstride(ga, (uint64_t)n << 32 | (uint64_t)c);                   \
    rvt_cfggr_hwstride(ga, (uint64_t)h << 32 | (uint64_t)w);                   \
  }

#define RVT_TR_STRIDE(ga, layout, n, c, h, w)                                  \
  if (layout == FREE_LAYOUT) {                                                 \
    rvt_cfgtr_ncstride(ga, (uint64_t)n << 32 | (uint64_t)c);                   \
    rvt_cfgtr_hwstride(ga, (uint64_t)h << 32 | (uint64_t)w);                   \
  }

#define RVT_CR(ga, is_int, teew, subtype, val)                                 \
  rvt_cfgcr(ga,                                                                \
            ((uint64_t)is_int << 63 | (uint64_t)teew << 60 |                   \
             (uint64_t)subtype << 59 | ((uint64_t)val & 0xffffffffffffff)))
inline static u64 gdma_get_lane_mask() {
  u64 lane_mask = 0xffffffffffffffff;
  // #if defined(USING_CMODEL)
  //     char *en_lane_mask = getenv("TV_GEN_EN_LANE_MASK");
  //     char *p = getenv("TV_GEN_LOG_PATH");
  //     char path_int[1024 * 2] = {'\0'};
  //     if (p) {
  //         strcpy(path_int, p);
  //     } else {
  //         strcpy(path_int, "");
  //     }
  //     strcat(path_int, "lane_mask_param");
  //     if (en_lane_mask && access(path_int, F_OK) == 0 && atoi(en_lane_mask)
  //     == 1) {
  //         FILE *file = fopen(path_int, "r");
  //         fscanf(file, "%llx\n", &lane_mask);
  //         fclose(file);
  //     } else if (en_lane_mask && atoi(en_lane_mask) == 1) {
  //         lane_mask = 0;
  //         for (int i = 0; i < 64; i++) {
  //             lane_mask |= (rand() % 3 ? 1ull : 0ull) << i;
  //         }
  //         if (lane_mask == 0)
  //             lane_mask = 1ull << (rand() % NPU_NUM);

  //         FILE *file = fopen(path_int, "w");
  //         fprintf(file, "%llx\n", lane_mask);
  //         fclose(file);
  //     }
  // #endif
  return lane_mask;
}

#endif
