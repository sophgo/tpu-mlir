#ifndef MEMMAP_H
#define MEMMAP_H

// =============================================
// The following is allocation for static memory
// For lookup table
// align with 64 bytes
#define SFU_TABLE_SIZE              256
#define SFU_TAYLOR_TABLE_SIZE       32
#define SFU_TAYLOR_L_TABLE_SIZE     64
#define ERF_TAYLOR_SIZE             16
#define STATIC_MEM_OFFSET           0
#define SERIAL_NUMBER_SIZE          64
#define SIN_TAYLOR_SIZE             32
#define COS_TAYLOR_SIZE             32
#define ARCSIN_TAYLOR_SIZE          64
#define TAN_TAYLOR_SIZE             32
#define ARCTAN_TAYLOR_SIZE          32
#define EXP_TABLE_OFFSET            (STATIC_MEM_OFFSET)
#define EXP_TAYLOR_OFFSET           (EXP_TABLE_OFFSET + SFU_TABLE_SIZE * sizeof(float))
#define LOG_TAYLOR_OFFSET           (EXP_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float))
#define ERF_TAYLOR_OFFSET           (LOG_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(float))
#define SERIAL_NUMBER_OFFSET        (ERF_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_TAYLOR_OFFSET           (SERIAL_NUMBER_OFFSET + SERIAL_NUMBER_SIZE * sizeof(float))
#define COS_TAYLOR_OFFSET           (SIN_TAYLOR_OFFSET + SIN_TAYLOR_SIZE * sizeof(float))
#define ARCSIN_TAYLOR_OFFSET        (COS_TAYLOR_OFFSET + COS_TAYLOR_SIZE * sizeof(float))
#define TAN_TAYLOR_OFFSET           (ARCSIN_TAYLOR_OFFSET + ARCSIN_TAYLOR_SIZE * sizeof(float))
#define EXP_FP16_TAYLOR_OFFSET      (TAN_TAYLOR_OFFSET + TAN_TAYLOR_SIZE * sizeof(float))
#define EXP_BF16_TAYLOR_OFFSET      (EXP_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define LOG_FP16_TAYLOR_OFFSET      (EXP_BF16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define LOG_BF16_TAYLOR_OFFSET      (LOG_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_FP16_TAYLOR_OFFSET      (LOG_BF16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_BFP16_TAYLOR_OFFSET     (SIN_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_FP16_TAYLOR_OFFSET      (SIN_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_BFP16_TAYLOR_OFFSET     (COS_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define ARCTAN_TAYLOR_OFFSET        (COS_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SMEM_STATIC_END_OFFSET      (ARCTAN_TAYLOR_OFFSET + ARCTAN_TAYLOR_SIZE * sizeof(float))
#endif
