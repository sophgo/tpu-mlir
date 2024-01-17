#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp  : 5;  // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 7; // mantissa
    uint16_t exp  : 8; // exponent
    uint16_t sign : 1; // sign
  } format;
} bf16;

typedef union {
  float    fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp  : 8;  // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

typedef union {
  double double_val;
  uint64_t bits;
} Double;

typedef struct {
  int8_t val : 4;
} int4_t;

typedef struct {
  uint8_t val : 4;
} uint4_t;

/* info about cmd_node */
typedef struct gdma_cmd_node_info_s {
  int n;
  int c;
  int h;
  int w;
  int direction;
  int src_format;
  int dest_format;
  bool setted;
} gdma_cmd_node_info_t;

typedef struct inst_profile {
  unsigned long long cycle;
  unsigned long long gdma_size;
  int gdma_direction;
  int src_format;
  int dst_format;
  double op_dyn_energy; //nJ
  double sram_rw_energy; // nJ
  double compute_ability;
  bool b_gdma_use_l2;
} INST_PROFILE;

//bm1688 also use it
#define CONFIG_MAX_CDMA_NUM 10
typedef struct cmd_id_node {
  unsigned int bd_cmd_id;
  unsigned int gdma_cmd_id;
  unsigned int hau_cmd_id;
  bool in_parallel_state;
  long long cycle_count;
  long long cur_op_cycle;
  char cmd_name[16];
  char name_prefix[64];
  gdma_cmd_node_info_t gdma_cmd_info;
  INST_PROFILE inst_profile;
  unsigned int sdma_cmd_id;
  unsigned int cdma_cmd_id[CONFIG_MAX_CDMA_NUM];
} CMD_ID_NODE;

#ifdef __cplusplus
}
#endif

#endif
