//===- bm1690Dailect.cpp - BM1690 dialect  --------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Dialect.h.inc"
#include "tpu-mlir/Dialect/BM1690/IR/StructuredOpsInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Types.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690AttrDefs.h.inc"

#include "tpu-mlir/Dialect/BM1690/IR/StructuredOpsInterfaces.h.inc"

#include "BM1690RegDef.h"
#define GET_OP_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Ops.h.inc"

namespace tpu_mlir {
namespace bm1690 {
namespace tiuType {
struct CONV {
  operator int() const { return 0; }
  enum { NORMAL = 0 };
};
struct PD {
  operator int() const { return 1; }
  enum {
    DEPTHWISE = 0,
    AVG_POOLING = 1,
    MIN_POOLING = 3,
    MAX_POOLING = 4,
    ROI_DEPTHWISE = 5,
    ROI_AVG_POOLING = 6,
    ROI_MAX_POOLING = 7,
    ROI_MIN_POOLING = 8
  };
};
struct MM {
  operator int() const { return 2; }
  enum { NORMAL = 1, WRQ = 2, WRQ_RELU = 3, NN = 4, NT = 5, TT = 6 };
};
struct AR {
  operator int() const { return 3; }
  enum {
    MUL = 0,
    NOT = 1,
    ADD = 2,
    SUB = 3,
    MAX = 4,
    MIN = 5,
    LOGIC_SHIFT = 6,
    AND = 7,
    OR = 8,
    XOR = 9,
    SG = 10,
    SE = 11,
    DIV = 12,
    SL = 13,
    DATA_CONVERT = 14,
    ADD_SATU = 15,
    SUB_SATU = 16,
    MAC = 18,
    COPY = 19,
    MUL_SATU = 20,
    ARITH_SHIFT = 21,
    ROTATE_SHIFT = 22,
    ABS = 26,
    FSUBABS = 27,
    GET_FIRST_ONE = 29,
    GET_FIRST_ZERO = 30
  };
};
struct RQDQ {
  operator int() const { return 4; }
  enum {
    RQ_0 = 0,
    RQ_1 = 1,
    DQ_0 = 3,
    DQ_1 = 4,
  };
};
struct TRANS_BC {
  operator int() const { return 5; }
  enum {
    TRAN_C_W_TRANSPOSE = 0,
    TRAN_W_C_TRANSPOSE = 1,
    LANE_COPY = 2,
    LANE_BROAD = 3,
    STATIC_BROAD = 4,
    STATIC_DISTRIBUTE = 5
  };
};
struct SG {
  operator int() const { return 6; }
  enum {
    PL_gather_d1coor = 0,
    PL_gather_d2coor = 1,
    // PL_gather_rec = 2,
    PL_scatter_d1coor = 3,
    PL_scatter_d2coor = 4,
    PE_S_gather_d1coor = 5,
    PE_S_scatter_d1coor = 6,
    PE_M_gather_d1coor = 7,
    PE_S_mask_select = 8,
    PE_S_nonzero = 9,
    // PE_S_scatter_pp_d1coor = 10,
    PE_S_gather_hzd = 13,
    PE_S_scatter_hzd = 14,
    PE_S_mask_selhzd = 15,
    PE_S_nonzero_hzd = 16,
    PE_S_gather_line = 17,
    PE_S_scatter_line = 18,
    // PE_S_mask_seline = 19,
  };
};
struct LAR {
  operator int() const { return 6; }
  enum {
    MM_NORMAL = 0,
  };
};
struct RANDOM_GEN {
  operator int() const { return 8; }
  enum {
    PRNG = 0,                   // set seed
    PRNG_WITH_INTIAL_SEED = 1,  // load state from lmem
    PRNG_WITH_LOADED_STATES = 2 // use global state to generate random number
  };
};
struct SFU {
  operator int() const { return 9; }
  enum { TAYLOR_4X = 12, TAYLOR = 13, NORM = 15, RSQ = 17 };
};
struct LIN {
  operator int() const { return 10; }
  enum { MAC = 1, ADD_SQR = 20, SUB_SQR = 21 };
};
struct SYS_TRWR {
  operator int() const { return 12; }
};
struct CMP {
  operator int() const { return 13; }
  enum { GT_AND_SG = 22, SG = 23, SE = 24, LT_AND_SL = 25, SL = 26 };
};
struct VC {
  operator int() const { return 14; }
  enum {
    MUL = 0,
    ADD = 2,
    SUB = 3,
    MAX = 4,
    MIN = 5,
    AND = 7,
    OR = 8,
    XOR = 9,
    SG = 10,
    SE = 11,
    DIV = 12,
    SL = 13,
    ADD_SATU = 15,
    SUB_SATU = 16,
    MUL_SATU = 20,
    MULDHR = 23
  };
};
struct SYS {
  operator int() const { return 15; }
  enum {
    SPB = 1,               // software power boot
    SWR = 2,               // set bd lane_mask
    SWR_FROM_LMEM = 3,     // set bd lane mask
    SWR_COL_FROM_LMEM = 4, // set bd lane mask
    // BD_SYNC_ID = 5,
    SEND_MSG = 8,
    WAIT_MSG = 9,
    RANDOM_SEED = 10,
    END = 31 // end instruction for descriptor mode
  };
};
} // namespace tiuType

namespace dmaType {
struct TENSOR {
  operator int() const { return 0; }
  enum { NONE = 0, TRANS = 1, BROADCAST = 3 };
};
struct MATRIX {
  operator int() const { return 1; }
  enum { NONE = 0, TRANS = 1 };
};
struct FILTER {
  operator int() const { return 2; }
  enum { NONE = 0, NCW = 1 };
};
struct GENERAL {
  operator int() const { return 3; }
  enum { NONE = 0, BROADCAST = 1 };
};
struct CW_TRANS {
  operator int() const { return 4; }
};
struct NONZERO {
  operator int() const { return 5; }
};
struct SYS {
  operator int() const { return 6; }
  enum {
    END = 0,
    NOP = 1,
    TRWR = 2,
    SEND_MSG = 3,
    WAIT_MSG = 4,
    FORK = 5,
    JOIN = 6,
    EXIT = 7
  };
};
struct GATHER {
  operator int() const { return 7; }
  enum {
    FUNC_NONE = 0,
    FUNC_TRANS = 1 // NC Transpose or Matrix Transpose
  };
};
struct SCATTER {
  operator int() const { return 8; }
  enum { SCATTER_INPLACE = 0, SCATTER_ADD = 1 };
};
struct REVERSE {
  operator int() const { return 9; }
  enum { W_REVERSE = 0, H_REVERSE = 1, C_REVERSE = 2, N_REVERSE = 3 };
};
struct COMPRESS {
  operator int() const { return 10; }
  enum { NON_RANDOM_ACCESS = 0, RANDOM_ACCESS = 1 };
};
struct DECOMPRESS {
  operator int() const { return 11; }
  enum { NON_RANDOM_ACCESS = 0, RANDOM_ACCESS = 1 };
};
struct LOSSY_COMPRESS {
  operator int() const { return 12; }
  enum { NON_RANDOM_ACCESS = 0, RANDOM_ACCESS = 1 };
};
struct LOSSY_DECOMPRESS {
  operator int() const { return 13; }
  enum { NON_RANDOM_ACCESS = 0, RANDOM_ACCESS = 1 };
};
struct RANDOM_MASK {
  operator int() const { return 15; }
  enum { NON_RANDOM_ACCESS = 0, RANDOM_ACCESS = 1 };
};
struct TRANSFER {
  operator int() const { return 16; }
};
} // namespace dmaType
} // namespace bm1690
} // namespace tpu_mlir
