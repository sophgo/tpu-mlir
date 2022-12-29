//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_COMPRESS_UTIL_H_
#define TPU_COMPRESS_UTIL_H_

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Support/Module.h"

// // TPU DMA engine supports compression and decompression during data transfer
// // between global memory and local memory.
// // Compiler compresses the tiled weight and update the weight file in advanced.
// // The backend generates the DMA command which transfers the compressed weight.
// // The DMA engine decompresses the weight before writing to the local memory.

namespace tpu_mlir {

struct CompressCommandInfo {
  uint8_t signedness;
  uint8_t is_bfloat16;
  uint8_t bias0;
  uint8_t bias1;
  uint8_t zero_guard_en;
};

//
//  dataType
//    0: 8bit
//    1: 16bit
int getCompressedDataSize(int unCompressedDatasize, int dataType);

void getCompressParameter(
    const uint8_t *ibuf, size_t isz, uint8_t signedness, uint8_t isBfloat16,
    CompressCommandInfo *cmd_info);

void compressInt8Data(
    const uint8_t *ibuf, int isz, uint8_t *obuf, int *osz,
    CompressCommandInfo *cmd_info);

void compressBf16Data(
    const uint8_t *ibuf, int isz, uint8_t *obuf, int *osz,
    CompressCommandInfo *cmd_info);

// unit test
void testCompress(void);

class WeightCompresser {
public:
  WeightCompresser(Operation* op, bool do_compress);
  ~WeightCompresser();

public:
  bool done = false;
  std::vector<uint8_t> old_data;
  std::vector<uint8_t> new_data;

protected:
  Operation* op;
  RankedTensorType rtype;
};
} // namespace tpu_mlir

#endif // TPU_COMPRESS_UTIL_H_
