#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace sophgo {

// =======================
// interfece for moduleop
// =======================
// TODO(pengchao.hu): define state by enum in td
mlir::ModuleOp getModuleOp(mlir::Operation*op);
llvm::StringRef getMlirWeightFile(mlir::ModuleOp module);
void setMlirWeightFile(mlir::ModuleOp module, llvm::StringRef weight_file);
llvm::StringRef getMlirState(mlir::ModuleOp module);
void setMlirState(mlir::ModuleOp module, llvm::StringRef state);
llvm::StringRef getMlirChip(mlir::ModuleOp module);
void setMlirChip(mlir::ModuleOp module, llvm::StringRef chip);

// =======================
// interfece for quantization
// =======================
void get_scale_and_shift(float scale_f, int &scale, int &shift,
                         int bitwidth = 32);
void get_scale_and_shift_positive(float scale_f, int &scale, int &shift,
                                  int bitwidth = 32);
void get_scale_and_shift_positive_maxshift(float scale_f, int &scale,
                                           int &shift, int bitwidth,
                                           int max_shift = 8);
template <typename Dtype> float findMaxabs(const Dtype *pSrcData, int len);
template <typename Dtype>
void findMinMax(const Dtype *pSrcData, int len, float *minVal, float *maxVal);
int calRightShiftNum(float fmax, double thBottom, double thTop, int numBits);
template <typename T> void func_abs(int n, T *src, T *dst);
template <typename T> void func_log(int n, T *src, T *dst);
int calRightShiftNumUseCblas(float fmax, double thBottom, double thTop,
                             int numBits);
float func_log2(double dataInput);
float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
void quantizeToInt8(const float *pSrc, int8_t *pDst, int len, float scale);
} // namespace sophgo
