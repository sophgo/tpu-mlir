//===- TopTFLiteToTpu.h - Convert TOP TFLite to TPU dialect --*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_CONVERSION_TOPTFLITETOTPU_H
#define SOPHGO_CONVERSION_TOPTFLITETOTPU_H

#include "mlir/Pass/Pass.h"

namespace sophgo {

void populateTopToTpuConversionPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createLowerTopTFLitePass();

} // namespace sophgo

#endif // SOPHGO_CONVERSION_TOPTFLITETOTPU_H
