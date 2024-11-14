//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct ConvertPriorBoxPattern : public OpRewriterPatternEx<PriorBoxOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConvertPriorBoxPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PriorBoxOp>(context, "ConvertPriorBoxPattern") {}

  LogicalResult matchAndRewriteImpl(PriorBoxOp op,
                                    PatternRewriter &rewriter) const override {
    auto result = op.getResult();

    auto shape = module::getShape(result);
    auto size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    auto resultT = std::vector<float>(size);
    auto min_size = module::getF64Array(op.getMinSize());
    auto max_size = module::getF64Array(op.getMaxSize());
    auto aspect_ratios = module::getF64Array(op.getAspectRatios());
    auto variance = module::getF64Array(op.getVariance());

    bool clip = op.getClip();
    bool use_default_aspect_ratio = op.getUseDefaultAspectRatio();
    double offset = op.getOffset().convertToDouble();
    double step_w = op.getStepW().convertToDouble();
    double step_h = op.getStepH().convertToDouble();
    int64_t num_priors = op.getNumPriors();
    int64_t img_height = op.getImgH();
    int64_t img_width = op.getImgW();

    if (max_size->size() > 0) {
      assert(max_size->size() == min_size->size() &&
             "num of max_size should the same with min_size");
    }

    // Must and only provide 4 variance.
    assert(variance->size() == 4 && "variance size must be 4");

    auto shape0 = module::getShape(op.getInputs()[0]);
    auto shape1 = module::getShape(op.getInputs()[1]);

    assert(shape1.size() == 4 && shape0.size() == 4);
    const int64_t layer_width = shape0[3];
    const int64_t layer_height = shape0[2];

    if (img_height == 0 || img_width == 0) {
      img_height = shape1[2];
      img_width = shape1[3];
    }

    if (std::abs(step_w) < 1e-6 || std::abs(step_h) < 1e-6) {
      step_w = static_cast<double>(img_width) / layer_width;
      step_h = static_cast<double>(img_height) / layer_height;
    }

    std::vector<float> top_data(size);
    int64_t dim = layer_height * layer_width * num_priors * 4;
    int idx = 0;
    for (int64_t h = 0; h < layer_height; ++h) {
      for (int64_t w = 0; w < layer_width; ++w) {
        double center_x = (w + offset) * step_w;
        double center_y = (h + offset) * step_h;
        double box_width, box_height;
        for (size_t s = 0; s < min_size->size(); ++s) {
          int min_size_ = (int)min_size->at(s);
          if (use_default_aspect_ratio) {
            // first prior: aspect_ratio = 1, size = min_size
            box_width = box_height = min_size_;
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }

          if (max_size->size() > 0) {
            int max_size_ = max_size->at(s);
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }

          // rest of priors
          for (size_t r = 0; r < aspect_ratios->size(); ++r) {
            double ar = aspect_ratios->at(r);
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size_ * sqrt(ar);
            box_height = min_size_ / sqrt(ar);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
        }
      }
    }
    // clip the prior's coordidate such that it is within [0, 1]
    if (clip) {
      for (int d = 0; d < dim; ++d) {
        top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
      }
    }

    auto o_s = module::getShape(op.getOutput());

    // set the variance.
    // top_data += (o_s[2]);

    int count = 0;
    for (int64_t h = 0; h < layer_height; ++h) {
      for (int64_t w = 0; w < layer_width; ++w) {
        for (int64_t i = 0; i < num_priors; ++i) {
          for (auto &v : *variance) {
            top_data[(o_s[2]) + count] = v;
            ++count;
          }
        }
      }
    }
    auto op_name = module::getName(op.getOutput());
    auto weight_name = op_name.str() + "loadweight";
    auto weight_type = RankedTensorType::get(shape, rewriter.getF32Type());
    auto weight_operand =
        WeightOp::create(op, weight_name, top_data, weight_type);
    rewriter.replaceOp(op, weight_operand);
    return success();
  }
};

void PriorBoxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<ConvertPriorBoxPattern>(context);
}
