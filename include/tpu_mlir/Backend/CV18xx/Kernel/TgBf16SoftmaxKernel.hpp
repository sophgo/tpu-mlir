//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {
class TgSoftmaxKernel {
public:
  TgSoftmaxKernel() {}

  void init(uint32_t layer_id,
            gaddr_t ga_input,
            gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
            gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
            gaddr_t ga_output,
            int64_t* shape, int axis, int dimension, bool do_log);
  void selectTilePolicy();
  void schedule();

protected:
  typedef struct {
    int pos_h;
    int h;
  } tiling_t;
  enum SoftmaxMode {Softmax2D, Softmax4D};
  enum OperateMode {Sub, Mul};

  /**
	 * @brief Select softmax mode
	 */
  void selectSoftmaxMode(int64_t* shape);

  /**
	 * @brief Fill one to dram as golden
	 */
  void fillOneAsGolden();

  /**
	 * @brief Transform matrix to tensor
   * @param tensor output tensor
   * @param matrix input matrix
	 */
  void matrixToTensor(cvk_tl_t *tensor, const cvk_ml_t &matrix);

  /**
	 * @brief Split height of softmax2D which parallel inner size to NPU_NUM
	 */
  unsigned int doSplitHeightBf16softmax2DParallelInnerSize();

  /**
	 * @brief Split height of softmax2D which parallel outer size to NPU_NUM
	 */
  unsigned int doSplitHeightBf16softmax2DParallelOuterSize();

  /**
	 * @brief Do softmax2D which inner size is too large to handle in normal case. For example, shape(1, 16002) and shape(400, 16002)
	 */
  void softmaxLargeSizeHandler();

  /**
	 * @brief Do softmax2D which parallel inner size to NPU_NUM
	 */
  void bf16_softmax_kernel_2d_parallel_inner_size();

  /**
	 * @brief Do softmax2D which parallel outer size to NPU_NUM
	 */
  void bf16_softmax_kernel_2d_parallel_outer_size();

  /**
	 * @brief Split height of softmax4D
	 */
  int doSplitHeightBf16softmax4D();

  /**
	 * @brief Split width of softmax4D
	 */
  int doSplitWidthBf16softmax4D();

  /**
	 * @brief Do softmax 4D
	 */
  void bf16_softmax_kernel_4d();

  /**
	 * @brief Do softmax 2D
	 */
  void bf16_softmax_kernel_2d();

  /**
	 * @brief Get exponential value
	 * @param tl_in Input tensor
   * @param tl_out Output tensor
   * @param tl_work Working space
	 */
  void exponential(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work);

  /**
	 * @brief Get reciprocal value
	 * @param tl_in Input tensor
   * @param tl_out Output tensor
   * @param tl_work Working space
	 */
  void reciprocal(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work);

  /**
	 * @brief Get log value
	 * @param tl_in Input tensor
   * @param tl_out Output tensor
   * @param tl_work Working space
	 */
  void log(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work);

  /**
	 * @brief Broadcast one data to all lane in the same address
	 * @param tl_in input broadcasted data address
   * @param tl_out output broadcasted data address
	 */
  void broadcast_one_data_to_all_lane(cvk_tl_t *tl_in, cvk_tl_t *tl_out);

  /**
	 * @brief Every input sub one specific data
	 * @param tl_in_out Input/output tensor
   * @param tl_operand operand tensor
   * @param operate operate mode
   * @param isParallelInLane is outerSize parallel in lane
	 */
  void every_input_operate_one_specific_data(cvk_tl_t *tl_in_out, cvk_tl_t *tl_operand, OperateMode operate, bool isParallelInLane);

  /**
	 * @brief Initialize and load table
	 */
  void init_table();

  /**
	 * @brief Free table
	 */
  void free_table();

  /**
	 * @brief Get max value in tl_in and store to tl_out
	 */
  void max_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out);

  /**
	 * @brief Accumulate data in tl_in per land and store to tl_out
	 */
  void accumulate_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out);

protected:
  gaddr_t ga_input;
  gaddr_t ga_exponential_table_data_lut;
  gaddr_t ga_exponential_slope_table_data_lut;
  gaddr_t ga_reciprocal_table_data_lut;
  gaddr_t ga_reciprocal_table_mantissa_data_lut;
  gaddr_t ga_output;
  int axis;
  bool do_log;
  int dimension;
  int outer_size;
  int inner_size;
  cvk_fmt_t fmt;
  int fmt_size;
  SoftmaxMode functionMode;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t layer_id;

  // for lmem addr alloc
  cvk_tl_shape_t table_shape;
  cvk_tl_t *tl_exponential_table_answer;
  cvk_tl_t *tl_exponential_table_answer_slope;
  cvk_tl_t *tl_reciprocal_table_answer;
  cvk_tl_t *tl_reciprocal_mantissa_table_answer;
};
}
}
