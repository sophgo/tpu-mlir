//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/Debug.h>
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
//
// Dimension indices for 2D tensor.
//
struct NCHW {
  enum dim { N = 0, C = 1, H = 2, W = 3 };
};

//
// dimension indices for 3D tensor.
//
struct NGCHW {
  enum dim { N = 0, G = 1, C = 2, H = 3, W = 4 };
};

struct NCDHW {
  enum dim { N = 0, C = 1, D = 2, H = 3, W = 4 };
};

class MemoryDescriptor {
public:
  MemoryDescriptor(){};

  void setShapes(std::vector<uint32_t> shapes) { shapes_ = shapes; }

  std::vector<uint32_t> getShapes() { return shapes_; }

  void setDataFormat(cvk_fmt_t fmt) { fmt_ = fmt; }

  cvk_fmt_t getDataFormat() { return fmt_; }

  void setStrides(std::vector<uint32_t> strides) { strides_ = strides; }

  std::vector<uint32_t> getStrides() {
    if (strides_.size()){
      return strides_;
    }
    return std::vector<uint32_t>();
  }

  void setAddress(uint64_t address) { address_ = address; }

  uint64_t getAddress() { return address_; }

  uint32_t getDataFormatSize() {
    switch (fmt_) {
    case CVK_FMT_F32:
      return 4;
    case CVK_FMT_BF16:
      return 2;
    default:
      return 1;
    }
  }

  void setLayerId(uint32_t layer_id) { layerId_ = layer_id; }

  uint32_t getLayerId() { return layerId_; }

  // Calculate offset using current positions
  uint64_t getCurrentOffset(std::vector<uint32_t> cur_poss) {
    assert(cur_poss.size() == shapes_.size() &&
           "Expect current positions and shapes have same dims");
    assert(strides_.size() == shapes_.size() &&
           "Expect stride and shapes have same dims");

    uint64_t offset = 0;
    for (uint32_t i = 0; i < cur_poss.size(); ++i) {
      offset += cur_poss[i] * strides_[i];
    }

    return offset;
  }

  // Expect physical shape, but it is very difficult in our system.
  std::vector<uint32_t> shapes_;

  // Default: int8 data type
  cvk_fmt_t fmt_ = {CVK_FMT_I8};

  // Physical layout
  // Not use logical shape to derive physical layout (e.g. do_ic_alignment)
  std::vector<uint32_t> strides_;

  uint64_t address_ = {0};
  uint32_t layerId_ = {0};
};

class LocalMemoryDescriptor : public MemoryDescriptor {
public:
  LocalMemoryDescriptor(std::vector<uint32_t> shapes, cvk_fmt_t fmt,
                        uint8_t eu_align) {
    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;
  }

  LocalMemoryDescriptor() {}

  ~LocalMemoryDescriptor() {
    // Kernel release resource in reverse order.
    assert(!cvk_tl_ && "Expect cvk freed");
  }

  std::vector<uint32_t> getStrides() {
    if (strides_.size())
      return strides_;

    assert(shapes_.size() == 5 && "Expect 5D tensor now");
    assert(shapes_[NGCHW::G] == 1 && "Expect 1 group");
    cvk_tl_shape_t tl_shapes = {shapes_[NGCHW::N], shapes_[NGCHW::C],
                                shapes_[NGCHW::H], shapes_[NGCHW::W]};
    cvk_tl_stride_t tl_strides =
        CV18xx::tl_default_stride(tl_shapes, fmt_, eu_align_);

    strides_ = {tl_strides.n, tl_strides.c, tl_strides.h, tl_strides.w};

    return strides_;
  }

  void initialize(std::vector<uint32_t> shapes, cvk_fmt_t fmt,
                  uint8_t eu_align) {
    // Group should not appear in local memory descriptor since H/W does not
    // support grouped convolution.
    // And we do not support conv3d yet.
    assert(shapes.size() == 5 && "Expect 5D tensor");
    assert(shapes_[NGCHW::G] == 1 && "Expect 1 group");

    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;

    cvk_tl_shape_t tl_shapes = {shapes_[NCHW::N], shapes_[NCHW::C],
                                shapes_[NCHW::H], shapes_[NCHW::W]};
    cvk_tl_stride_t tl_strides =
        CV18xx::tl_default_stride(tl_shapes, fmt_, eu_align_);

    strides_ = {tl_strides.n, tl_strides.c, tl_strides.h, tl_strides.w};
  }

  void setEuAlign(uint8_t eu_align) { eu_align_ = eu_align; }

  uint8_t getEuAlign() { return eu_align_; }

  void allocate(std::vector<uint32_t> shapes, cvk_fmt_t fmt, uint8_t eu_align) {
    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;

    allocate();
  }

  void allocate() {
    assert(!cvk_tl_ && "Expect no allocated before");
    assert(shapes_.size() == 5 && "Expect 5D tensor");

    cvk_tl_shape_t tl_shape = {shapes_[NGCHW::N],
                               shapes_[NGCHW::G] * shapes_[NGCHW::C],
                               shapes_[NGCHW::H], shapes_[NGCHW::W]};
    cvk_tl_ = CV18xx::lmem_alloc_tensor(tl_shape, fmt_, eu_align_);
    assert(cvk_tl_ && "Expect allocated");

    address_ = cvk_tl_->start_address;
    strides_ = {cvk_tl_->stride.n, cvk_tl_->stride.c, cvk_tl_->stride.h,
                cvk_tl_->stride.w};
  }

  // Return previously allocated kernel local memory information
  // DO NOT use it for tmda/tiu operation directly !
  // It is not always that each tile size equals to it.
  // And tdma load/store and tiu op may use different shape.
  cvk_tl_t *getAllocated() {
    assert(cvk_tl_ && "Expected allocated");
    return cvk_tl_;
  }

  void free() {
    if (cvk_tl_) {
      CV18xx::lmem_free_tensor(cvk_tl_);
      cvk_tl_ = nullptr;
    }
  }

  cvk_tl_shape_t getCvkShape() {
    assert(shapes_.size() && "Expect shape assigned");
    return {shapes_[NCHW::N], shapes_[NCHW::C], shapes_[NCHW::H],
            shapes_[NCHW::W]};
  }

  uint32_t getSizePerLane() {
    if (shapes_.size() && strides_.size()) {
      assert(shapes_.size() == strides_.size() &&
             "Expect shape and strid have same size");
      return shapes_[NCHW::N] * strides_[NCHW::N];
    }

    assert(shapes_.size() && "Expect shape assigned");

    return CV18xx::lmem_tensor_to_size(getCvkShape(), fmt_, eu_align_);
  }

private:
  uint8_t eu_align_ = {0};
  cvk_tl_t *cvk_tl_ = {nullptr};
};

class GlobalMemoryDescriptor : public MemoryDescriptor {
public:
  GlobalMemoryDescriptor(std::vector<uint32_t> shapes, cvk_fmt_t fmt) {
    shapes_ = shapes;
    fmt_ = fmt;

    setDefaultStrides();
  }

  void setDefaultStrides() {
    assert(((shapes_.size() == 4) || (shapes_.size() == 5)) &&
           "Expect 4D or 5D tensor");
    if (((shapes_.size() != 4) && (shapes_.size() != 5)))
      return;

    strides_.resize(shapes_.size());
    strides_[strides_.size() - 1] = getDataFormatSize();
    for (int i = (int)strides_.size() - 2; i >= 0; --i)
      strides_[i] = shapes_[i + 1] * strides_[i + 1];
  }

  std::vector<uint32_t> getStrides() {
    if (strides_.size())
      return strides_;

    setDefaultStrides();
    return strides_;
  }

private:
};

class Tdma {
public:
  Tdma(MemoryDescriptor *dst, MemoryDescriptor *src) : dst_(dst), src_(src) {}

  void transfer() {
    if (static_cast<GlobalMemoryDescriptor *>(dst_) &&
        static_cast<LocalMemoryDescriptor *>(src_))
      load();
  }

private:
  void load() {}

  MemoryDescriptor *dst_;
  MemoryDescriptor *src_;
};

class CmdDescriptor {
public:
  enum CmdTypeEnum {
    LoadBiasCmdType,
    LoadQuantCmdType,
    LoadInputCmdType,
    LoadWeightCmdType,
    LoadScaleLutTblCmdType,
    ComputCmdType,
    ComputeQuantCmdType,
    ComputeScaleLutCmdType,
    StoreOutputCmdType,
    ParallelCmdType
  };

  CmdDescriptor(CmdTypeEnum cmdType, bool parallelEnabled)
      : cmdType_(cmdType), parallelEnabled_(parallelEnabled) {}

  CmdDescriptor(CmdTypeEnum cmdType, uint32_t lmIndex) : cmdType_(cmdType) {
    lmIndexes_.push_back(lmIndex);
  }
  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
                uint32_t lmIndex)
      : cmdType_(cmdType), gmOutputPoss_(gmOutputPoss) {
    lmIndexes_.push_back(lmIndex);
  }

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
                uint32_t lmIndex, uint32_t icPos)
      : cmdType_(cmdType), gmOutputPoss_(gmOutputPoss) {
    lmIndexes_.push_back(lmIndex);
    icPos_ = icPos;
  }

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
                std::vector<uint32_t> lmIndexes)
      : cmdType_(cmdType), gmOutputPoss_(gmOutputPoss), lmIndexes_(lmIndexes) {}

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
                std::vector<uint32_t> lmIndexes, uint32_t icPos)
      : cmdType_(cmdType), gmOutputPoss_(gmOutputPoss), lmIndexes_(lmIndexes),
        icPos_(icPos) {}

  static std::string getCmdTypeStr(CmdTypeEnum cmdType);

  CmdTypeEnum getCmdType() { return cmdType_; }

  std::vector<uint32_t> getGmOutputPoss() { return gmOutputPoss_; }

  std::vector<uint32_t> getLmIndexes() { return lmIndexes_; }

  bool isParallelEnabled() { return parallelEnabled_; }

  void setIntraCmdParalEnabled(bool enabled) { intraCmdParalEnabled_ = true; }

  bool isIntraCmdParalEnabled() { return intraCmdParalEnabled_; }

  uint32_t getIcPos() { return icPos_; }

private:
  CmdTypeEnum cmdType_;
  std::vector<uint32_t> gmOutputPoss_;
  std::vector<uint32_t> lmIndexes_;
  bool parallelEnabled_ = {false};
  bool intraCmdParalEnabled_ = {false};
  uint32_t icPos_ = {0};
};

std::string CmdDescriptor::getCmdTypeStr(CmdTypeEnum cmdType) {
  switch (cmdType) {
  case LoadBiasCmdType:
    return "LoadBias";

  case LoadInputCmdType:
    return "LoadInput";

  case LoadWeightCmdType:
    return "LoadWeight";

  case ComputCmdType:
    return "Compute";

  case StoreOutputCmdType:
    return "StoreOutput";

  case ParallelCmdType:
    return "Parallel";

  default:
    assert(0 && "Unexpected cmd type");
  }

  return " ";
}

// Command sequence pattern for 1822 intra command parallism:
//   cmd            local memory access state
//   Load bias      (Write)
//   Load input     (Write)
//   Load weight    (Write)
//   TPU compute    (Output Write)
//   Store output   (ReadAfterWrite)
//
//
// Case 1: reuse activation
//                     bias      weight      input      output
//   LD input0                                W|
//   LD bias0           W|
//   LD weight0                    W|                              (O)
//   TIU0             RAW|       RAW|       RAW|          W|       (O)
//   ST output0                                         RAW|       (O)
//   LD bias1            |R
//   LD weight1                     |R
//   TIU1                |RAW       |RAW    RAR|           |W      (X)
//   ST output1                                            |RAW
//
//
// Case 2: reuse weight
//                     bias      weight      input      output
//   LD bias0          W|
//   LD weight0                    W|                              => swap
//   LD input0                                W|                   => swap
//   TIU0            RAW|        RAW|       RAW|           W|
//   ST output0                                          RAW|
//   LD input1                                 |W
//   TIU1            RAR|        RAR|          |RAW         |W     (X)
//   ST output1                                             |RAW
//
class IntraCmdParallelAnalysis {
public:
  IntraCmdParallelAnalysis(
      std::vector<std::unique_ptr<CmdDescriptor>> &cmdQueue)
      : cmdQueue(cmdQueue) {

    // Double buffer
    for (uint32_t i = 0; i < 2; ++i) {
      lmBiasAccessStates_.push_back(UnknownState);
      lmQuantAccessStates_.push_back(UnknownState);
      lmWeightAccessStates_.push_back(UnknownState);
      lmInputAccessStates_.push_back(UnknownState);
      lmOutputAccessStates_.push_back(UnknownState);
    }

    assignLmAccessState();
  }

  void assignLmAccessState();

  // Simplified data dependency process based on hand-crafted double-buffer
  // assignment.
  enum AccessEvent { WriteEvent, ReadEvent };
  enum AccessState {
    UnknownState,
    WriteState,
    ReadAfterWriteState,
    ReadAfterReadState,
    WriteAfterWriteState,
  };

  struct CmdLmState {
    CmdLmState(CmdDescriptor::CmdTypeEnum cmdType,
               std::vector<AccessState> biass, std::vector<AccessState> quants,
               std::vector<AccessState> weights,
               std::vector<AccessState> inputs,
               std::vector<AccessState> outputs)
        : cmdType_(cmdType), biass_(biass), quants_(quants), weights_(weights),
          inputs_(inputs), outputs_(outputs) {}

    CmdDescriptor::CmdTypeEnum cmdType_;
    std::vector<AccessState> biass_;
    std::vector<AccessState> quants_;
    std::vector<AccessState> weights_;
    std::vector<AccessState> inputs_;
    std::vector<AccessState> outputs_;
    bool isIntraCmdParal_ = {false};
  };

  static std::string getAccessEventStr(AccessEvent event);
  static std::string getAccessStateStr(AccessState state);

  void receiveAccessEvent(AccessState *state, AccessEvent event);

  uint32_t reverseSearchBiasOrWeight(AccessState state, uint32_t lmIndex,
                                     uint32_t endQueueIndex);
  uint32_t searchStoreOutput(AccessState state, uint32_t lmIndex,
                             uint32_t startQueueIndex);

  bool isIntrCmdParalTiu(uint32_t index);
  bool isIntrCmdParalLoadWeight(uint32_t index, uint32_t lmWeightIndex);
  bool isIntrCmdParalStoreOutput(uint32_t index, uint32_t lmOutputIndex);
  void tryEnableIntraCmdParal(uint32_t index);

  void analyze();

  void dumpStates();

private:
  const std::vector<std::unique_ptr<CmdDescriptor>> &cmdQueue;

  std::vector<AccessState> lmBiasAccessStates_;
  std::vector<AccessState> lmQuantAccessStates_;
  std::vector<AccessState> lmWeightAccessStates_;
  std::vector<AccessState> lmInputAccessStates_;
  std::vector<AccessState> lmOutputAccessStates_;

  // Record local memory status of each command
  std::vector<std::unique_ptr<CmdLmState>> cmdLmStates_;
};

// Manual CMODEL Debug:
//   Backend:
//     Assign layer id, output position.
//     Record input, output, weight bias information.
//     Change layer id to ((1 << 15) | layer_id) in TIU command buffer.
//
//   CMODEL:
//     Detect altered layer_id.
//     Convolution store input, weight, bias and output.
//
//   HOST:
//     Extract input/output/weight from npz used in mlir.
//     Compare data from host and CMODEL.
//
//   E.g.
//     Output positions:
//       [ig=0][oc_pos=736][n_pos=6][oh_pos=31][ow_pos=0][ic_pos=0]
//
//     MLIR:
//       tl_lw_memopt_func_tg_Conv2d_int8.mlir
//
//     TPU output:
//       data1 = np.load('WZC-0_cmdbuf_out_bs8.npz')
//       data11 = data1['6fac0227e623ad2e7a08b330d5a6ffe3']
//
//     Tiled TPU output::
//       oc_pos=736, oc_step=32, oh_pos=31, oh_step=25,  (1, 32, 56, 56)
//       data12 = data11[6:7, 736:736+32, 31:31+25, :]
//       np.savetxt('tpu_conv_oc_pos_736_oh_pos_31.txt',
//                  np.reshape(data12, (np.size(data12), 1)))
//
//     CPU output:
//       data2 = np.load('WZC-0_tensor_all_int8.npz')
//       data21 = data2['6fac0227e623ad2e7a08b330d5a6ffe3']
//
//     Tiled CPU output:
//       oc_pos=736, oc_step=32, oh_pos=31, oh_step=25,  (1, 32, 56, 56)
//       data22 = data21[6:7, 736:736+32, 31:31+25, :]
//       np.savetxt('cpu_conv_oc_pos_736_oh_pos_31.txt',
//                  np.reshape(data22, (np.size(data22), 1)))
//
//     Weight:
//       Weight (2048, 256, 1, 1)
//       data3 = np.load('WZC-0_4_558b2e062f9d.npz')
//       weight = data3['6fac0227e623ad2e7a08b330d5a6ffe3_0_quant_lowered']
//
//     Tiled weight:
//       Weight oc_pos=736, oc_step=32, (32, 256, 1, 1)
//       weight1 = weight[736:736+32, :, :, :]
//       np.savetxt('weight_pos_736_step_32.txt',
//                  np.reshape(weight1, (np.size(weight1), 1)))
//
//     Bias:
//       bias = data3['6fac0227e623ad2e7a08b330d5a6ffe3_1_quant_pack']
//
//     Tiled bias:
//       oc_pos=736, oc_step=32, (32, 1, 9)
//       bias1 = bias[736:736+32,:,:]
//       np.savetxt('bias_pos_736_step_32.txt',
//                  np.reshape(bias1, (np.size(bias1), 1)))
//
struct CModelDebug {
  bool enabled_;
  bool found_;
  uint16_t layerId_;

  struct GmInfo {
    uint64_t addr;
    uint64_t addrOffset;
    std::vector<uint32_t> shapes;
    std::vector<uint32_t> poss;
  };

  void assignOutput(uint32_t layerId, std::vector<uint32_t> poss) {
    enabled_ = true;
    found_ = false;
    layerId_ = layerId;
    output_.poss = poss;
  }

  bool isOutputMatched(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
                       bool isWeightOrBias = false) {
    if (!enabled_ || (layerId_ != layerId))
      return false;

    if (isWeightOrBias) {
      assert(!gmOutputPoss[NGCHW::N] && !gmOutputPoss[NGCHW::H] &&
             !gmOutputPoss[NGCHW::W]);
      gmOutputPoss[NGCHW::N] = output_.poss[NGCHW::N];
      gmOutputPoss[NGCHW::H] = output_.poss[NGCHW::H];
      gmOutputPoss[NGCHW::W] = output_.poss[NGCHW::W];
    }

    if ((output_.poss[NGCHW::N] == gmOutputPoss[NGCHW::N]) &&
        (output_.poss[NGCHW::G] == gmOutputPoss[NGCHW::G]) &&
        (output_.poss[NGCHW::C] == gmOutputPoss[NGCHW::C]) &&
        (output_.poss[NGCHW::H] == gmOutputPoss[NGCHW::H]) &&
        (output_.poss[NGCHW::W] == gmOutputPoss[NGCHW::W]))
      return true;
    return false;
  }

  void updateLayerId(uint32_t &layerId, std::vector<uint32_t> gmOutputPoss) {

    if (isOutputMatched(layerId, gmOutputPoss)) {
      layerId = (1 << 15) | layerId;
    }
  }

  void recordGmInfo(GmInfo &entity, uint64_t addr, uint64_t addrOffset,
                    std::vector<uint32_t> poss, std::vector<uint32_t> shapes) {
    entity.addr = addr;
    entity.addrOffset = addrOffset;
    entity.poss = poss;
    entity.shapes = shapes;
  }

  void recordOutput(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
                    uint64_t addr, uint64_t addrOffset,
                    std::vector<uint32_t> gmOutputShapes) {
    if (isOutputMatched(layerId, gmOutputPoss)) {
      recordGmInfo(output_, addr, addrOffset, gmOutputPoss, gmOutputShapes);
      found_ = true;
    }
  }

  void recordInput(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
                   uint64_t addr, uint64_t addrOffset,
                   std::vector<uint32_t> gmInputPoss,
                   std::vector<uint32_t> gmInputShapes,
                   bool ignoreOutputChannel) {

    if (!enabled_)
      return;

    if (ignoreOutputChannel) {
      assert(!gmOutputPoss[NGCHW::C]);
      gmOutputPoss[NGCHW::C] = output_.poss[NGCHW::C];
    }

    if (isOutputMatched(layerId, gmOutputPoss))
      recordGmInfo(input_, addr, addrOffset, gmInputPoss, gmInputShapes);
  }

  void recordWeight(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
                    uint64_t addr, uint64_t addrOffset,
                    std::vector<uint32_t> gmWeightPoss,
                    std::vector<uint32_t> gmWeightShapes) {
    if (isOutputMatched(layerId, gmOutputPoss, true))
      recordGmInfo(weight_, addr, addrOffset, gmWeightPoss, gmWeightShapes);
  }

  void recordBias(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
                  uint64_t addr, uint64_t addrOffset,
                  std::vector<uint32_t> gmBiasPoss,
                  std::vector<uint32_t> gmBiasShapes) {
    if (isOutputMatched(layerId, gmOutputPoss, true))
      recordGmInfo(bias_, addr, addrOffset, gmBiasPoss, gmBiasShapes);
  }

  void dumpDims(std::vector<uint32_t> &dims);

  void dump();

  GmInfo output_;
  GmInfo input_;
  GmInfo weight_;
  GmInfo bias_;
};

struct Conv_ARGS {
  gaddr_t ga_ifmap;
  gaddr_t ga_ofmap;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  gaddr_t ga_scale;
  gaddr_t ga_zeropoint;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int groups;
  int output_c;
  uint16_t kh;
  uint16_t kw;
  uint16_t dilation_h;
  uint16_t dilation_w;
  uint8_t pad_top;
  uint8_t pad_bottom;
  uint8_t pad_left;
  uint8_t pad_right;
  uint8_t insert_h;
  uint8_t insert_w;
  uint8_t stride_h;
  uint8_t stride_w;
  bool do_bias;
  bool do_activation;
  bool do_quant;
  float *activation_arg;
  int activation_gt_scale;
  int activation_gt_rshift;
  int activation_le_scale; // slope; TODO
  int activation_le_rshift;
  int right_shift_width;
  bool do_chl_quan;
  uint32_t layer_id;
  bool do_ic_alignment;
  bool fused_conv_relu;
  bool do_leaky_relu;
  bool do_load_cmpr_wgt;
  cvk_fmt_t input_fmt;
  cvk_fmt_t output_fmt;
  cvk_fmt_t tiu_fmt;
  uint8_t gm_input_region;
  uint8_t gm_output_region;
  uint8_t gm_activation_region;
  uint8_t gm_weight_region;
  bool ps32_output;
  int pad_value;
  bool do_scale_lut;
  gaddr_t ga_scale_lut;
  std::vector<uint8_t> *filter;
  std::vector<uint8_t> *new_filter;
};

typedef struct {
  int n;
  int oc;
  int ic;
  int h;
  int w;
  uint32_t n_step;
  uint32_t oc_step;
  uint32_t oh_step;
  uint32_t ow_step;
  uint32_t ih_step;
  uint32_t iw_step;
  uint32_t ic_step;
  uint32_t total_needed;
  bool favor_dma;
} TileInfo;

//
// We use the local memory to determine the tiled size in both global and local
// memory.
// Then we split the output in global memory and use it to derive:
//   1. tiled ouput size, position in global memory for tmda load
//   2. tiled output size, position in local memory for tdma load
//   3. tiled input size, position in global memory for tdma load
//   4. tiled input size, position in local memory for tdma load
//   5. tiled weight size, position in global memory for tdma load
//   6. tiled weight size, position in local memory for tdma load
//   5. tiled output size, position in local for tpu computation
//   6. tiled input size, position in local for tpu computation
//   7. tiled ouput size, position in global memory for tmda store
//   8. tiled output size, position in local memory for tdma store
//
//  It is really painful that shape/stride for tdma load, tpu compute, tdma
//  store are not always the same.
//
// 1. Double convolution for odd input channels:
//  input channel:        3
//  weight:               4 (offline modified)
//
//  ifmap lmem alloation: 4
//  ifmap tdma load:      3
//  ifmap tiu:            4
//
//
// 2. Align width for tdma efficiency:
//  input width:            28
//  kernel stride:          2
//
//  ifmap lmem allocation: 28
//  ifmap tdma load:       28
//  ifmap tiu:             27
//
class Conv {
public:
  Conv() {
    memset(&args, 0, sizeof(args));
    memset(&tile_info, 0, sizeof(tile_info));
    use_double_buffer = false;
  }

  bool checkDmaPolicy(TileInfo &tileInfo);
  bool determineTileSize(bool useDoubleBuffer, bool favor_dma);
  bool determinePs32TileSize(bool useDoubleBuffer);
  bool determineDwTileSize(bool useDoubleBuffer, bool favor_dma);

  void convReuseWeight();
  void convReuseActivation();
  void dwConv();

  void dwConv_pass();

  bool canNoTile();
  void convNoTile();

  void convNaive();

  void initializeGlobalMemInput();
  void initializeGlobalMemOutput();
  void initializeGlobalMemWeight();
  void initializeGlobalBias();
  void initializeGlobalQuant();
  void initializeGlobalScaleLut();
  void initializeGlobalMem();

  void initializeFusedActivation();
  void initializeTile();
  void determineTilePolicy();
  void doConvByTilePolicy();

  uint32_t getElementTypeSize(cvk_fmt_t fmt);

  void allocateTiledLocalMem(
      std::vector<std::unique_ptr<LocalMemoryDescriptor>> &lmDescs,
      uint32_t count, std::vector<uint32_t> shapes, uint32_t eu_align);
  void allocateLocalMemOfInput();
  void deallocateLocalMemOfInput();
  void allocateLocalMemOfOutput();
  void deallocateLocalMemOfOutput();
  void allocateLocalMemOfWeight();
  void deallocateLocalMemOfWeight();
  void allocateLocalMemOfBias();
  void deallocateLocalMemOfBias();
  void allocateLocalMemOfFusedActivation();
  void deallocateLocalMemOfFusedActivation();
  void allocateLocalMemOfPreProcess();
  void deallocateLocalMemOfPreProcess();
  void allocateLocalMemOfQuant();
  void deallocateLocalMemOfQuant();
  void allocateAllLocalMem();
  void deallocateAllLocalMem();

  std::vector<uint32_t> getTiledShapesForLmAllocationOfInput();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfOuput();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfWeight();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfBias();
  uint32_t getTiledEuAlignForLmAllocationOfInput();
  uint32_t getTiledEuAlignForLmAllocationOfOutput();
  uint32_t getTiledEuAlignForLmAllocationOfWeight();
  uint32_t getTiledEuAlignForLmAllocationOfBias();

  std::vector<uint32_t>
  getTiledGmShapesOfWeightForTdmaLoad(std::vector<uint32_t> gmOutputPoss,
                                      uint32_t icPos);
  std::vector<uint32_t>
  getTiledLmShapesOfWeightForTiu(std::vector<uint32_t> gmOutputPoss,
                                 uint32_t icPos);

  void getTiledGmPossAndShapesOfInputForTiu(
      std::vector<uint32_t> gmOutputPoss,
      std::vector<uint32_t> gmOutputPossShapes,
      std::vector<uint32_t> &cur_gm_input_poss,
      std::vector<uint32_t> &cur_gm_input_shapes,
      std::vector<uint32_t> &cur_gm_input_paddings, uint32_t ic_pos);

  std::vector<uint32_t>
  getTiledGmShapesOfBiasForTdmaLoad(std::vector<uint32_t> gmOutputPoss);
  std::vector<uint32_t>
  getTiledLmShapesOfBiasForTiu(std::vector<uint32_t> gmOutputPoss);

  std::vector<uint32_t>
  getTiledGmShapesOfQuantForTdmaLoad(std::vector<uint32_t> gmOutputPoss);
  std::vector<uint32_t>
  getTiledLmShapesOfQuantForTiu(std::vector<uint32_t> gmOutputPoss);

  std::vector<uint32_t>
  getTiledGmShapesOfOutputForTiu(std::vector<uint32_t> gmOutputPoss);

  void fillConstantLmInput(cvk_tl_t *lmLoad,
                           std::vector<uint32_t> &cur_gm_input_paddings);
  void
  adjustComputeForPadOnlyInput(cvk_tl_t *lmInput,
                               std::vector<uint32_t> &cur_gm_input_paddings);
  void adjustComputeForPs32Output(cvk_tl_t *lmOutput);
  void adjustStoreForPs32Output(cvk_tl_t *lmOutput, cvk_tg_t *gmOutput,
                                uint64_t ga_offset);
  void loadBias(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                uint32_t cmdQueueIndex);
  void loadQuant(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                 uint32_t cmdQueueIndex);
  void loadWeight(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                  uint32_t cmdQueueIndex, uint32_t icPos = 0);
  void loadInput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                 uint32_t cmdQueueIndex, uint32_t ic_pos = 0);
  void loadScaleLutTable(uint32_t lmIndex, uint32_t cmdQueueIndex);
  void computeConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input, cvk_tl_t *tl_weight,
                   cvk_tl_t *tl_bias,
                   std::vector<uint32_t> &cur_gm_input_paddings,
                   uint8_t cmdPreExeMode, uint32_t icPos = 0);
  void computePerTensorConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                            cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                            std::vector<uint32_t> &cur_gm_input_paddings,
                            uint8_t cmdPreExeMode, uint32_t icPos = 0);
  void computeDwConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                     cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                     std::vector<uint32_t> &cur_gm_input_paddings,
                     uint8_t cmdPreExeMode, uint32_t icPos = 0);
  void computePerTensorDwConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                              cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                              std::vector<uint32_t> &cur_gm_input_paddings,
                              uint8_t cmdPreExeMode, uint32_t icPos = 0);
  void computeLeakyRelu(cvk_tl_t *tl_output);
  void compute(std::vector<uint32_t> gmOutputPoss,
               std::vector<uint32_t> lmIndexes, uint32_t cmdQueueIndex,
               uint32_t icPos = 0);
  void computeScaleLut(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                       uint32_t cmdQueueIndex, uint32_t icPos = 0);
  void computeQuant(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                    uint32_t cmdQueueIndex, uint32_t icPos = 0);
  void storeOutput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                   uint32_t cmdQueueIndex);

  uint32_t getPs32Mode(uint32_t icPos);
  bool getReluAllowed(uint32_t icPos);
  bool getBiasAllowed(uint32_t icPos);
  bool getRshiftAllowed(uint32_t icPos);

  uint8_t getTdmaLoadWeightIntraCmdParal(uint32_t cmdQueueIndex);
  uint8_t getTdmaStoreOutputIntraCmdParal(uint32_t cmdQueueIndex);
  uint8_t getTiuCmdPreExeMode(uint32_t cmdQueueIndex, uint32_t icPos);

  bool isDwConv();
  bool isConvPs32();

  void enqueueLoadInputCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueLoadInputCmd(std::vector<uint32_t> poss, uint32_t index,
                           uint32_t icPos);
  void enqueueStoreOutputCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueLoadBiasCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueLoadQuantCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueLoadWeightCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueLoadWeightCmd(std::vector<uint32_t> poss, uint32_t index,
                            uint32_t icPos);
  void enqueueLoadScaleLutTblCmd();
  void enqueueComputeCmd(std::vector<uint32_t> poss,
                         std::vector<uint32_t> indexes);
  void enqueueComputeCmd(std::vector<uint32_t> poss,
                         std::vector<uint32_t> indexes, uint32_t icPos);
  void enqueueComputeScaleLutCmd(std::vector<uint32_t> poss, uint32_t index);
  void enqueueComputeScaleLutCmd(std::vector<uint32_t> poss, uint32_t index,
                                 uint32_t icPos);
  void enqueueComputeQuantCmd(std::vector<uint32_t> poss, uint32_t index,
                              uint32_t icPos = 0);
  void enqueueDisParallelCmd();
  void enqueueEnParallelCmd();

  void generateCmd();

  bool compressWeight();

  // CMODEL Debug
  void configCModelDebug();

  uint32_t batch_size() { return args.input_n; }

  uint32_t input_height() { return args.input_h; }

  uint32_t input_width() { return args.input_w; }

  uint32_t insert_height() { return args.insert_h; }

  uint32_t insert_width() { return args.insert_w; }

  uint32_t inserted_input_height() {
    return args.input_h + (args.input_h - 1) * args.insert_h;
  }

  uint32_t inserted_input_width() {
    return args.input_w + (args.input_w - 1) * args.insert_w;
  }

  uint32_t groups() { return args.groups; }

  uint32_t group_input_channels() { return args.input_c / args.groups; }

  uint32_t group_output_channels() { return args.output_c / args.groups; }

  uint32_t kernel_height() { return args.kh; }

  uint32_t kernel_width() { return args.kw; }

  uint32_t dilation_height() { return args.dilation_h; }

  uint32_t dilation_width() { return args.dilation_w; }

  uint32_t padding_top() { return args.pad_top; }

  uint32_t padding_bottom() { return args.pad_bottom; }

  uint32_t padding_left() { return args.pad_left; }

  uint32_t padding_right() { return args.pad_right; }

  int pad_value() { return args.pad_value; }
  int w_after_ins_pad(int w) {
    return (w - 1) * (1 + args.insert_w) + 1 + args.pad_left + args.pad_right;
  }
  int h_after_ins_pad(int h) {
    return (h - 1) * (1 + args.insert_h) + 1 + args.pad_bottom + args.pad_top;
  }

  uint32_t subsampling_height() {
    assert(args.stride_h >= 1);
    return args.stride_h;
  }

  uint32_t subsampling_width() {
    assert(args.stride_w >= 1);
    return args.stride_w;
  }

  uint32_t dilated_kernel_height() {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  uint32_t dilated_kernel_width() {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  uint32_t output_height() {
    uint32_t padded_input_height =
        padding_top() + inserted_input_height() + padding_bottom();
    return (padded_input_height - dilated_kernel_height()) /
               subsampling_height() +
           1;
  }

  uint32_t output_width() {
    uint32_t padded_input_width =
        padding_left() + inserted_input_width() + padding_right();
    return (padded_input_width - dilated_kernel_width()) / subsampling_width() +
           1;
  }

  enum TilePolicy {
    NoTilePolicyType,
    SingleBufferPolicyType,
    SingleBufferPs32PolicyType,
    ReuseWeightPolicyType,
    ReuseActivationPolicyType,
    MaxTilePolicyType,
  };
  struct CostModel {
    uint32_t totalRWSize;
    uint32_t wgtReadSize;
    uint32_t actReadSize;
    uint32_t actWriteSize;
  };

  void showCost(CostModel &cost);
  void getCost(CostModel &cost);
  bool isBetterCost(CostModel &from, CostModel &to);
  TilePolicy getReuseWgtOrActByCost();

  // Arguments from dialect
  Conv_ARGS args;

private:
  TileInfo tile_info;
  bool use_double_buffer;

  TilePolicy tilePolicy;

  // Global memory descriptor
  std::unique_ptr<GlobalMemoryDescriptor> gmInputDesc;
  std::unique_ptr<GlobalMemoryDescriptor> gmOutputDesc;
  std::unique_ptr<GlobalMemoryDescriptor> gmWeightDesc;
  std::unique_ptr<GlobalMemoryDescriptor> gmBiasDesc;
  std::unique_ptr<GlobalMemoryDescriptor>
      gmQuantDesc[2]; // quant_scale + quant_zeropoint
  std::unique_ptr<GlobalMemoryDescriptor> gmScaleLutDesc;

  // Local memory descriptor
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmInputDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmOutputDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmWeightDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmBiasDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmQuantDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmFusedActDescs;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmPreProcessDescs;

  // Collection of tiled commands
  std::vector<std::unique_ptr<CmdDescriptor>> cmdQueue;

  CModelDebug cModelDebug = {0};
};
} // namespace backend
} // namespace tpu_mlir
