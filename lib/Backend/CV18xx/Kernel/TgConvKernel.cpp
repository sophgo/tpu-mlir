//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgConvKernel.hpp"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"

#define DEBUG_TYPE "cvi_backend_conv_kernel"

namespace tpu_mlir {
namespace backend {
std::string IntraCmdParallelAnalysis::getAccessEventStr(AccessEvent event) {
  switch (event) {
  case WriteEvent:
    return "W";

  case ReadEvent:
    return "R";

  default:
    assert(0 && "Unexpected event");
    break;
  }

  return " ";
}

std::string IntraCmdParallelAnalysis::getAccessStateStr(AccessState state) {
  switch (state) {
  case UnknownState:
    return "UNK";

  case WriteState:
    return "W";

  case ReadAfterWriteState:
    return "RAW";

  case ReadAfterReadState:
    return "RAR";

  case WriteAfterWriteState:
    return "WAW";

  default:
    assert(0 && "Unexpected state");
  }

  return " ";
}

void IntraCmdParallelAnalysis::assignLmAccessState() {
  for (const auto &it : cmdQueue) {
    std::vector<uint32_t> lmIndexes = it->getLmIndexes();

    if (it->getCmdType() == CmdDescriptor::LoadBiasCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadBiasCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmBiasAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::LoadQuantCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadBiasCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmQuantAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::LoadInputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadInputCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmInputAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::LoadWeightCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadWeightCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmWeightAccessStates_[lmIndexes[0]], WriteEvent);
    } else if (it->getCmdType() == CmdDescriptor::LoadScaleLutTblCmdType) {

    } else if (it->getCmdType() == CmdDescriptor::ComputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  ComputeCmdDesc\n");

      uint32_t lmInputIndex = lmIndexes[0];  // input
      uint32_t lmWeightIndex = lmIndexes[1]; // weight
      uint32_t lmOutputIndex = lmIndexes[2]; // output

      receiveAccessEvent(&lmBiasAccessStates_[lmWeightIndex], ReadEvent);
      receiveAccessEvent(&lmWeightAccessStates_[lmWeightIndex], ReadEvent);
      receiveAccessEvent(&lmInputAccessStates_[lmInputIndex], ReadEvent);
      receiveAccessEvent(&lmOutputAccessStates_[lmOutputIndex], WriteEvent);
    } else if (it->getCmdType() == CmdDescriptor::ComputeScaleLutCmdType) {

    } else if (it->getCmdType() == CmdDescriptor::ComputeQuantCmdType) {
      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmWeightAccessStates_[lmIndexes[0]], WriteEvent);
    } else if (it->getCmdType() == CmdDescriptor::StoreOutputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  StoreOutputDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmOutputAccessStates_[lmIndexes[0]], ReadEvent);

    } else if (it->getCmdType() == CmdDescriptor::ParallelCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  ParallelCmdDesc\n");
    } else {
      assert(0 && "Unexpected cmd desc\n");
    }

    cmdLmStates_.push_back(std::make_unique<CmdLmState>(
        it->getCmdType(), lmBiasAccessStates_, lmQuantAccessStates_,
        lmWeightAccessStates_, lmInputAccessStates_, lmOutputAccessStates_));
  }
}

void IntraCmdParallelAnalysis::receiveAccessEvent(AccessState *state,
                                                  AccessEvent event) {
  switch (*state) {
  case UnknownState:
    assert(event == WriteEvent && "Expect write event in UNK state");
    // UNK -> W
    if (event == WriteEvent)
      *state = WriteState;
    break;

  case WriteState:
    // assert(event == ReadEvent && "Expect read event in W state");
    if (event == ReadEvent) {
      // W -> RAW
      *state = ReadAfterWriteState;
    } else if (event == WriteEvent) {
      // W -> WAW
      // Only the output state allowed in ps32 mode.
      *state = WriteAfterWriteState;
    }
    break;

  case ReadAfterWriteState:
    if (event == ReadEvent) {
      // RAW -> RAR
      *state = ReadAfterReadState;
    } else if (event == WriteEvent) {
      // RAW -> W, next tpu operation
      *state = WriteState;
    }
    break;

  case ReadAfterReadState:
    // RAR -> W, next tpu operation
    if (event == WriteEvent)
      *state = WriteState;
    break;

  case WriteAfterWriteState:
    if (event == ReadEvent) {
      // WAW -> WAR
      *state = ReadAfterWriteState;
    } else if (event == WriteEvent) {
      // WAW -> WAW
    } else {
      assert(0 && "Unexpected event in WAW state");
    }
    break;

  default:
    assert(0 && "Unexpected event");
    break;
  }
}

// Backward search of LoadBias or LoadWeight.
// Caller check the required command type.
uint32_t IntraCmdParallelAnalysis::reverseSearchBiasOrWeight(
    AccessState state, uint32_t lmIndex, uint32_t endQueueIndex) {
  assert(endQueueIndex < cmdQueue.size() && "Expect valid range");
  assert(lmIndex < cmdLmStates_[lmIndex]->weights_.size());

  for (int i = endQueueIndex; i >= 0; --i) {
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::LoadWeightCmdType &&
        cmdLmStates_[i]->weights_[lmIndex] == state)
      return static_cast<uint32_t>(i);
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::LoadBiasCmdType &&
        cmdLmStates_[i]->biass_[lmIndex] == state)
      return static_cast<uint32_t>(i);
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::LoadQuantCmdType &&
        cmdLmStates_[i]->quants_[lmIndex] == state)
      return static_cast<uint32_t>(i);
  }

  assert(0 && "Expect valid index of bias/weight");

  return endQueueIndex;
}

// Forward search of StoreOutput
uint32_t IntraCmdParallelAnalysis::searchStoreOutput(AccessState state,
                                                     uint32_t lmIndex,
                                                     uint32_t startQueueIndex) {
  assert(startQueueIndex < cmdQueue.size() && "Expect valid range");
  assert(lmIndex < cmdLmStates_[lmIndex]->outputs_.size());

  for (uint32_t i = startQueueIndex; i < cmdQueue.size(); ++i) {
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::StoreOutputCmdType &&
        cmdLmStates_[i]->outputs_[lmIndex] == state)
      return i;
  }

  assert(0 && "Expect valid index of output");

  return startQueueIndex;
}

// All of bias, weight and input are at RAW (read after write) state
bool IntraCmdParallelAnalysis::isIntrCmdParalTiu(uint32_t index) {
  assert(index < cmdQueue.size() && "Expect valid range");
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::ComputCmdType &&
         "Expect compute cmd");

  std::vector<uint32_t> lmIndexes = cmdQueue[index]->getLmIndexes();
  uint32_t lmInputIndex = lmIndexes[0];  // input
  uint32_t lmWeightIndex = lmIndexes[1]; // weight

  AccessState quantState = cmdLmStates_[index]->quants_[lmWeightIndex];
  AccessState biasState = cmdLmStates_[index]->biass_[lmWeightIndex];
  AccessState weightState = cmdLmStates_[index]->weights_[lmWeightIndex];
  AccessState inputState = cmdLmStates_[index]->inputs_[lmInputIndex];

  if (quantState == ReadAfterWriteState && biasState == ReadAfterWriteState &&
      weightState == ReadAfterWriteState && inputState == ReadAfterWriteState)
    return true;

  return false;
}

// LoadWeight at W (write) state
bool IntraCmdParallelAnalysis::isIntrCmdParalLoadWeight(
    uint32_t index, uint32_t lmWeightIndex) {
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadWeightCmdType &&
         "Expect load weight cmd");

  if (cmdLmStates_[index]->weights_[lmWeightIndex] == WriteState)
    return true;

  return false;
}

// StoreOutput at RAW(read after writer) state
bool IntraCmdParallelAnalysis::isIntrCmdParalStoreOutput(
    uint32_t index, uint32_t lmOutputIndex) {
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::StoreOutputCmdType &&
         "Expect load weight cmd");

  if (cmdLmStates_[index]->outputs_[lmOutputIndex] == ReadAfterWriteState)
    return true;

  return false;
}

void IntraCmdParallelAnalysis::tryEnableIntraCmdParal(uint32_t index) {
  assert(index < cmdQueue.size() && "Expect valid index");

  // Find compute command first
  if (cmdLmStates_[index]->cmdType_ != CmdDescriptor::ComputCmdType)
    return;

  std::vector<uint32_t> lmIndexes = cmdQueue[index]->getLmIndexes();
  uint32_t lmWeightIndex = lmIndexes[1]; // weight
  uint32_t lmOutputIndex = lmIndexes[2]; // output

  // Check compute command
  if (!isIntrCmdParalTiu(index))
    return;

  // Check loadWeight and StoreOutput
  uint32_t firstArgIndex =
      reverseSearchBiasOrWeight(WriteState, lmWeightIndex, index);
  uint32_t firstOutputIndex =
      searchStoreOutput(ReadAfterWriteState, lmOutputIndex, index);

  if (isIntrCmdParalLoadWeight(firstArgIndex, lmWeightIndex) &&
      isIntrCmdParalStoreOutput(firstOutputIndex, lmOutputIndex)) {

    // tuple of tdma load, tiu and tdma store
    cmdQueue[index]->setIntraCmdParalEnabled(true);
    cmdQueue[firstArgIndex]->setIntraCmdParalEnabled(true);
    cmdQueue[firstOutputIndex]->setIntraCmdParalEnabled(true);

    cmdLmStates_[index]->isIntraCmdParal_ = true;
    cmdLmStates_[firstArgIndex]->isIntraCmdParal_ = true;
    cmdLmStates_[firstOutputIndex]->isIntraCmdParal_ = true;
  }
}

void IntraCmdParallelAnalysis::analyze() {
  for (uint32_t i = 0; i < cmdQueue.size(); ++i) {
    assert(cmdQueue[i]->getCmdType() == cmdLmStates_[i]->cmdType_);

    tryEnableIntraCmdParal(i);
  }
}

void IntraCmdParallelAnalysis::dumpStates() {
  LLVM_DEBUG(llvm::dbgs() << "\n  IntraCmdParallelAnalysis::dumpStates:\n");

  // For input, bias, weight, store
  auto dumpLmIndex = [=](uint32_t index) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" << cmdQueue[index]->getLmIndexes()[0] << "]");
  };

  // For compute
  auto dumpLmIndexes = [=](uint32_t index) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" << cmdQueue[index]->getLmIndexes()[2] << "]");
  };

  // For parallel
  auto dumpLmParal = [=](uint32_t index) {
    if (cmdQueue[index]->isParallelEnabled())
      LLVM_DEBUG(llvm::dbgs() << "(E)");
    else
      LLVM_DEBUG(llvm::dbgs() << "(D)");
  };

  auto dumpCmdLmIndex = [=](uint32_t index) {
    if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadInputCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadBiasCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadQuantCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadWeightCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::StoreOutputCmdType)
      dumpLmIndex(index);
    else if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::ComputCmdType)
      dumpLmIndexes(index);
    else if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::ParallelCmdType)
      dumpLmParal(index);
  };

  for (uint32_t i = 0; i < cmdQueue.size(); ++i) {
    assert(cmdQueue[i]->getCmdType() == cmdLmStates_[i]->cmdType_);

    LLVM_DEBUG(llvm::dbgs()
               << "    [" << i << "] cmd "
               << CmdDescriptor::getCmdTypeStr(cmdLmStates_[i]->cmdType_));

    dumpCmdLmIndex(i);

    LLVM_DEBUG(llvm::dbgs()
               << ": bias " << getAccessStateStr(cmdLmStates_[i]->biass_[0])
               << "|" << getAccessStateStr(cmdLmStates_[i]->biass_[1])
               << ", weight " << getAccessStateStr(cmdLmStates_[i]->weights_[0])
               << "|" << getAccessStateStr(cmdLmStates_[i]->weights_[1])
               << ", input " << getAccessStateStr(cmdLmStates_[i]->inputs_[0])
               << "|" << getAccessStateStr(cmdLmStates_[i]->inputs_[1])
               << ", ouput " << getAccessStateStr(cmdLmStates_[i]->outputs_[0])
               << "|" << getAccessStateStr(cmdLmStates_[i]->outputs_[1]));

    if (cmdQueue[i]->isIntraCmdParalEnabled())
      LLVM_DEBUG(llvm::dbgs() << "    => O");

    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "\n");
}

// Input data layout (N, C, H, W) => (N, G, C, H, W)
void Conv::initializeGlobalMemInput() {
  // Actual physical layout
  // not logical layout (e.g. do_ic_alignment)
  std::vector<uint32_t> shapes = {batch_size(), groups(),
                                  group_input_channels(), input_height(),
                                  input_width()};
  gmInputDesc =
      std::make_unique<GlobalMemoryDescriptor>(shapes, args.input_fmt);
  gmInputDesc->setLayerId(args.layer_id);
  gmInputDesc->setAddress(args.ga_ifmap);
}

// Output data layout (N, C, H, W) => (N, G, C, H, W)
void Conv::initializeGlobalMemOutput() {
  uint32_t og = isDwConv() ? 1 : groups();
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  std::vector<uint32_t> shapes = {batch_size(), og, oc, output_height(),
                                  output_width()};
  gmOutputDesc =
      std::make_unique<GlobalMemoryDescriptor>(shapes, args.output_fmt);
  gmOutputDesc->setLayerId(args.layer_id);
  gmOutputDesc->setAddress(args.ga_ofmap);
}

// Weight data layout (Og, Oc, Kh*Kw, Ic) => (1, Og, Oc, Kh*Kw, Ic)
void Conv::initializeGlobalMemWeight() {
  uint32_t input_c = group_input_channels();

  // Physical layout
  // weight is already altered for do_ic_alignment
  // do_ic_alignment is not applied in depthwise convolution.
  input_c =
      !isDwConv() && args.do_ic_alignment ? align_up(input_c, 2l) : input_c;

  std::vector<uint32_t> shapes = {1, groups(), group_output_channels(),
                                  kernel_height() * kernel_width(), input_c};
  auto fmt = args.do_quant ? CVK_FMT_I8 : args.tiu_fmt;
  gmWeightDesc = std::make_unique<GlobalMemoryDescriptor>(shapes, fmt);
  gmWeightDesc->setLayerId(args.layer_id);
  gmWeightDesc->setAddress(args.ga_weight);
}

// Bias data layout
//   Per-channel: (1, Og, Oc, 1, [9/5])
//   Per-tensor:  (2, Og, Oc, 1, 1)
void Conv::initializeGlobalBias() {
  if (!args.do_chl_quan && !args.do_bias)
    return;

  std::vector<uint32_t> shapes;
  if (args.do_chl_quan) {
    uint32_t pc_bias_size = CV18xx::chan_quan_param_size(args.do_bias);
    shapes = {1, groups(), group_output_channels(), 1, pc_bias_size};
  } else {
    shapes = {2, groups(), group_output_channels(), 1, 1};
  }

  gmBiasDesc = std::make_unique<GlobalMemoryDescriptor>(shapes, args.tiu_fmt);
  gmBiasDesc->setLayerId(args.layer_id);
  gmBiasDesc->setAddress(args.ga_bias);
}

void Conv::initializeGlobalQuant() {
  if (!args.do_quant)
    return;

  std::vector<uint32_t> shapes = {1, groups(), group_output_channels(), 1, 1};

  gmQuantDesc[0] =
      std::make_unique<GlobalMemoryDescriptor>(shapes, args.tiu_fmt);
  gmQuantDesc[0]->setLayerId(args.layer_id);
  gmQuantDesc[0]->setAddress(args.ga_scale);
  gmQuantDesc[1] =
      std::make_unique<GlobalMemoryDescriptor>(shapes, args.tiu_fmt);
  gmQuantDesc[1]->setLayerId(args.layer_id);
  gmQuantDesc[1]->setAddress(args.ga_zeropoint);
}

void CModelDebug::dumpDims(std::vector<uint32_t> &dims) {
  if (dims.size() < 5)
    return;

  LLVM_DEBUG(llvm::dbgs() << "(" << dims[NGCHW::N] << ", " << dims[NGCHW::G]
                          << ", " << dims[NGCHW::C] << ", " << dims[NGCHW::H]
                          << ", " << dims[NGCHW::W] << ")");
}

void CModelDebug::dump() {
  if (!enabled_ || !found_)
    return;

  // Replace with raw_ostream
  LLVM_DEBUG(llvm::dbgs() << "CMODEL Debug:\n"
                          << "  enabled " << enabled_ << ", layer_id "
                          << layerId_ << "\n"
                          << "  output addr "
                          << llvm::format_hex(output_.addr, 10) << "(offset="
                          << llvm::format_hex(output_.addrOffset, 10)
                          << "), poss");
  dumpDims(output_.poss);
  LLVM_DEBUG(llvm::dbgs() << ", shapes");
  dumpDims(output_.shapes);

  LLVM_DEBUG(llvm::dbgs() << "\n  input addr "
                          << llvm::format_hex(input_.addr, 10) << "(offset="
                          << llvm::format_hex(input_.addrOffset, 10)
                          << "), poss");
  dumpDims(input_.poss);
  LLVM_DEBUG(llvm::dbgs() << ", shapes");
  dumpDims(input_.shapes);
  LLVM_DEBUG(llvm::dbgs() << "\n  weight addr "
                          << llvm::format_hex(weight_.addr, 10) << "(offset="
                          << llvm::format_hex(weight_.addrOffset, 10)
                          << ", poss");
  dumpDims(weight_.poss);
  LLVM_DEBUG(llvm::dbgs() << ", shape");
  dumpDims(weight_.shapes);
  LLVM_DEBUG(llvm::dbgs() << "\n  bias addr "
                          << llvm::format_hex(bias_.addr, 10) << "(offset="
                          << llvm::format_hex(bias_.addrOffset, 10)
                          << "), poss");
  dumpDims(bias_.poss);
  LLVM_DEBUG(llvm::dbgs() << ", shapes");
  dumpDims(bias_.shapes);
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

void Conv::initializeGlobalScaleLut() {
  if (args.do_scale_lut) {
    cvk_tl_shape_t tl_shape = CV18xx::lut_table_shape(args.input_fmt);
    std::vector<uint32_t> shapes = {tl_shape.n, tl_shape.c, tl_shape.h,
                                    tl_shape.w};
    gmScaleLutDesc =
        std::make_unique<GlobalMemoryDescriptor>(shapes, args.input_fmt);
    gmScaleLutDesc->setLayerId(args.layer_id);
    gmScaleLutDesc->setAddress(args.ga_scale_lut);
  }
}

void Conv::initializeGlobalMem() {
  initializeGlobalMemInput();
  initializeGlobalMemOutput();
  initializeGlobalMemWeight();
  initializeGlobalBias();
  initializeGlobalQuant();
  initializeGlobalScaleLut();
}

void Conv::initializeTile() {
  tile_info.n = 1;
  tile_info.oc = 1;
  tile_info.ic = 1;
  tile_info.h = 1;
  tile_info.w = 1;
  tile_info.n_step = batch_size();
  tile_info.oc_step = group_output_channels();
  tile_info.oh_step = output_height();
  tile_info.ow_step = output_width();
  tile_info.ih_step = input_height();
  tile_info.iw_step = input_width();
  tile_info.ic_step = group_input_channels();

  use_double_buffer = false;
}

void Conv::initializeFusedActivation() {
  // Check conv+relu or conv+leaky relu
  args.fused_conv_relu = false;
  args.do_leaky_relu = false;
  if (args.do_activation) {
    if (!args.activation_arg || args.activation_arg[0] == 0.0f)
      args.fused_conv_relu = true;
    else
      args.do_leaky_relu = true;
  }
}

uint32_t Conv::getElementTypeSize(cvk_fmt_t fmt) {
  switch (fmt) {
  case CVK_FMT_F32:
    return 4;
  case CVK_FMT_BF16:
    return 2;
  default:
    return 1;
  }
}

void Conv::allocateTiledLocalMem(
    std::vector<std::unique_ptr<LocalMemoryDescriptor>> &lmDescs,
    uint32_t count, std::vector<uint32_t> shapes, uint32_t eu_align) {
  assert(shapes.size() == 5 && "Expect 5D tensor");

  auto curSize = lmDescs.size();
  for (uint32_t i = 0; i < count; ++i) {
    lmDescs.push_back(std::make_unique<LocalMemoryDescriptor>(
        shapes, args.tiu_fmt, eu_align));
    lmDescs.back()->setLayerId(args.layer_id);
    lmDescs.back()->allocate();
  }

  assert((lmDescs.size() - curSize) == count && "Expect all allocated");
}

// Shape (tiledN, 1, IC/g, tiledIH, tiledIW)
// Shape (tiledN, 1, OC, tiledIH, tiledIW) for dw-conv
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfInput() {
  uint32_t ic_step = isDwConv() ? tile_info.oc_step : tile_info.ic_step;
  std::vector<uint32_t> shapes = {tile_info.n_step, 1, ic_step,
                                  tile_info.ih_step, tile_info.iw_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfInput() {
  return 1; // aligned
}

void Conv::allocateLocalMemOfInput() {
  uint32_t count = use_double_buffer ? 2 : 1;

  allocateTiledLocalMem(lmInputDescs, count,
                        getTiledShapesForLmAllocationOfInput(),
                        getTiledEuAlignForLmAllocationOfInput());
}

void Conv::deallocateLocalMemOfInput() {
  if (use_double_buffer)
    lmInputDescs[1]->free();

  lmInputDescs[0]->free();
}

// Shape (tiledN, 1, Oc/g, tiledOH, tiledOW)
// Shape (tiledN, 1, Oc, tiledOH, tiledOW) for dw-conv
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfOuput() {
  uint32_t ofmapSizeMultiplier =
      (tile_info.ic_step < group_input_channels()) ? 4 : 1;

  // dw-conv supports direct ps32 output, but not ps32 mode
  if (isDwConv()) {
    if (args.ps32_output && args.tiu_fmt == CVK_FMT_BF16)
      ofmapSizeMultiplier = 2;
    else if (args.ps32_output && args.tiu_fmt == CVK_FMT_I8)
      ofmapSizeMultiplier = 4;
  }

  std::vector<uint32_t> shapes = {tile_info.n_step * ofmapSizeMultiplier, 1,
                                  tile_info.oc_step, tile_info.oh_step,
                                  tile_info.ow_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfOutput() {
  return 1; // aligned
}

void Conv::allocateLocalMemOfOutput() {
  uint32_t count = use_double_buffer ? 2 : 1;
  allocateTiledLocalMem(lmOutputDescs, count,
                        getTiledShapesForLmAllocationOfOuput(),
                        getTiledEuAlignForLmAllocationOfOutput());
}

void Conv::deallocateLocalMemOfOutput() {
  if (use_double_buffer)
    lmOutputDescs[1]->free();

  lmOutputDescs[0]->free();
}

// Shape (1, 1, tiledOc, kh*kw, ic)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t>
Conv::getTiledGmShapesOfWeightForTdmaLoad(std::vector<uint32_t> gmOutputPoss,
                                          uint32_t icPos) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);
  uint32_t cur_ic = std::min(group_input_channels() - icPos, tile_info.ic_step);

  std::vector<uint32_t> tiledShapes = {
      1, 1, cur_oc, kernel_height() * kernel_width(), cur_ic};

  return tiledShapes;
}

// TIU shape != tdma shape
//   TDMA shapes (1, 1, tiledOc, kh*kw, ic)
//   TIU shapes  (1, ic, tiledOc, kh, kw)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t>
Conv::getTiledLmShapesOfWeightForTiu(std::vector<uint32_t> gmOutputPoss,
                                     uint32_t icPos) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);
  uint32_t cur_ic = std::min(group_input_channels() - icPos, tile_info.ic_step);

  std::vector<uint32_t> shapes = {1, cur_ic, cur_oc, kernel_height(),
                                  kernel_width()};

  return shapes;
}

// Shape (1, 1, tiledOc, Kh*Kw, Ic)
// Shape (1, 1, tiledOc, Kh*Kw, 1) for dw-conv
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfWeight() {
  std::vector<uint32_t> shapes = {1, 1, tile_info.oc_step,
                                  kernel_height() * kernel_width(),
                                  tile_info.ic_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfWeight() {
  // return 0; // Not aligned
  return isDwConv() ? 1 : 0;
}

void Conv::allocateLocalMemOfWeight() {
  uint32_t count = use_double_buffer ? 2 : 1;

  allocateTiledLocalMem(lmWeightDescs, count,
                        getTiledShapesForLmAllocationOfWeight(),
                        getTiledEuAlignForLmAllocationOfWeight());
}

void Conv::deallocateLocalMemOfWeight() {
  if (use_double_buffer)
    lmWeightDescs[1]->free();

  lmWeightDescs[0]->free();
}

// Per-channel: (1, 1, tiled_oc, 1, [5/9])
//   w/  bias: bias(4) + multiplier(4) + shift(1)
//   w/o bias: multiplier(4) + shift(1)
// Per-tensor:  (2, 1, tiled_oc, 1, 1)
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfBias() {
  std::vector<uint32_t> shapes;

  if (args.do_chl_quan) {
    uint32_t pc_bias_size = CV18xx::chan_quan_param_size(args.do_bias);
    shapes = {1, 1, tile_info.oc_step, 1, pc_bias_size};
  } else if (args.do_bias) {
    shapes = {2, 1, tile_info.oc_step, 1, 1};
  }

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfBias() {
  return 0; // Not aligned
}

void Conv::allocateLocalMemOfBias() {
  if (args.do_chl_quan || args.do_bias) {
    uint32_t count = use_double_buffer ? 2 : 1;
    allocateTiledLocalMem(lmBiasDescs, count,
                          getTiledShapesForLmAllocationOfBias(),
                          getTiledEuAlignForLmAllocationOfBias());
  }
}

void Conv::deallocateLocalMemOfBias() {
  if (args.do_chl_quan || args.do_bias) {
    if (use_double_buffer)
      lmBiasDescs[1]->free();

    lmBiasDescs[0]->free();
  }
}

void Conv::allocateLocalMemOfQuant() {
  if (args.do_quant) {
    uint32_t count = use_double_buffer ? 4 : 2;
    std::vector<uint32_t> shape = {1, 1, tile_info.oc_step, 1, 1};
    allocateTiledLocalMem(lmQuantDescs, count, shape, 0);
  }
}

void Conv::deallocateLocalMemOfQuant() {
  if (lmQuantDescs.size()) {
    for (int i = lmQuantDescs.size() - 1; i >= 0; --i)
      lmQuantDescs[i]->free();
  }
}

void Conv::allocateLocalMemOfFusedActivation() {
  if (args.do_leaky_relu) {
    // Leaky relu needs two local memory for tl_reg, tl_relu
    // Same setting as output
    allocateTiledLocalMem(lmFusedActDescs, 2,
                          getTiledShapesForLmAllocationOfOuput(),
                          getTiledEuAlignForLmAllocationOfOutput());
  }
}

void Conv::deallocateLocalMemOfFusedActivation() {
  if (lmFusedActDescs.size()) {
    for (int i = lmFusedActDescs.size() - 1; i >= 0; --i)
      lmFusedActDescs[i]->free();
  }
}

void Conv::allocateLocalMemOfPreProcess() {
  if (args.do_scale_lut) {
    cvk_tl_shape_t tl_shape = CV18xx::lut_table_shape(args.input_fmt);
    std::vector<uint32_t> shapes = {tl_shape.n, 1, tl_shape.c, tl_shape.h,
                                    tl_shape.w};
    allocateTiledLocalMem(lmPreProcessDescs, 1, shapes, 1);
  }
}

void Conv::deallocateLocalMemOfPreProcess() {
  if (lmPreProcessDescs.size()) {
    for (int i = lmPreProcessDescs.size() - 1; i >= 0; --i)
      lmPreProcessDescs[i]->free();
  }
}

void Conv::allocateAllLocalMem() {
  allocateLocalMemOfWeight();
  allocateLocalMemOfInput();
  allocateLocalMemOfOutput();
  allocateLocalMemOfBias();
  allocateLocalMemOfQuant();
  allocateLocalMemOfFusedActivation();
  allocateLocalMemOfPreProcess();
}

void Conv::deallocateAllLocalMem() {
  //
  // Release resource in reverse order
  //
  deallocateLocalMemOfPreProcess();
  deallocateLocalMemOfFusedActivation();
  deallocateLocalMemOfQuant();
  deallocateLocalMemOfBias();
  deallocateLocalMemOfOutput();
  deallocateLocalMemOfInput();
  deallocateLocalMemOfWeight();
}

// H/W does not support group convolution.
// S/W handles one group at once.
//
// Bias data layout
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t>
Conv::getTiledGmShapesOfBiasForTdmaLoad(std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);

  // TDMA shapes same as allocation except group fixed to 1
  std::vector<uint32_t> shapes = {gmBiasDesc->getShapes()[NGCHW::N], 1, cur_oc,
                                  1, gmBiasDesc->getShapes()[NGCHW::W]};

  return shapes;
}

std::vector<uint32_t>
Conv::getTiledGmShapesOfQuantForTdmaLoad(std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);

  // TDMA shapes same as allocation except group fixed to 1
  std::vector<uint32_t> shapes = {gmQuantDesc[0]->getShapes()[NGCHW::N], 1,
                                  cur_oc, 1,
                                  gmQuantDesc[0]->getShapes()[NGCHW::W]};

  return shapes;
}

// Bias data layout
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// But
//   TIU per-channel: (1, 1, tiledOc, 1, 1)
//   TIU per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t>
Conv::getTiledLmShapesOfBiasForTiu(std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);

  std::vector<uint32_t> shapes = {gmBiasDesc->getShapes()[NGCHW::N], 1, cur_oc,
                                  1, 1};

  return shapes;
}

std::vector<uint32_t>
Conv::getTiledLmShapesOfQuantForTiu(std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oc = isDwConv() ? groups() : group_output_channels();
  uint32_t cur_oc = std::min(oc - oc_pos, tile_info.oc_step);

  std::vector<uint32_t> shapes = {gmQuantDesc[0]->getShapes()[NGCHW::N], 1,
                                  cur_oc, 1, 1};

  return shapes;
}

// Bias shape
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
void Conv::loadBias(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                    uint32_t cmdQueueIndex) {
  if (!args.do_chl_quan && !args.do_bias)
    return;

  uint32_t ig_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];

  // Global memory
  std::vector<uint32_t> gm_shapes =
      getTiledGmShapesOfBiasForTdmaLoad(gmOutputPoss);
  std::vector<uint32_t> tiled_cur_poss = {0, ig_pos, oc_pos, 0, 0};

  uint64_t ga_offset = gmBiasDesc->getCurrentOffset(tiled_cur_poss);
  uint64_t ga_load = gmBiasDesc->getAddress() + ga_offset;
  cvk_tg_stride_t gm_stride = {
      gmBiasDesc->getStrides()[NGCHW::N], gmBiasDesc->getStrides()[NGCHW::C],
      gmBiasDesc->getStrides()[NGCHW::H], gmBiasDesc->getStrides()[NGCHW::W]};

  // Local memory
  cvk_tl_t tl_bias;
  cvk_tl_shape_t tl_bias_shape = {gm_shapes[NGCHW::N], gm_shapes[NGCHW::C],
                                  gm_shapes[NGCHW::H], gm_shapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_bias, tl_bias_shape,
                           lmBiasDescs[lmIndex]->getDataFormat(),
                           lmBiasDescs[lmIndex]->getEuAlign());
  tl_bias.start_address = lmBiasDescs[lmIndex]->getAddress();

  // LLVM_DEBUG(
  //     llvm::dbgs() << "\n  [ig=" << ig_pos << "][oc_pos=" << oc_pos
  //                  << "] tdma_load_stride:\n"
  //                  << "    tl_bias gaddr " << llvm::format_hex(ga_load, 10)
  //                  << "(offset=" << llvm::format_hex(ga_offset, 10)
  //                  << "), gstride (" << gm_stride.n << ", " << gm_stride.c
  //                  << ", " << gm_stride.h << ")\n"
  //                  << "    laddr "
  //                  << llvm::format_hex(tl_bias.start_address, 10) << ", shape
  //                  ("
  //                  << tl_bias.shape.n << ", " << tl_bias.shape.c << ", "
  //                  << tl_bias.shape.h << ", " << tl_bias.shape.w << ")\n\n");

  if (args.tiu_fmt == CVK_FMT_I8)
    CV18xx::tdma_load_stride(&tl_bias, ga_load, gm_stride);
  else if (args.tiu_fmt == CVK_FMT_BF16)
    CV18xx::tdma_load_stride(&tl_bias, ga_load, gm_stride);
  else {
    assert(0 && "Bias only supports i8/bf16");
  }

  cModelDebug.recordBias(args.layer_id, gmOutputPoss, ga_load, ga_offset,
                         tiled_cur_poss, gm_shapes);
}

void Conv::loadQuant(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                     uint32_t cmdQueueIndex) {
  if (!args.do_quant)
    return;

  uint32_t ig_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];

  // Global memory
  std::vector<uint32_t> gm_shapes =
      getTiledGmShapesOfQuantForTdmaLoad(gmOutputPoss);
  std::vector<uint32_t> tiled_cur_poss = {0, ig_pos, oc_pos, 0, 0};

  for (int i = 0; i < 2; i++) {
    uint64_t ga_offset = gmQuantDesc[i]->getCurrentOffset(tiled_cur_poss);
    uint64_t ga_load = gmQuantDesc[i]->getAddress() + ga_offset;
    cvk_tg_stride_t gm_stride = {gmQuantDesc[i]->getStrides()[NGCHW::N],
                                 gmQuantDesc[i]->getStrides()[NGCHW::C],
                                 gmQuantDesc[i]->getStrides()[NGCHW::H],
                                 gmQuantDesc[i]->getStrides()[NGCHW::W]};

    // Local memory
    cvk_tl_t tl_quant;
    cvk_tl_shape_t tl_quant_shape = {gm_shapes[NGCHW::N], gm_shapes[NGCHW::C],
                                     gm_shapes[NGCHW::H], gm_shapes[NGCHW::W]};
    CV18xx::lmem_init_tensor(&tl_quant, tl_quant_shape,
                             lmQuantDescs[i + lmIndex * 2]->getDataFormat(),
                             lmQuantDescs[i + lmIndex * 2]->getEuAlign());
    tl_quant.start_address = lmQuantDescs[i + lmIndex * 2]->getAddress();

    CV18xx::tdma_load_stride(&tl_quant, ga_load, gm_stride);
  }
}

// Weight shape (1, 1, tiledOc, kh*kw, ic)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
void Conv::loadWeight(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                      uint32_t cmdQueueIndex, uint32_t icPos) {
  uint32_t ig_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  if (!isDwConv()) {
    assert(group_output_channels() > oc_pos && "Expect valid tiled weight");
    assert(group_output_channels() >= tile_info.oc_step &&
           "Expect valid tiled weight");
  }

  std::vector<uint32_t> tiled_shapes =
      getTiledGmShapesOfWeightForTdmaLoad(gmOutputPoss, icPos);
  std::vector<uint32_t> tiled_cur_poss = {0, ig_pos, oc_pos, 0, icPos};

  // Global memory
  uint64_t ga_offset = gmWeightDesc->getCurrentOffset(tiled_cur_poss);
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(gmWeightDesc->getAddress());
  ts_data.start_address = gmWeightDesc->getAddress() + ga_offset;
  ts_data.fmt = gmWeightDesc->getDataFormat();
  ts_data.shape = {tiled_shapes[NGCHW::N], tiled_shapes[NGCHW::C],
                   tiled_shapes[NGCHW::H], tiled_shapes[NGCHW::W]};
  ts_data.stride = {gmWeightDesc->getStrides()[NGCHW::N],
                    gmWeightDesc->getStrides()[NGCHW::C],
                    gmWeightDesc->getStrides()[NGCHW::H],
                    gmWeightDesc->getStrides()[NGCHW::W]};

  // Local memory
  cvk_tl_t *tl_allocated_weight = lmWeightDescs[lmIndex]->getAllocated();

  cvk_tl_shape_t tl_load_shape = {
      tiled_shapes[NGCHW::N], tiled_shapes[NGCHW::C], tiled_shapes[NGCHW::H],
      tiled_shapes[NGCHW::W]};
  cvk_tl_t tl_load_weight;
  CV18xx::lmem_init_tensor(&tl_load_weight, tl_load_shape,
                           tl_allocated_weight->fmt,
                           tl_allocated_weight->eu_align);
  tl_load_weight.start_address = tl_allocated_weight->start_address;

  uint8_t intraCmdParal = getTdmaLoadWeightIntraCmdParal(cmdQueueIndex);

  // LLVM_DEBUG(
  //     llvm::errs() << "  [ig=" << ig_pos << "][oc_pos=" << oc_pos
  //                  << "] tdma_load_stride:\n"
  //                  << "    loadWeight tl_weight gaddr "
  //                  << llvm::format_hex(ts_data.start_address, 10) <<
  //                  "(offset="
  //                  << ga_offset << "), gstride (" << ts_data.stride.n << ", "
  //                  << ts_data.stride.c << ", " << ts_data.stride.h << ")\n"
  //                  << "    laddr "
  //                  << llvm::format_hex(tl_load_weight.start_address, 10)
  //                  << ", shape (" << tl_load_weight.shape.n << ", "
  //                  << tl_load_weight.shape.c << ", " <<
  //                  tl_load_weight.shape.h
  //                  << ", " << tl_load_weight.shape.w << ")\n"
  //                  << "    intraCmdParal " << (int)intraCmdParal << "\n");

  if (!args.do_load_cmpr_wgt) {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = &tl_load_weight;
    p1.layer_id = args.layer_id;
    p1.intra_cmd_paral = intraCmdParal;
    CV18xx::tdma_g2l_tensor_copy(&p1);
  } else {
    cvk_tdma_g2l_tensor_copy_decompressed_param_t p1 = {0};
    cvk_cmpr_tg_t ts_cmpr = {0};
    ts_cmpr.t = ts_data;
    p1.src = &ts_cmpr;
    p1.dst = &tl_load_weight;
    p1.layer_id = args.layer_id;
    p1.intra_cmd_paral = intraCmdParal;
    CV18xx::tdma_g2l_tensor_copy_decompressed(&p1);
  }

  cModelDebug.recordWeight(args.layer_id, gmOutputPoss, ts_data.start_address,
                           ga_offset, tiled_cur_poss, tiled_shapes);
}

// Calculate the position, shape, padding of tiled input from the tiled output.
// For do_ic_alignment, cur_gm_input_shapes is not physical layout.
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
// padding (top, bottom, left, right)
void Conv::getTiledGmPossAndShapesOfInputForTiu(
    std::vector<uint32_t> gmOutputPoss,
    std::vector<uint32_t> gmOutputPossShapes,
    std::vector<uint32_t> &cur_gm_input_poss,
    std::vector<uint32_t> &cur_gm_input_shapes,
    std::vector<uint32_t> &cur_gm_input_paddings, uint32_t icPos) {

  uint32_t g_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t oh_pos = gmOutputPoss[NGCHW::H];
  uint32_t cur_oc = gmOutputPossShapes[NGCHW::C];
  uint32_t cur_oh = gmOutputPossShapes[NGCHW::H];
  uint32_t oh_top = oh_pos;
  uint32_t oh_bot = oh_top + cur_oh;
  uint32_t ih_top =
      std::max(int(oh_top * subsampling_height()) - int(padding_top()), 0);
  uint32_t ih_bot = std::min((oh_bot - 1) * subsampling_height() +
                                 dilated_kernel_height() - padding_top(),
                             inserted_input_height());

  ih_top = ceiling_func(ih_top, 1 + insert_height());
  ih_bot = ceiling_func(ih_bot, 1 + insert_height());
  uint32_t cur_ih = ih_bot - ih_top;

  uint32_t ph_top = 0;
  if (ih_top == 0) {
    ph_top = padding_top() - oh_top * subsampling_height();
  } else {
    int gap =
        (oh_top * subsampling_height() - padding_top()) % (1 + insert_height());
    ph_top = (gap == 0) ? 0 : (1 + insert_height() - gap);
  }

  uint32_t ph_bot = 0;
  if (ih_bot == input_height()) {
    ph_bot = (oh_bot - 1) * subsampling_height() + dilated_kernel_height() -
             padding_top() - inserted_input_height();
  } else {
    ph_bot = (oh_bot - 1) * subsampling_height() + dilated_kernel_height() -
             padding_top() - (ih_bot + (ih_bot - 1) * insert_height());
  }

  uint32_t ow_pos = gmOutputPoss[NGCHW::W];       // NCHW
  uint32_t cur_ow = gmOutputPossShapes[NGCHW::W]; // NCHW
  uint32_t ow_left = ow_pos;
  uint32_t ow_right = ow_left + cur_ow;
  uint32_t iw_left =
      std::max(int(ow_left * subsampling_width()) - int(padding_left()), 0);
  uint32_t iw_right = std::min((ow_right - 1) * subsampling_width() +
                                   dilated_kernel_width() - padding_left(),
                               inserted_input_width());
  iw_left = ceiling_func(iw_left, 1 + insert_width());
  iw_right = ceiling_func(iw_right, 1 + insert_width());
  uint32_t cur_iw = iw_right - iw_left;

  // For better DMA transfer efficiency, use whole width.
  //   E.g.
  //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
  //
  //     input (27, 27) needed, but (27, 28) is better
  if ((int)tile_info.iw_step == args.input_w && cur_iw < tile_info.iw_step)
    cur_iw = tile_info.iw_step;

  uint32_t pw_left = 0;
  if (iw_left == 0) {
    pw_left = padding_left() - ow_left * subsampling_width();
  } else {
    int gap =
        (ow_left * subsampling_width() - padding_left()) % (1 + insert_width());
    pw_left = (gap == 0) ? 0 : (1 + insert_width() - gap);
  }

  uint32_t pw_right = 0;
  if (iw_right == input_width()) {
    pw_right = (ow_right - 1) * subsampling_width() + dilated_kernel_width() -
               padding_left() - inserted_input_width();
  } else {
    pw_right = (ow_right - 1) * subsampling_width() + dilated_kernel_width() -
               padding_left() - (iw_right + (iw_right - 1) * insert_width());
  }

  uint32_t n_pos = gmOutputPoss[NGCHW::N];
  uint32_t cur_n = gmOutputPossShapes[NGCHW::N];
  uint32_t cur_ic = std::min(group_input_channels() - icPos, tile_info.ic_step);
  uint32_t cur_c = isDwConv() ? cur_oc : cur_ic;
  uint32_t c_pos = isDwConv() ? oc_pos : icPos;
  cur_gm_input_shapes = {cur_n, 1, cur_c, cur_ih, cur_iw};
  cur_gm_input_poss = {n_pos, g_pos, c_pos, ih_top, iw_left};

  // {top, bottom, left, right}
  cur_gm_input_paddings = {ph_top, ph_bot, pw_left, pw_right};

  // LLVM_DEBUG(llvm::dbgs() << "\n  [n_pos=" << gmOutputPoss[NGCHW::N]
  //                         << "][ig=" << gmOutputPoss[NGCHW::G]
  //                         << "][oc_pos=" << gmOutputPoss[NGCHW::C]
  //                         << "][oh_pos=" << gmOutputPoss[NGCHW::H]
  //                         << "][ow_pos=" << gmOutputPoss[NGCHW::W]
  //                         << "][ic_pos=" << icPos
  //                         << "] input pos (n=" << cur_gm_input_poss[NGCHW::N]
  //                         << ", g=" << cur_gm_input_poss[NGCHW::G]
  //                         << ", c=" << cur_gm_input_poss[NGCHW::C]
  //                         << ", h=" << cur_gm_input_poss[NGCHW::H]
  //                         << ", w=" << cur_gm_input_poss[NGCHW::W]
  //                         << "), shapes (n=" << cur_gm_input_shapes[NGCHW::N]
  //                         << ", g=" << cur_gm_input_shapes[NGCHW::G] << ", c"
  //                         << cur_gm_input_shapes[NGCHW::C]
  //                         << ", h=" << cur_gm_input_shapes[NGCHW::H]
  //                         << ", w=" << cur_gm_input_shapes[NGCHW::W] <<
  //                         ")\n");
}

// H/W does not support group convolution.
// S/W handles one group at once.
//
// Shape (tiledN, 1, tiledOc, tiledOh, tiledOw)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t>
Conv::getTiledGmShapesOfOutputForTiu(std::vector<uint32_t> gmOutputPoss) {
  std::vector<uint32_t> outputShapes = gmOutputDesc->getShapes();
  std::vector<uint32_t> tiledOutputSteps = {
      tile_info.n_step, 1, tile_info.oc_step, tile_info.oh_step,
      tile_info.ow_step};

  std::vector<uint32_t> tiledOutputShapes;
  for (uint32_t i = 0; i < tiledOutputSteps.size(); ++i)
    tiledOutputShapes.push_back(
        std::min(outputShapes[i] - gmOutputPoss[i], tiledOutputSteps[i]));

  return tiledOutputShapes;
}

// It is possible that last tile of input only includes padding.
// Fill constant instead of load from global memory.
void Conv::fillConstantLmInput(cvk_tl_t *lmLoad,
                               std::vector<uint32_t> &cur_gm_input_paddings) {
  // Use pad top or bottom as height

  lmLoad->shape.h = std::max(cur_gm_input_paddings[0],
                             cur_gm_input_paddings[1]); // top or bottom
  lmLoad->stride =
      CV18xx::tl_default_stride(lmLoad->shape, lmLoad->fmt, lmLoad->eu_align);
  CV18xx::tiu_zeros(args.layer_id, lmLoad);
}

// Adjust input and padding for pad-only input.
void Conv::adjustComputeForPadOnlyInput(
    cvk_tl_t *lmInput, std::vector<uint32_t> &cur_gm_input_paddings) {

  // No need to change if height is non-zero
  if (lmInput->shape.h)
    return;

  // Use pad bottom as height
  // Clear pad bottom.
  if (cur_gm_input_paddings[0] > 0) {
    lmInput->shape.h = cur_gm_input_paddings[0];
    cur_gm_input_paddings[0] = 0;
  } else {
    lmInput->shape.h = cur_gm_input_paddings[1];
    cur_gm_input_paddings[1] = 0;
  }
  lmInput->stride = CV18xx::tl_default_stride(lmInput->shape, lmInput->fmt,
                                              lmInput->eu_align);
}

void Conv::adjustComputeForPs32Output(cvk_tl_t *lmOutput) {
  if (!args.ps32_output)
    return;

  int ofmap_multiplier = 1;
  if (args.ps32_output && args.tiu_fmt == CVK_FMT_BF16)
    ofmap_multiplier = 2;
  else if (args.ps32_output && args.tiu_fmt == CVK_FMT_I8)
    ofmap_multiplier = 4;

  lmOutput->shape.n *= ofmap_multiplier;
}

void Conv::adjustStoreForPs32Output(cvk_tl_t *lmOutput, cvk_tg_t *gmOutput,
                                    uint64_t ga_offset) {
  if (!args.ps32_output)
    return;

  int ofmap_multiplier = 1;
  if (args.ps32_output && args.tiu_fmt == CVK_FMT_BF16)
    ofmap_multiplier = 2;
  else if (args.ps32_output && args.tiu_fmt == CVK_FMT_I8)
    ofmap_multiplier = 4;

  lmOutput->shape.n *= ofmap_multiplier;

  uint64_t start_addr = gmOutput->start_address - ga_offset;
  gmOutput->start_address = start_addr + ga_offset * ofmap_multiplier;
}

// Input shape (tiledN, 1, tiledOc, tiledIh, tiledIw)
//
// Calculate input shape from output
// gmOutputPoss shape (n_pos, (ig_pos, oc_pos), oh_pos, ow_pos)
void Conv::loadInput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                     uint32_t cmdQueueIndex, uint32_t icPos) {
  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);

  std::vector<uint32_t> cur_gm_input_poss;
  std::vector<uint32_t> cur_gm_input_shapes;
  std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
  getTiledGmPossAndShapesOfInputForTiu(gmOutputPoss, gmOutputPossShapes,
                                       cur_gm_input_poss, cur_gm_input_shapes,
                                       cur_gm_input_paddings, icPos);

  uint64_t ga_input_offset = gmInputDesc->getCurrentOffset(cur_gm_input_poss);
  uint64_t ga_input_load = gmInputDesc->getAddress() + ga_input_offset;

  std::vector<uint32_t> gm_input_strides = gmInputDesc->getStrides();
  cvk_tg_stride_t cvk_gm_input_stride = {gm_input_strides[NGCHW::N],
                                         gm_input_strides[NGCHW::C],
                                         gm_input_strides[NGCHW::H]};

  cvk_tl_shape_t tl_shape = {
      cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
      cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
  cvk_tl_t tl_load;
  CV18xx::lmem_init_tensor(&tl_load, tl_shape,
                           lmInputDescs[lmIndex]->getDataFormat(),
                           lmInputDescs[lmIndex]->getEuAlign());
  tl_load.start_address = lmInputDescs[lmIndex]->getAddress();

  // Input is not altered, use actual shape/stride.
  if (args.do_ic_alignment) {
    tl_load.shape.c -= 1;
    tl_load.stride =
        CV18xx::tl_default_stride(tl_load.shape, tl_load.fmt, tl_load.eu_align);
  }

  // LLVM_DEBUG(llvm::dbgs()
  //            << "\n  [n_pos=" << gmOutputPoss[NGCHW::N] << "][ig="
  //            << gmOutputPoss[NGCHW::G] << "][oc_pos=" <<
  //            gmOutputPoss[NGCHW::C]
  //            << "][oh_pos=" << gmOutputPoss[NGCHW::H]
  //            << "][ow_pos=" << gmOutputPoss[NGCHW::W] << "][ic_pos=" << icPos
  //            << "] new tdma_load_stride:\n"
  //            << "    tl_ifmap gaddr " << llvm::format_hex(ga_input_load, 10)
  //            << "(offset=" << llvm::format_hex(ga_input_offset, 10)
  //            << "), gstride (" << cvk_gm_input_stride.n << ", "
  //            << cvk_gm_input_stride.c << ", " << cvk_gm_input_stride.h <<
  //            ")\n"
  //            << "    laddr " << llvm::format_hex(tl_load.start_address, 10)
  //            << ", shape (" << tl_load.shape.n << ", " << tl_load.shape.c
  //            << ", " << tl_load.shape.h << ", " << tl_load.shape.w
  //            << "), stride (" << tl_load.stride.n << ", " << tl_load.stride.c
  //            << ", " << tl_load.stride.h << ", " << tl_load.stride.w
  //            << ")\n\n");

  // Load uncompressed input
  if ((args.input_fmt == CVK_FMT_I8) && (args.tiu_fmt == CVK_FMT_I8)) {
    if (tl_load.shape.h)
      CV18xx::tdma_load_stride(&tl_load, ga_input_load, cvk_gm_input_stride);
    else
      fillConstantLmInput(&tl_load, cur_gm_input_paddings);
  } else if ((args.input_fmt == CVK_FMT_BF16) &&
             (args.tiu_fmt == CVK_FMT_BF16)) {
    if (tl_load.shape.h)
      CV18xx::tdma_load_stride(&tl_load, ga_input_load, cvk_gm_input_stride);
    else
      fillConstantLmInput(&tl_load, cur_gm_input_paddings);
  } else {
    assert(0 && "Input only supports i8/bf16");
  }

  bool ignoreOutputChannel = false;
  if ((tilePolicy == ReuseActivationPolicyType) ||
      (tilePolicy == SingleBufferPolicyType))
    ignoreOutputChannel = true;

  cModelDebug.recordInput(args.layer_id, gmOutputPoss, ga_input_load,
                          ga_input_offset, cur_gm_input_poss,
                          cur_gm_input_shapes, ignoreOutputChannel);
}

uint32_t Conv::getPs32Mode(uint32_t icPos) {
  // dw-conv supports direct ps32 output, but not ps32 mode
  if (isDwConv())
    return args.ps32_output ? 2 : 0;

  // Normal mode
  if (tile_info.ic_step == group_input_channels())
    return 0;

  // write 32b result at the first time
  if (icPos == 0)
    return 2;

  // load previous 32b result
  if ((icPos + tile_info.ic_step) >= group_input_channels())
    return 1;

  // init & write 32bits partial sum
  return 3;
}

bool Conv::getReluAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  bool reluAllowed = ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
  return (args.fused_conv_relu && reluAllowed);
}

bool Conv::getBiasAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  bool biasAllowed = ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
  return ((args.do_bias || args.do_chl_quan) && biasAllowed);
}

bool Conv::getRshiftAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  return ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
}

// Disable cmd-pre-exe if compressed output is enabled since compressed output
// is divided into multiple tdma stores.
uint8_t Conv::getTdmaLoadWeightIntraCmdParal(uint32_t cmdQueueIndex) {

  uint8_t intraCmdParal = 0;
  if (CV18xx::has_cmd_pre_exe() && cmdQueueIndex < cmdQueue.size())
    intraCmdParal = cmdQueue[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  return intraCmdParal;
}

uint8_t Conv::getTdmaStoreOutputIntraCmdParal(uint32_t cmdQueueIndex) {

  uint8_t intraCmdParal = 0;
  if (CV18xx::has_cmd_pre_exe() && cmdQueueIndex < cmdQueue.size() &&
      !args.do_leaky_relu)
    intraCmdParal = cmdQueue[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  return intraCmdParal;
}

// Hardware constraint:
//   if (des_ps32_md>=2)  {des_cmd_pre_exe <= 1};
//
//   The final stage of ps32 mode generates int8 result.
//   Only enable early-store of cmd-pre-exe for no-ps32 mode or final stage of
//   ps32 mode
//
uint8_t Conv::getTiuCmdPreExeMode(uint32_t cmdQueueIndex, uint32_t icPos) {

  uint8_t cmdPreExeMode = 0;
  if (CV18xx::has_cmd_pre_exe() && cmdQueueIndex < cmdQueue.size() &&
      cmdQueue[cmdQueueIndex]->isIntraCmdParalEnabled()) {
    cmdPreExeMode = 1; // bit[0]: load

    if (getPs32Mode(icPos) <= 1 && !args.do_leaky_relu)
      cmdPreExeMode += 2; // bit[1]: store
  }

  return cmdPreExeMode;
}

void Conv::loadScaleLutTable(uint32_t lmIndex, uint32_t cmdQueueIndex) {
  cvk_tl_t *tl_lut = lmPreProcessDescs[lmIndex]->getAllocated();
  cvi_backend_tl_load(args.layer_id, tl_lut->start_address,
                      gmScaleLutDesc->getAddress(), args.input_fmt,
                      tl_lut->shape.n, tl_lut->shape.c, tl_lut->shape.h,
                      tl_lut->shape.w);

  LLVM_DEBUG(llvm::dbgs() << "  loadScaleLut\n    "
                          << "src " << gmScaleLutDesc->getAddress()
                          << ", shape (" << tl_lut->shape.n << ", "
                          << tl_lut->shape.c << ", " << tl_lut->shape.h << ", "
                          << tl_lut->shape.w << ")\n");
}

void Conv::computeConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                       cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                       std::vector<uint32_t> &cur_gm_input_paddings,
                       uint8_t cmdPreExeMode, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);
  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.chl_quan_param = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.has_bias = getBiasAllowed(icPos) ? args.do_bias : 0;
  param.relu_enable = getReluAllowed(icPos);
  param.ps32_mode = getPs32Mode(icPos);
  param.w_is_const = 0;
  param.layer_id = args.layer_id;
  param.cmd_pre_exe_typ = cmdPreExeMode ? 1 : 0; // wait weight
  param.cmd_pre_exe = cmdPreExeMode;
  param.ins_val = pad_value();
  param.ins_fp = CV18xx::convert_fp32_to_bf16((float)pad_value());
  CV18xx::tiu_convolution(&param);
}

void Conv::computePerTensorConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                                cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                                std::vector<uint32_t> &cur_gm_input_paddings,
                                uint8_t cmdPreExeMode, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);
  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_pt_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.bias = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.relu_enable = getReluAllowed(icPos) ? 1 : 0;
  param.rshift_bits = getRshiftAllowed(icPos) ? args.right_shift_width : 0;
  param.ps32_mode = getPs32Mode(icPos);
  param.w_is_const = 0;
  param.layer_id = args.layer_id;
  param.cmd_pre_exe_typ = cmdPreExeMode ? 1 : 0; // wait weight
  param.cmd_pre_exe = cmdPreExeMode;
  param.ins_val = pad_value();
  param.ins_fp = CV18xx::convert_fp32_to_bf16(float(pad_value()));

  CV18xx::tiu_pt_convolution(&param);
}

void Conv::computeDwConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                         cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                         std::vector<uint32_t> &cur_gm_input_paddings,
                         uint8_t cmdPreExeMode, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);
  adjustComputeForPs32Output(tl_output);
  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_depthwise_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.chl_quan_param = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.has_bias = getBiasAllowed(icPos) ? args.do_bias : 0;
  param.relu_enable = getReluAllowed(icPos);
  param.layer_id = args.layer_id;
  param.cmd_pre_exe_typ = cmdPreExeMode ? 1 : 0; // wait weight
  param.cmd_pre_exe = cmdPreExeMode;
  param.ins_val = pad_value();
  param.ins_fp = CV18xx::convert_fp32_to_bf16((float)pad_value());
  CV18xx::tiu_depthwise_convolution(&param);
}

void Conv::computePerTensorDwConv(cvk_tl_t *tl_output, cvk_tl_t *tl_input,
                                  cvk_tl_t *tl_weight, cvk_tl_t *tl_bias,
                                  std::vector<uint32_t> &cur_gm_input_paddings,
                                  uint8_t cmdPreExeMode, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);
  adjustComputeForPs32Output(tl_output);
  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_depthwise_pt_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.bias = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.relu_enable = getReluAllowed(icPos) ? 1 : 0;
  param.rshift_bits = getRshiftAllowed(icPos) ? args.right_shift_width : 0;
  param.ps32_mode = getPs32Mode(icPos);
  param.layer_id = args.layer_id;
  param.cmd_pre_exe_typ = cmdPreExeMode ? 1 : 0; // wait weight
  param.cmd_pre_exe = cmdPreExeMode;
  param.ins_val = pad_value();
  param.ins_fp = CV18xx::convert_fp32_to_bf16(float(pad_value()));

  CV18xx::tiu_pt_depthwise_convolution(&param);
}

void Conv::computeLeakyRelu(cvk_tl_t *tl_output) {
  cvk_tl_t tl_neg;
  CV18xx::lmem_init_tensor(&tl_neg, tl_output->shape, tl_output->fmt,
                           tl_output->eu_align);
  tl_neg.start_address = lmFusedActDescs[0]->getAddress();

  cvk_tl_t tl_relu;
  CV18xx::lmem_init_tensor(&tl_relu, tl_output->shape, tl_output->fmt,
                           tl_output->eu_align);
  tl_relu.start_address = lmFusedActDescs[1]->getAddress();

  bool isIgnorePosPart = (args.activation_gt_scale == 1);
  bool isSlopeSmallerThanOne =
      ((args.activation_le_scale >> args.activation_le_rshift) == 0);

  if (isIgnorePosPart && args.activation_le_scale >= 0) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_relu;
    p4.a = tl_output;
    p4.b_const.val = args.activation_le_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = args.activation_le_rshift;
    p4.layer_id = args.layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p1 = {0};
      p1.max = tl_output;
      p1.a = tl_output;
      p1.b = &tl_relu;
      p1.b_is_const = 0;
      p1.layer_id = args.layer_id;
      CV18xx::tiu_max(&p1);
    } else {
      cvk_tiu_min_param_t p1 = {0};
      p1.min = tl_output;
      p1.a = tl_output;
      p1.b = &tl_relu;
      p1.b_is_const = 0;
      p1.layer_id = args.layer_id;
      CV18xx::tiu_min(&p1);
    }
  } else {
    cvk_tiu_max_param_t p1 = {0};
    p1.max = &tl_relu;
    p1.a = tl_output;
    p1.b_is_const = 1;
    p1.b_const.is_signed = 1;
    p1.b_const.val = 0;
    p1.layer_id = args.layer_id;
    CV18xx::tiu_max(&p1);
    if (!isIgnorePosPart) {
      cvk_tiu_mul_param_t p2 = {0};
      p2.res_high = nullptr;
      p2.res_low = &tl_relu;
      p2.a = &tl_relu;
      p2.b_const.val = args.activation_gt_scale;
      p2.b_const.is_signed = true;
      p2.b_is_const = 1;
      p2.rshift_bits = args.activation_gt_rshift;
      p2.layer_id = args.layer_id;
      p2.relu_enable = 0;
      CV18xx::tiu_mul(&p2);
    }
    cvk_tiu_min_param_t p3 = {0};
    p3.min = &tl_neg;
    p3.a = tl_output;
    p3.b_is_const = 1;
    p3.b_const.val = 0;
    p3.b_const.is_signed = 1;
    p3.layer_id = args.layer_id;
    CV18xx::tiu_min(&p3);

    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_neg;
    p4.a = &tl_neg;
    p4.b_const.val = args.activation_le_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = args.activation_le_rshift;
    p4.layer_id = args.layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);

    cvk_tiu_or_int8_param_t p5 = {0};
    p5.res = tl_output;
    p5.a = &tl_relu;
    p5.b = &tl_neg;
    p5.layer_id = args.layer_id;
    CV18xx::tiu_or_int8(&p5);
  }
}

// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
// lmIndex: (input, weight, output)
void Conv::compute(std::vector<uint32_t> gmOutputPoss,
                   std::vector<uint32_t> lmIndexes, uint32_t cmdQueueIndex,
                   uint32_t icPos) {
  uint32_t lm_input_index = lmIndexes[0];  // input
  uint32_t lm_weight_index = lmIndexes[1]; // weight
  uint32_t lm_output_index = lmIndexes[2]; // output

  // Input information also used in loadInput()
  // Output for TIU
  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);
  cvk_tl_t tl_output;
  cvk_tl_shape_t tl_output_shape = {
      gmOutputPossShapes[NGCHW::N], gmOutputPossShapes[NGCHW::C],
      gmOutputPossShapes[NGCHW::H], gmOutputPossShapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_output, tl_output_shape,
                           lmOutputDescs[lm_output_index]->getDataFormat(),
                           lmOutputDescs[lm_output_index]->getEuAlign());
  tl_output.start_address = lmOutputDescs[lm_output_index]->getAddress();

  // Input for TIU
  std::vector<uint32_t> cur_gm_input_poss;
  std::vector<uint32_t> cur_gm_input_shapes;
  std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
  getTiledGmPossAndShapesOfInputForTiu(gmOutputPoss, gmOutputPossShapes,
                                       cur_gm_input_poss, cur_gm_input_shapes,
                                       cur_gm_input_paddings, icPos);

  cvk_tl_t tl_input;
  cvk_tl_shape_t tl_input_shape = {
      cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
      cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_input, tl_input_shape,
                           lmInputDescs[lm_input_index]->getDataFormat(),
                           lmInputDescs[lm_input_index]->getEuAlign());
  tl_input.start_address = lmInputDescs[lm_input_index]->getAddress();

  // Bias for TIU
  std::vector<uint32_t> bias_shapes;
  cvk_tl_t tl_bias = {0};
  cvk_tl_shape_t tl_bias_shape = {0};
  if (getBiasAllowed(icPos)) {
    bias_shapes = getTiledLmShapesOfBiasForTiu(gmOutputPoss);
    tl_bias_shape = {bias_shapes[NGCHW::N], bias_shapes[NGCHW::C],
                     bias_shapes[NGCHW::H], bias_shapes[NGCHW::W]};
    CV18xx::lmem_init_tensor(&tl_bias, tl_bias_shape,
                             lmBiasDescs[lm_weight_index]->getDataFormat(),
                             lmBiasDescs[lm_weight_index]->getEuAlign());
    tl_bias.start_address = lmBiasDescs[lm_weight_index]->getAddress();
  }

  // Weight for TIU, shapes (1, ic, tiledOc, kh, kw)
  //
  std::vector<uint32_t> weight_shapes =
      getTiledLmShapesOfWeightForTiu(gmOutputPoss, icPos);
  cvk_tl_t tl_weight;
  cvk_tl_shape_t tl_weight_shape = {
      weight_shapes[NGCHW::G], weight_shapes[NGCHW::C], weight_shapes[NGCHW::H],
      weight_shapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_weight, tl_weight_shape,
                           lmWeightDescs[lm_weight_index]->getDataFormat(),
                           lmWeightDescs[lm_weight_index]->getEuAlign());
  tl_weight.start_address = lmWeightDescs[lm_weight_index]->getAddress();

  uint8_t cmdPreExeMode = getTiuCmdPreExeMode(cmdQueueIndex, icPos);

  // LLVM_DEBUG(
  //     llvm::dbgs()
  //     << "    compute\n"
  //     << "      ifmap laddr " << llvm::format_hex(tl_input.start_address, 10)
  //     << ", shape (" << tl_input.shape.n << ", " << tl_input.shape.c << ", "
  //     << tl_input.shape.h << ", " << tl_input.shape.w << ")\n"
  //     << "      weight laddr " << llvm::format_hex(tl_weight.start_address,
  //     10)
  //     << ", shape (" << tl_weight.shape.n << ", " << tl_weight.shape.c << ",
  //     "
  //     << tl_weight.shape.h << ", " << tl_weight.shape.w << ")\n"
  //     << "      bias laddr " << llvm::format_hex(tl_bias.start_address, 10)
  //     << ", shape (" << tl_bias.shape.n << ", " << tl_bias.shape.c << ", "
  //     << tl_bias.shape.h << ", " << tl_bias.shape.w << ")\n"
  //     << "      ofmap laddr " << llvm::format_hex(tl_output.start_address,
  //     10)
  //     << ", shape (" << tl_output.shape.n << ", " << tl_output.shape.c << ",
  //     "
  //     << tl_output.shape.h << ", " << tl_output.shape.w << ")"
  //     << ", cmdPreExeMode " << (int)cmdPreExeMode << ", ps32mode "
  //     << (int)getPs32Mode(icPos) << "\n");

  // Use LayerId as trigger point
  uint32_t originalLayerId = args.layer_id;
  cModelDebug.updateLayerId(args.layer_id, gmOutputPoss);

  if (isDwConv()) {
    if (args.do_chl_quan)
      computeDwConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                    cur_gm_input_paddings, cmdPreExeMode, icPos);
    else
      computePerTensorDwConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                             cur_gm_input_paddings, cmdPreExeMode, icPos);
  } else {
    if (args.do_chl_quan && (getPs32Mode(icPos) <= 1))
      computeConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                  cur_gm_input_paddings, cmdPreExeMode, icPos);
    else
      computePerTensorConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                           cur_gm_input_paddings, cmdPreExeMode, icPos);
  }

  // Restore LayerId
  args.layer_id = originalLayerId;

  if (args.do_leaky_relu)
    computeLeakyRelu(&tl_output);
}

void Conv::computeScaleLut(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                           uint32_t cmdQueueIndex, uint32_t icPos) {

  cvk_tl_t *tl_lut = lmPreProcessDescs[0]->getAllocated();

  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);

  std::vector<uint32_t> cur_gm_input_poss;
  std::vector<uint32_t> cur_gm_input_shapes;
  std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
  getTiledGmPossAndShapesOfInputForTiu(gmOutputPoss, gmOutputPossShapes,
                                       cur_gm_input_poss, cur_gm_input_shapes,
                                       cur_gm_input_paddings, icPos);

  cvk_tl_shape_t tl_shape = {
      cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
      cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
  cvk_tl_t tl_load;
  CV18xx::lmem_init_tensor(&tl_load, tl_shape,
                           lmInputDescs[lmIndex]->getDataFormat(),
                           lmInputDescs[lmIndex]->getEuAlign());
  tl_load.start_address = lmInputDescs[lmIndex]->getAddress();

  // Input is not altered, use actual shape/stride.
  if (args.do_ic_alignment) {
    tl_load.shape.c -= 1;
    tl_load.stride =
        CV18xx::tl_default_stride(tl_load.shape, tl_load.fmt, tl_load.eu_align);
  }

  cvk_tiu_lookup_table_param_t p = {0};
  p.ofmap = &tl_load; // in-place update
  p.ifmap = &tl_load;
  p.table = tl_lut;
  p.layer_id = args.layer_id;
  CV18xx::tiu_lookup_table(&p);

  LLVM_DEBUG(llvm::dbgs() << "  computeScaleLut \n    "
                          << "ifmap " << tl_load.start_address << ", shape ("
                          << tl_load.shape.n << ", " << tl_load.shape.c << ", "
                          << tl_load.shape.h << ", " << tl_load.shape.w
                          << "), stride (" << tl_load.stride.n << ", "
                          << tl_load.stride.c << ", " << tl_load.stride.h
                          << ", " << tl_load.stride.w << ")\n    "
                          << "table " << tl_load.start_address << ", shape ("
                          << tl_load.shape.n << ", " << tl_lut->shape.c << ", "
                          << tl_load.shape.h << ", " << tl_lut->shape.w
                          << ")\n");
}

void Conv::computeQuant(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                        uint32_t cmdQueueIndex, uint32_t icPos) {
  // init weight
  auto weight_shapes = getTiledLmShapesOfWeightForTiu(gmOutputPoss, icPos);
  cvk_tl_t tl_weight;
  cvk_tl_shape_t tl_weight_shape = {
      weight_shapes[NGCHW::G], weight_shapes[NGCHW::C], weight_shapes[NGCHW::H],
      weight_shapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_weight, tl_weight_shape,
                           lmWeightDescs[lmIndex]->getDataFormat(),
                           lmWeightDescs[lmIndex]->getEuAlign());
  tl_weight.start_address = lmWeightDescs[lmIndex]->getAddress();

  // init quant
  cvk_tl_t tl_quant[2] = {0};
  cvk_tl_shape_t tl_quant_shape = {0};
  auto quant_shapes = getTiledLmShapesOfQuantForTiu(gmOutputPoss);
  tl_quant_shape = {quant_shapes[NGCHW::N], quant_shapes[NGCHW::C],
                    quant_shapes[NGCHW::H], quant_shapes[NGCHW::W]};
  for (int i = 0; i < 2; i++) {
    CV18xx::lmem_init_tensor(&tl_quant[i], tl_quant_shape,
                             lmQuantDescs[i + 2 * lmIndex]->getDataFormat(),
                             lmQuantDescs[i + 2 * lmIndex]->getEuAlign());
    tl_quant[i].start_address = lmQuantDescs[i + 2 * lmIndex]->getAddress();
    tl_quant[i].shape = tl_weight.shape;
    tl_quant[i].stride.n = 0;
    tl_quant[i].stride.h = 0;
    tl_quant[i].stride.w = 0;
  }
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_weight;
  p.a = &tl_weight;
  p.b_is_const = 0;
  p.b = &tl_quant[0];
  p.rshift_bits = 0;
  p.layer_id = args.layer_id;
  p.relu_enable = 0;
  CV18xx::tiu_mul(&p);

  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &tl_weight;
  p1.a_high = nullptr;
  p1.a_low = &tl_weight;
  p1.b_is_const = false;
  p1.b.high = nullptr;
  p1.b.low = &tl_quant[1];
  p1.rshift_bits = 0;
  p1.layer_id = args.layer_id;
  p1.relu_enable = 0;
  CV18xx::tiu_add(&p1);
}

void Conv::storeOutput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
                       uint32_t cmdQueueIndex) {
  assert(gmOutputPoss.size() == 5 && "Expect 5D tensor");

  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);

  uint64_t ga_output_offset = gmOutputDesc->getCurrentOffset(gmOutputPoss);
  uint64_t ga_output_store = gmOutputDesc->getAddress() + ga_output_offset;

  std::vector<uint32_t> gm_output_strides = gmOutputDesc->getStrides();
  cvk_tg_stride_t cvk_gm_output_stride = {gm_output_strides[NGCHW::N],
                                          gm_output_strides[NGCHW::C],
                                          gm_output_strides[NGCHW::H]};

  cvk_tl_t tl_output;
  cvk_tl_shape_t tl_output_shape = {
      gmOutputPossShapes[NGCHW::N], gmOutputPossShapes[NGCHW::C],
      gmOutputPossShapes[NGCHW::H], gmOutputPossShapes[NGCHW::W]};
  CV18xx::lmem_init_tensor(&tl_output, tl_output_shape,
                           lmOutputDescs[lmIndex]->getDataFormat(),
                           lmOutputDescs[lmIndex]->getEuAlign());
  tl_output.start_address = lmOutputDescs[lmIndex]->getAddress();

  uint8_t intraCmdParal = getTdmaStoreOutputIntraCmdParal(cmdQueueIndex);
  if (CV18xx::has_cmd_pre_exe() && cmdQueueIndex < cmdQueue.size() &&
      !args.do_leaky_relu)
    intraCmdParal = cmdQueue[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  // LLVM_DEBUG(llvm::dbgs()
  //            << "  [n_pos=" << gmOutputPoss[NGCHW::N] << "][ig="
  //            << gmOutputPoss[NGCHW::G] << "][oc_pos=" <<
  //            gmOutputPoss[NGCHW::C]
  //            << "][oh_pos=" << gmOutputPoss[NGCHW::H] << "][ow_pos="
  //            << gmOutputPoss[NGCHW::W] << "] new tdma_store_stride:\n"
  //            << "    tl_ofmap gaddr " << llvm::format_hex(ga_output_store,
  //            10)
  //            << "(offset=" << llvm::format_hex(ga_output_offset, 10)
  //            << "), gstride (" << cvk_gm_output_stride.n << ", "
  //            << cvk_gm_output_stride.c << ", " << cvk_gm_output_stride.h
  //            << ")\n"
  //            << "    laddr " << llvm::format_hex(tl_output.start_address, 10)
  //            << ", shape (" << tl_output.shape.n << ", " << tl_output.shape.c
  //            << ", " << tl_output.shape.h << ", " << tl_output.shape.w <<
  //            ")\n"
  //            << "    intraCmdParal " << (int)intraCmdParal << "\n");

  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(gmOutputDesc->getAddress());
  ts_data.fmt = tl_output.fmt;
  ts_data.start_address = ga_output_store;
  ts_data.shape = {tl_output.shape.n, tl_output.shape.c, tl_output.shape.h,
                   tl_output.shape.w};
  ts_data.stride = cvk_gm_output_stride;

  if (!args.ps32_output) {
    // Store normal output
    cvk_tdma_l2g_tensor_copy_param_t param = {0};
    param.src = &tl_output;
    param.dst = &ts_data;
    param.intra_cmd_paral = intraCmdParal ? 1 : 0;
    param.layer_id = args.layer_id;
    CV18xx::tdma_l2g_tensor_copy(&param);
  } else {
    adjustStoreForPs32Output(&tl_output, &ts_data, ga_output_offset);

    cvi_backend_tl_bf16_ps32_to_fp32(args.layer_id, tl_output.start_address,
                                     tl_output.shape.n, tl_output.shape.c,
                                     tl_output.shape.h, tl_output.shape.w);

    cvi_backend_tl_store_fp32(args.layer_id, ts_data.start_address,
                              tl_output.start_address, tl_output.shape.n,
                              tl_output.shape.c, tl_output.shape.h,
                              tl_output.shape.w);
  }

  cModelDebug.recordOutput(args.layer_id, gmOutputPoss, ga_output_store,
                           ga_output_offset, gmOutputPossShapes);
}

bool Conv::isDwConv() {
  if ((groups() != 1) && (group_input_channels() == 1) &&
      (group_output_channels() == 1))
    return true;

  return false;
}

bool Conv::isConvPs32() {
  return (tile_info.ic_step != group_input_channels()) ? true : false;
}

// I try to maximize the local memory utilization,
// but it causes large write latency, especially in cross-layer.
// However TDMA engine can handle small data transfer efficiently.
//
// E.g. Resnet50 scale2b_branch2c in DDR3 platform.
//   (1, 96, 56, 56) tiu 19471, store 31056, 77 fps
//   (1, 32, 56, 56) tiu 6535, store 10376, 84 fps
//
// The load/store reorder may be useful in intra-layer and
// inter-layer.
//
// The next-generation chip will do DMA store once intermediate
// result is generated.
//
// The following is temporary solution.
// I decrease the output channel size to trigger frequent DMA store.
// So local memory is wasted.
bool Conv::checkDmaPolicy(TileInfo &tileInfo) {
  if (!tileInfo.favor_dma)
    return true;

  // DMA efficiency: OH * OW >= 256B
  const int dma_min_size = 256;
  int ofmap_plane_size = tileInfo.oh_step * tileInfo.ow_step;

  if ((tileInfo.oc_step > (uint32_t)CV18xx::NPU_NUM) &&
      (ofmap_plane_size > (1 * dma_min_size))) {
    return false;
  }
  if ((tileInfo.oc_step > (2 * (uint32_t)CV18xx::NPU_NUM)) &&
      (ofmap_plane_size < dma_min_size)) {
    // even oh*ow is smaller, use at most 2xlanes_num
    return false;
  }

  return true;
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Not split ic since it needs 32b ofmap for partial sum.
bool Conv::determineTileSize(bool useDoubleBuffer, bool favor_dma) {
  int32_t input_n = args.input_n;
  int32_t input_c = args.input_c;
  int32_t input_h = args.input_h;
  int32_t input_w = args.input_w;
  int32_t groups = args.groups;
  int32_t output_c = args.output_c;
  int32_t do_bias = args.do_bias;
  bool do_chl_quan = args.do_chl_quan;
  int32_t do_activation = args.do_activation;
  float *activation_arg = args.activation_arg;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  uint16_t dilation_h = args.dilation_h;
  uint16_t dilation_w = args.dilation_w;
  uint8_t pad_top = args.pad_top;
  uint8_t pad_bottom = args.pad_bottom;
  uint8_t pad_left = args.pad_left;
  uint8_t pad_right = args.pad_right;
  uint8_t stride_h = args.stride_h;
  uint8_t stride_w = args.stride_w;

  int32_t ic = input_c / groups;
  int32_t oc = output_c / groups;
  int32_t kh_extent = dilation_h * (kh - 1) + 1;
  int32_t kw_extent = dilation_w * (kw - 1) + 1;
  int32_t oh =
      (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h +
      1;
  int32_t ow =
      (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w +
      1;
  int32_t ih = input_h;
  int32_t iw = input_w;
  int32_t n = input_n;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  int32_t npu_num = static_cast<int32_t>(CV18xx::NPU_NUM);
  tile_info.n = 1;
  tile_info.oc = ceiling_func(oc, npu_num); // lane parallelism
  tile_info.ic = 1;
  tile_info.h = (ih + (MAX_HEIGHT - 1)) / MAX_HEIGHT;
  tile_info.w = (iw + (MAX_WIDTH - 1)) / MAX_WIDTH;

  int32_t num_oc_step = (oc + npu_num - 1) / npu_num;
  uint32_t ic_step =
      std::min(group_input_channels(), static_cast<uint32_t>(MAX_TIU_CHL));

  // Not handle ps32 tiling here.
  if (ic_step < group_input_channels()) {
    LLVM_DEBUG(llvm::errs() << "  determineTileSize fail\n");
    return false;
  }

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;
  uint32_t scaleLutSize = 0;
  if (args.do_scale_lut) {
    cvk_tl_shape_t shape = CV18xx::lut_table_shape(args.input_fmt);
    scaleLutSize =
        CV18xx::lmem_tensor_to_size(shape, args.input_fmt, /*eu_align=*/1);
  }
  int32_t max_oh = std::min(oh, MAX_HEIGHT);
  int32_t max_ow = std::min(ow, MAX_WIDTH);
  int32_t min_ow = std::min((int)kw, max_ow);
  // Split ow
  for (int32_t ow_step = max_ow; ow_step >= min_ow; --ow_step) {
    int32_t iw_step =
        ceiling_func((ow_step - 1) * stride_w + kw_extent, 1 + insert_width());
    iw_step = std::min(iw_step, iw);

    if ((stride_w > 1) && ((iw_step + stride_w) > iw)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, iw);
      tile_info.iw_step = iw_step;
    }

    if (w_after_ins_pad(iw_step) > MAX_WIDTH) {
      continue;
    }

    // Split oh
    int32_t oh_step = max_oh;
    while (oh_step > 0) {
      // int32_t oh_step = ceiling_func(oh, tile_info.h);
      int32_t ih_step = ceiling_func((oh_step - 1) * stride_h + kh_extent,
                                     1 + insert_height());
      ih_step = std::min(ih_step, ih);

      if (h_after_ins_pad(ih_step) <= MAX_HEIGHT) {

        // Split oc
        for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc) {
          // Downward, align lanes
          //   E.g. oc = 48, oc_step: 48, 32
          int32_t npu_num = static_cast<int32_t>(CV18xx::NPU_NUM);
          int32_t oc_step = std::min((num_oc_step - slice_oc) * npu_num, oc);
          if (args.do_quant) {
            oc_step = std::min(oc_step, npu_num);
          }

          // We may need to put EU-alignment info in one place
          cvk_tl_shape_t coeff_shape_i16 =
              CV18xx::tl_shape_t4(2, oc_step, 1, 1);

          uint32_t coeff_oc_step_size = 0;

          if (do_chl_quan) {
            uint32_t pc_bias_size = CV18xx::chan_quan_param_size(args.do_bias);
            cvk_tl_shape_t coeff_shape =
                CV18xx::tl_shape_t4(1, oc_step, 1, pc_bias_size);
            coeff_oc_step_size +=
                CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                            /*eu_align=*/0);
          } else if (do_bias) {
            // 16 bit
            coeff_oc_step_size += CV18xx::lmem_tensor_to_size(
                coeff_shape_i16, args.tiu_fmt, /*eu_align=*/0);
          }
          if (args.do_quant) {
            auto coeff_shape = CV18xx::tl_shape_t4(1, oc_step, 1, 1);
            coeff_oc_step_size +=
                2 * CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                                /*eu_align=*/0);
          }

          // Add weight size
          uint32_t weight_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(ic_step, oc_step, kh, kw), args.tiu_fmt,
              /*eu_align=*/0);

          // split n
          for (tile_info.n = 1; tile_info.n <= n; ++tile_info.n) {
            int32_t n_step = ceiling_func(n, tile_info.n);

            uint32_t ofmap_size = CV18xx::lmem_tensor_to_size(
                CV18xx::tl_shape_t4(n_step, oc_step, oh_step, ow_step),
                args.tiu_fmt,
                /*eu_align=*/1);

            uint32_t ifmap_size = CV18xx::lmem_tensor_to_size(
                CV18xx::tl_shape_t4(n_step, ic_step, ih_step, iw_step),
                args.tiu_fmt,
                /*eu_align=*/1);

            // Leaky relu need tl_neg, tl_relu.
            // tl_relu, tl_neg are not from tmda and not final output.
            // One copy is enough.
            uint32_t extra_size = scaleLutSize;
            if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
              extra_size += 2 * ofmap_size; // tl_relu + tl_neg
            }

            uint32_t total_needed =
                (ifmap_size + ofmap_size + weight_size + coeff_oc_step_size) *
                    bufferMultiplier +
                extra_size;

            tile_info.n_step = n_step;
            tile_info.oc_step = oc_step;
            tile_info.oh_step = oh_step;
            tile_info.ow_step = ow_step;
            tile_info.ih_step = ih_step;
            tile_info.iw_step = iw_step;
            tile_info.ic_step = ic_step;
            tile_info.total_needed = total_needed;
            tile_info.favor_dma = favor_dma;

            if (total_needed <= CV18xx::LMEM_BYTES &&
                checkDmaPolicy(tile_info)) {
              uint32_t total_size =
                  ic * ih * iw + oc * ic * kh * kw + oc * oh * ow;
              uint32_t ut = (total_needed * 100) / CV18xx::LMEM_BYTES;
              bool is_tiled =
                  ((oc != oc_step) || (oh != oh_step)) ? true : false;
              const char *ut_msg =
                  (ut < 70 && is_tiled && !favor_dma) ? " => NG" : "";

              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  Conv::determineTileSize\n    "
                             "layer_id %d\n    "
                             "groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, "
                             "%d, %d)\n    "
                             "kernel (%d, %d), pad (top=%d, bot=%d, left=%d, "
                             "right=%d)\n    "
                             "stride (%d, %d), dilation (%d, %d)\n    "
                             "useDoubleBuffer %d\n",
                             args.layer_id, groups, input_n, input_c, input_h,
                             input_w, input_n, oc, oh, ow, kh, kw, pad_top,
                             pad_bottom, pad_left, pad_right, stride_h,
                             stride_w, dilation_h, dilation_w,
                             useDoubleBuffer));
              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
                      "    Tile (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d"
                      ", ih_step=%d, iw_step=%d, ic_step=%d)\n    "
                      "inputSize %d, outputSize %d, weightSize %d"
                      ", biasSize %d, totalSizePerLane %d/%d"
                      ", totalSize %d/%d(%d), %d/%d(%d)%s\n    "
                      "ifmap shape (%d, %d, %d, %d)\n    "
                      "weight shape (%d, %d, %d, %d)\n    "
                      "ofmap shape (%d, %d, %d, %d)\n    "
                      "useDoubleBuffer %d, favor_dma %d\n",
                      n_step, oc_step, oh_step, ow_step, ih_step, iw_step,
                      ic_step, ifmap_size * bufferMultiplier,
                      ofmap_size * bufferMultiplier,
                      weight_size * bufferMultiplier,
                      coeff_oc_step_size * bufferMultiplier, total_needed,
                      CV18xx::LMEM_BYTES, total_needed * CV18xx::NPU_NUM,
                      CV18xx::LMEM_BYTES * CV18xx::NPU_NUM,
                      (total_needed * 100) / CV18xx::LMEM_BYTES, total_needed,
                      total_size, (total_needed * 100) / total_size, ut_msg,
                      n_step, ic_step, ih_step, iw_step, oc_step, ic_step, kh,
                      kw, n_step, oc_step, oh_step, ow_step, useDoubleBuffer,
                      favor_dma));
              return true;
            }

          } // for (tile_info.n = 1; tile_info.n < n; ++tile_info.n)
        }   // for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc)
      }
      if (ow_step < max_ow) {
        // When the width tiling is used, there is no need to do height tiling.
        break;
      }
      oh_step--;
    } // for (tile_info.h = 1; tile_info.h <= oh; ++tile_info.h)
  }   // for (tile_info.w = 1; tile_info.w <= ow; ++tile_info.ow)

  tile_info = {0};

  LLVM_DEBUG(llvm::errs() << "  determineTileSize fail\n");

  return false;
}

bool Conv::determinePs32TileSize(bool useDoubleBuffer) {
  int32_t input_n = args.input_n;
  int32_t input_c = args.input_c;
  int32_t input_h = args.input_h;
  int32_t input_w = args.input_w;
  int32_t groups = args.groups;
  int32_t output_c = args.output_c;
  int32_t do_bias = args.do_bias;
  bool do_chl_quan = args.do_chl_quan;
  int32_t do_activation = args.do_activation;
  float *activation_arg = args.activation_arg;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  uint16_t dilation_h = args.dilation_h;
  uint16_t dilation_w = args.dilation_w;
  uint8_t pad_top = args.pad_top;
  uint8_t pad_bottom = args.pad_bottom;
  uint8_t pad_left = args.pad_left;
  uint8_t pad_right = args.pad_right;
  uint8_t stride_h = args.stride_h;
  uint8_t stride_w = args.stride_w;

  int32_t ic = input_c / groups;
  int32_t oc = output_c / groups;
  int32_t kh_extent = dilation_h * (kh - 1) + 1;
  int32_t kw_extent = dilation_w * (kw - 1) + 1;
  int32_t oh =
      (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h +
      1;
  int32_t ow =
      (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w +
      1;
  int32_t ih = input_h;
  int32_t iw = input_w;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "  determinePs32TileSize =>\n"
          "    layer_id %d\n"
          "    groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
          "    kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
          "    stride (%d, %d), dilation (%d, %d)\n"
          "    useDoubleBuffer %d\n",
          args.layer_id, groups, input_n, input_c, input_h, input_w, input_n,
          oc, oh, ow, kh, kw, pad_top, pad_bottom, pad_left, pad_right,
          stride_h, stride_w, dilation_h, dilation_w, useDoubleBuffer));

  int32_t npu_num = static_cast<int32_t>(CV18xx::NPU_NUM);
  tile_info.n = 1;
  tile_info.oc = ceiling_func(oc, npu_num); // lane parallelism
  tile_info.ic = 1;
  tile_info.h = (ih + (MAX_HEIGHT - 1)) / MAX_HEIGHT;
  tile_info.w = (iw + (MAX_WIDTH - 1)) / MAX_WIDTH;

  uint32_t max_ic_step =
      std::min(group_input_channels(), static_cast<uint32_t>(MAX_TIU_CHL));

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;

  int32_t n_step = 1;
  int32_t oc_step = std::min(static_cast<int32_t>(group_output_channels()),
                             static_cast<int32_t>(CV18xx::NPU_NUM));
  // Split ow
  for (int32_t ow_step = std::min(ow, MAX_WIDTH); ow_step > 0; --ow_step) {
    // int32_t ow_step = ceiling_func(ow, tile_info.w);
    int32_t iw_step = std::min((ow_step - 1) * stride_w + kw_extent, iw);

    if ((stride_w > 1) && ((iw_step + stride_w) > iw)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, iw);
      tile_info.iw_step = iw_step;
    }
    if (w_after_ins_pad(iw_step) > MAX_WIDTH) {
      continue;
    }

    // Split oh
    // for (tile_info.h = 1; tile_info.h <= oh; ++tile_info.h) {
    for (int32_t oh_step = std::min(oh, MAX_HEIGHT); oh_step > 0; --oh_step) {
      // When the width tiling is used, there is no need to do height tiling.
      if (ow_step < std::min(ow, MAX_WIDTH))
        oh_step = 1;

      // int32_t oh_step = ceiling_func(oh, tile_info.h);
      int32_t ih_step = std::min((oh_step - 1) * stride_h + kh_extent, ih);
      if (h_after_ins_pad(ih_step) > MAX_HEIGHT) {
        continue;
      }

      // We may need to put EU-alignment info in one place
      cvk_tl_shape_t coeff_shape_i16 = CV18xx::tl_shape_t4(2, oc_step, 1, 1);

      uint32_t coeff_oc_step_size = 0;

      if (do_chl_quan) {
        uint32_t pc_bias_size = CV18xx::chan_quan_param_size(args.do_bias);
        cvk_tl_shape_t coeff_shape =
            CV18xx::tl_shape_t4(1, oc_step, 1, pc_bias_size);
        coeff_oc_step_size +=
            CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                        /*eu_align=*/0);
      } else if (do_bias) {
        // 16 bit
        coeff_oc_step_size += CV18xx::lmem_tensor_to_size(
            coeff_shape_i16, args.tiu_fmt, /*eu_align=*/0);
      }

      if (args.do_quant) {
        auto coeff_shape = CV18xx::tl_shape_t4(1, oc_step, 1, 1);
        coeff_oc_step_size +=
            2 * CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                            /*eu_align=*/0);
      }

      // Split ic
      for (int32_t ic_step = max_ic_step; ic_step > npu_num;
           ic_step = align_up(ic_step / 2, npu_num)) {
        uint32_t ofmapSizeMultiplier =
            (ic_step < static_cast<int32_t>(group_input_channels())) ? 4 : 1;

        // Add weight size
        uint32_t weight_size = CV18xx::lmem_tensor_to_size(
            CV18xx::tl_shape_t4(ic_step, oc_step, kh, kw), args.tiu_fmt,
            /*eu_align=*/0);

        uint32_t ofmap_size = CV18xx::lmem_tensor_to_size(
            CV18xx::tl_shape_t4(n_step, oc_step, oh_step, ow_step),
            args.tiu_fmt,
            /*eu_align=*/1);

        uint32_t ifmap_size = CV18xx::lmem_tensor_to_size(
            CV18xx::tl_shape_t4(n_step, ic_step, ih_step, iw_step),
            args.tiu_fmt,
            /*eu_align=*/1);

        uint32_t total_needed = coeff_oc_step_size + weight_size + ifmap_size +
                                ofmap_size * ofmapSizeMultiplier;

        // Double buffers so that TDMA load and store can run during TIU
        // executes.
        total_needed *= bufferMultiplier;

        // Leaky relu need tl_neg, tl_relu.
        // tl_relu, tl_neg are not from tmda and not final output.
        // One copy is enough.
        if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
          total_needed += 2 * ofmap_size; // tl_relu + tl_neg
        }

        if (total_needed <= CV18xx::LMEM_BYTES) {
          LLVM_DEBUG(llvm::dbgs()
                     << "      [n_step=" << n_step << "][oc_step=" << oc_step
                     << "][oh_step=" << oh_step << "][ow_step=" << ow_step
                     << "][ih_step=" << ih_step << "][iw_step=" << iw_step
                     << "][ic_step=" << ic_step << "] total_needed "
                     << total_needed << ", LMEM_SIZE " << CV18xx::LMEM_BYTES
                     << ", bufferMultiplier " << bufferMultiplier
                     << ", ofmapSizeMultiplier " << ofmapSizeMultiplier << "\n"
                     << "        ifmap shape(" << n_step << ", " << ic_step
                     << ", " << ih_step << ", " << iw_step << "), size "
                     << ifmap_size << "\n"
                     << "        weight shape(" << oc_step << ", " << ic_step
                     << ", " << kh << ", " << kw << "), size " << weight_size
                     << "\n"
                     << "        ofmap shape (" << n_step << ", " << oc_step
                     << ", " << oh_step << ", " << ow_step << "), size "
                     << ofmap_size << "\n");

          LLVM_DEBUG(
              llvm::errs() << llvm::format(
                  "    Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d"
                  ", ih_step=%d, iw_step=%d, ic_step=%d)\n"
                  "      coeff_oc_step_size %d, total_needed %d\n"
                  "      ifmap shape (%d, %d, %d, %d)\n"
                  "      weight shape (%d, %d, %d, %d)\n"
                  "      ofmap shape (%d, %d, %d, %d)\n",
                  n_step, oc_step, oh_step, ow_step, ih_step, iw_step, ic_step,
                  coeff_oc_step_size, total_needed, n_step, ic_step, ih_step,
                  iw_step, oc_step, ic_step, kh, kw, n_step, oc_step, oh_step,
                  ow_step));
          LLVM_DEBUG(llvm::errs() << "  <= determinePs32TileSize succeed\n");

          tile_info.n_step = n_step;
          tile_info.oc_step = oc_step;
          tile_info.oh_step = oh_step;
          tile_info.ow_step = ow_step;
          tile_info.ih_step = ih_step;
          tile_info.iw_step = iw_step;
          tile_info.ic_step = ic_step;
          tile_info.total_needed = total_needed;

          return true;
        }

      } // uint32_t ic_step = group_input_channels(); ic_step > 0; --ic_step

    } // for (tile_info.h = 1; tile_info.h <= oh; ++tile_info.h)

  } // for (tile_info.w = 1; tile_info.w <= ow; ++tile_info.ow)

  LLVM_DEBUG(llvm::errs() << "  <= determinePs32TileSize fail\n");

  return false;
}

void Conv::enqueueLoadInputCmd(std::vector<uint32_t> poss, uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadInputCmdType, poss, index));
}

void Conv::enqueueLoadInputCmd(std::vector<uint32_t> poss, uint32_t index,
                               uint32_t icPos) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadInputCmdType, poss, index, icPos));
}

void Conv::enqueueStoreOutputCmd(std::vector<uint32_t> poss, uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::StoreOutputCmdType, poss, index));
}

void Conv::enqueueLoadBiasCmd(std::vector<uint32_t> poss, uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadBiasCmdType, poss, index));
}

void Conv::enqueueLoadQuantCmd(std::vector<uint32_t> poss, uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadQuantCmdType, poss, index));
}

void Conv::enqueueComputeQuantCmd(std::vector<uint32_t> poss, uint32_t index,
                                  uint32_t icPos) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputeQuantCmdType, poss, index, icPos));
}

void Conv::enqueueLoadWeightCmd(std::vector<uint32_t> poss, uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadWeightCmdType, poss, index));
}

void Conv::enqueueLoadWeightCmd(std::vector<uint32_t> poss, uint32_t index,
                                uint32_t icPos) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadWeightCmdType, poss, index, icPos));
}

void Conv::enqueueLoadScaleLutTblCmd() {
  std::vector<uint32_t> poss = {0, 0, 0, 0, 0};
  uint32_t index = 0;
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadScaleLutTblCmdType, poss, index));
}

void Conv::enqueueComputeCmd(std::vector<uint32_t> poss,
                             std::vector<uint32_t> indexes) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputCmdType, poss, indexes));
}

void Conv::enqueueComputeCmd(std::vector<uint32_t> poss,
                             std::vector<uint32_t> indexes, uint32_t icPos) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputCmdType, poss, indexes, icPos));
}

void Conv::enqueueComputeScaleLutCmd(std::vector<uint32_t> poss,
                                     uint32_t index) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputeScaleLutCmdType, poss, index));
}

void Conv::enqueueComputeScaleLutCmd(std::vector<uint32_t> poss, uint32_t index,
                                     uint32_t icPos) {
  cmdQueue.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputeScaleLutCmdType, poss, index, icPos));
}

void Conv::enqueueDisParallelCmd() {
  cmdQueue.push_back(
      std::make_unique<CmdDescriptor>(CmdDescriptor::ParallelCmdType, false));
}

void Conv::enqueueEnParallelCmd() {
  cmdQueue.push_back(
      std::make_unique<CmdDescriptor>(CmdDescriptor::ParallelCmdType, true));
}

void Conv::generateCmd() {
  auto genParallCmd = [&](uint32_t index) {
    if (cmdQueue[index]->isParallelEnabled())
      CV18xx::parallel_enable();
    else
      CV18xx::parallel_disable();
  };

  for (uint32_t i = 0; i < cmdQueue.size(); ++i) {
    CmdDescriptor::CmdTypeEnum cmdType = cmdQueue[i]->getCmdType();
    std::vector<uint32_t> gmOutputPoss = cmdQueue[i]->getGmOutputPoss();
    std::vector<uint32_t> lmIndexes = cmdQueue[i]->getLmIndexes();
    uint32_t icPos = cmdQueue[i]->getIcPos();

    if (cmdType == CmdDescriptor::LoadBiasCmdType) {
      loadBias(gmOutputPoss, lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::LoadQuantCmdType) {
      loadQuant(gmOutputPoss, lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::LoadInputCmdType) {
      loadInput(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::LoadWeightCmdType) {
      loadWeight(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::ComputeQuantCmdType) {
      computeQuant(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::LoadScaleLutTblCmdType) {
      loadScaleLutTable(lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::ComputCmdType) {
      compute(gmOutputPoss, lmIndexes, i, icPos);
    } else if (cmdType == CmdDescriptor::ComputeScaleLutCmdType) {
      computeScaleLut(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::StoreOutputCmdType) {
      storeOutput(gmOutputPoss, lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::ParallelCmdType) {
      genParallCmd(i);
    } else {
      assert(0 && "Expect valid command");
    }
  }
}

//
// This function implemnets weight reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
// TIU/TDMA command execution flow:
//   DMA G2L,  cmd_id 1, wait_id_tpu 0
//   DMA G2L,  cmd_id 2, wait_id_tpu 0
//   DMA G2L,  cmd_id 3, wait_id_tpu 0, LD0
//   TIU conv, cmd_id 1, cmd_id_gdma 3, TIU0, wait LD0
//   DMA G2L,  cmd_id 4, wait_id_tpu 0, LD1, no wait
//   TIU conv, cmd_id 2, cmd_id_gdma 4, TIU1, wait LD1
//   DMA L2G,  cmd_id 5, wait_id_tpu 1, SD0, wait TIU1
//   DMA G2L,  cmd_id 6, wait_id_tpu 0, LD2, no wait
//   TIU conv, cmd_id 3, cmd_id_gdma 6, TIU2, wait LD2
//   DMA L2G,  cmd_id 7, wait_id_tpu 2, SD1, wait TIU2
//   DMA G2L,  cmd_id 8, wait_id_tpu 0, LD3, no wait
//   TIU conv, cmd_id 4, cmd_id_gdma 8, TIU3, wait LD3
//
//   TDMA      TIU
//   LD0
//   LD1       TIU03
//   SD0/LD2   TIU1
//   SD1/LD3   TIU2
//
void Conv::convReuseWeight() {
  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "  convReuseWeight\n    "
          "groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n    "
          "kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n    "
          "stride (%d, %d), dilation (%d, %d)\n    "
          "do_bias %d, do_chl_quan %d\n    "
          "Tile (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, ih_step=%d"
          ", iw_step=%d, ic_step=%d)\n    ",
          args.groups, args.input_n, args.input_c, args.input_h, args.input_w,
          args.input_n, args.output_c, output_height(), output_width(), args.kh,
          args.kw, args.pad_top, args.pad_bottom, args.pad_left, args.pad_right,
          args.stride_h, args.stride_w, args.dilation_h, args.dilation_w,
          args.do_bias, args.do_chl_quan, tile_info.n_step, tile_info.oc_step,
          tile_info.oh_step, tile_info.ow_step, tile_info.ih_step,
          tile_info.iw_step, tile_info.ic_step));

  // tl_reg, tl_relu, tl_scale_lut
  if (args.do_scale_lut)
    enqueueLoadScaleLutTblCmd();

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    int first = 1;
    uint32_t flip = 0;
    uint32_t coeff_flip = 0;
    std::vector<uint32_t> gmOutputPoss[2];

    enqueueDisParallelCmd();

    // split oc
    for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
         oc_pos += tile_info.oc_step) {
      std::vector<uint32_t> cur_weight_pos = {/*n_pos=*/0, ig, oc_pos,
                                              /*oh_pos=*/0, /*ow_pos=*/0};
      enqueueLoadBiasCmd(cur_weight_pos, coeff_flip);
      if (args.do_quant) {
        enqueueDisParallelCmd();
        enqueueLoadQuantCmd(cur_weight_pos, coeff_flip);
      }
      enqueueLoadWeightCmd(cur_weight_pos, coeff_flip);
      if (args.do_quant) {
        enqueueComputeQuantCmd(cur_weight_pos, coeff_flip);
      }

      // split n
      for (uint32_t n_pos = 0; n_pos < batch_size();
           n_pos += tile_info.n_step) {
        // split h
        for (uint32_t oh_pos = 0; oh_pos < output_height();
             oh_pos += tile_info.oh_step) {

          // split w
          for (uint32_t ow_pos = 0; ow_pos < output_width();
               ow_pos += tile_info.ow_step) {
            gmOutputPoss[flip] = {n_pos, ig, oc_pos, oh_pos, ow_pos};

            if (args.do_scale_lut) {
              enqueueDisParallelCmd();
              enqueueLoadInputCmd(gmOutputPoss[flip], flip);
              enqueueComputeScaleLutCmd(gmOutputPoss[flip], flip);
            } else {
              enqueueLoadInputCmd(gmOutputPoss[flip], flip);
            }

            enqueueDisParallelCmd();
            enqueueEnParallelCmd();

            enqueueComputeCmd(gmOutputPoss[flip], {flip, coeff_flip, flip});

            if (first) {
              // postpone first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              uint32_t flip_back = 1 - flip;

              // Store back to global memory
              enqueueStoreOutputCmd(gmOutputPoss[flip_back], flip_back);
            }

            flip = 1 - flip;

          } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)
        }   // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)
      }     // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    enqueueDisParallelCmd();

    // The last iteration stored the other side, leave the last side not stored
    // store back to global memory
    uint32_t flip_back = 1 - flip;
    enqueueStoreOutputCmd(gmOutputPoss[flip_back], flip_back);
  } // for (int group_i = 0; group_i < groups; ++groups)
}

//
// This function implemnets activation(ifmap) reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
void Conv::convReuseActivation() {
  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "  convReuseActivation\n    "
          "groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n    "
          "kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n    "
          "stride (%d, %d), dilation (%d, %d)\n    "
          "do_bias %d, do_chl_quan %d\n    "
          "Tile (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, ih_step=%d"
          ", iw_step=%d, ic_step=%d)\n    ",
          args.groups, args.input_n, args.input_c, args.input_h, args.input_w,
          args.input_n, args.output_c, output_height(), output_width(), args.kh,
          args.kw, args.pad_top, args.pad_bottom, args.pad_left, args.pad_right,
          args.stride_h, args.stride_w, args.dilation_h, args.dilation_w,
          args.do_bias, args.do_chl_quan, tile_info.n_step, tile_info.oc_step,
          tile_info.oh_step, tile_info.ow_step, tile_info.ih_step,
          tile_info.iw_step, tile_info.ic_step));

  // tl_reg, tl_relu, tl_scale_lut
  if (args.do_scale_lut)
    enqueueLoadScaleLutTblCmd();

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    int first = 1;
    uint32_t flip = 0;
    uint32_t coeff_flip = 0;
    std::vector<uint32_t> gmOutputPoss[2];

    enqueueDisParallelCmd();

    // split n
    for (uint32_t n_pos = 0; n_pos < batch_size(); n_pos += tile_info.n_step) {

      // split h
      for (uint32_t oh_pos = 0; oh_pos < output_height();
           oh_pos += tile_info.oh_step) {

        // split w
        for (uint32_t ow_pos = 0; ow_pos < output_width();
             ow_pos += tile_info.ow_step) {

          if (args.do_scale_lut) {
            enqueueDisParallelCmd();
            enqueueLoadInputCmd({n_pos, ig, /*oc_pos=*/0, oh_pos, ow_pos},
                                flip);
            enqueueComputeScaleLutCmd({n_pos, ig, /*oc_pos=*/0, oh_pos, ow_pos},
                                      flip);
          } else
            enqueueLoadInputCmd({n_pos, ig, /*oc_pos=*/0, oh_pos, ow_pos},
                                flip);

          // split oc
          for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
               oc_pos += tile_info.oc_step) {
            gmOutputPoss[coeff_flip] = {n_pos, ig, oc_pos, oh_pos, ow_pos};
            std::vector<uint32_t> cur_weight_pos = {/*n_pos=*/0, ig, oc_pos,
                                                    /*oh_pos=*/0, /*ow_pos=*/0};
            enqueueLoadBiasCmd(cur_weight_pos, coeff_flip);
            enqueueLoadQuantCmd(cur_weight_pos, coeff_flip);
            enqueueLoadWeightCmd(cur_weight_pos, coeff_flip);

            enqueueDisParallelCmd();
            enqueueEnParallelCmd();

            if (args.do_quant) {
              enqueueComputeQuantCmd(cur_weight_pos, coeff_flip);
            }
            enqueueComputeCmd(gmOutputPoss[coeff_flip],
                              {flip, coeff_flip, coeff_flip});

            if (first) {
              // postpone first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              // Store back to global memory
              int coeff_flip_back = 1 - coeff_flip;
              enqueueStoreOutputCmd(gmOutputPoss[coeff_flip_back],
                                    coeff_flip_back);
            }

            coeff_flip = 1 - coeff_flip;

          } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

          flip = 1 - flip;

        } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

      } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

    } // for (int n_i = 0; n_i < n; ni += n_step)

    enqueueDisParallelCmd();

    // The last iteration stored the other side, leave the last side not stored
    // store back to global memory
    int coeff_flip_back = 1 - coeff_flip;
    enqueueStoreOutputCmd(gmOutputPoss[coeff_flip_back], coeff_flip_back);

  } // for (int group_i = 0; group_i < groups; ++groups)
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Borrowed from BM1880v2ConvFixedParallelv2::split
bool Conv::determineDwTileSize(bool useDoubleBuffer, bool favor_dma) {
  int input_n = args.input_n;
  int input_c = args.input_c;
  int input_h = args.input_h;
  int input_w = args.input_w;
  int do_bias = args.do_bias;
  bool do_chl_quan = args.do_chl_quan;
  int do_activation = args.do_activation;
  float *activation_arg = args.activation_arg;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  uint16_t dilation_h = args.dilation_h;
  uint16_t dilation_w = args.dilation_w;
  uint8_t pad_top = args.pad_top;
  uint8_t pad_bottom = args.pad_bottom;
  uint8_t pad_left = args.pad_left;
  uint8_t pad_right = args.pad_right;
  uint8_t stride_h = args.stride_h;
  uint8_t stride_w = args.stride_w;

  int ic = input_c;
  int oc = input_c;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh =
      (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h +
      1;
  int ow =
      (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w +
      1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  int32_t npu_num = static_cast<int32_t>(CV18xx::NPU_NUM);
  tile_info.n = 1;
  tile_info.oc = ceiling_func(oc, npu_num); // lane parallelism
  tile_info.ic = ic;
  tile_info.h = (ih + (MAX_HEIGHT - 1)) / MAX_HEIGHT;
  tile_info.w = (iw + (MAX_WIDTH - 1)) / MAX_WIDTH;

  int32_t num_oc_step = (oc + npu_num - 1) / npu_num;
  int32_t max_oc = std::min(oc, MAX_TIU_CHL);
  int ic_step = 1;

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;
  uint32_t scaleLutSize = 0;
  if (args.do_scale_lut) {
    cvk_tl_shape_t shape = CV18xx::lut_table_shape(args.input_fmt);
    scaleLutSize =
        CV18xx::lmem_tensor_to_size(shape, args.input_fmt, /*eu_align=*/1);
  }

  // Split ow
  for (tile_info.w = 1; tile_info.w <= ow; ++tile_info.w) {
    int ow_step = ceiling_func(ow, tile_info.w);
    int iw_step =
        ceiling_func((ow_step - 1) * stride_w + kw_extent, 1 + insert_width());
    iw_step = std::min(iw_step, iw);
    if (w_after_ins_pad(iw_step) > MAX_WIDTH || ow_step > MAX_WIDTH) {
      continue;
    }

    // Split oh
    for (tile_info.h = 1; tile_info.h <= oh; ++tile_info.h) {
      // split n
      for (tile_info.n = 1; tile_info.n <= n; ++tile_info.n) {
        int n_step = ceiling_func(n, tile_info.n);

        int oh_step = ceiling_func(oh, tile_info.h);
        int ih_step = ceiling_func((oh_step - 1) * stride_h + kh_extent,
                                   1 + insert_height());
        ih_step = std::min(ih_step, ih);
        if (h_after_ins_pad(ih_step) > MAX_HEIGHT || oh_step > MAX_HEIGHT) {
          continue;
        }

        // Split oc
        for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc) {
          // Downward, align lanes
          //   E.g. oc = 48, oc_step: 48, 32
          int32_t npu_num = static_cast<int32_t>(CV18xx::NPU_NUM);
          int32_t oc_step =
              std::min((num_oc_step - slice_oc) * npu_num, max_oc);

          uint32_t coeff_oc_step_size = 0;
          if (do_chl_quan) {
            uint32_t pc_bias_size = CV18xx::chan_quan_param_size(args.do_bias);
            cvk_tl_shape_t coeff_shape =
                CV18xx::tl_shape_t4(1, oc_step, 1, pc_bias_size);
            coeff_oc_step_size +=
                CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                            /*eu_align=*/0);
          } else if (do_bias) {
            // 16 bit
            cvk_tl_shape_t coeff_shape_i16 =
                CV18xx::tl_shape_t4(2, oc_step, 1, 1);
            coeff_oc_step_size += CV18xx::lmem_tensor_to_size(
                coeff_shape_i16, args.tiu_fmt, /*eu_align=*/0);
          }
          if (args.do_quant) {
            auto coeff_shape = CV18xx::tl_shape_t4(1, oc_step, 1, 1);
            coeff_oc_step_size +=
                2 * CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                                /*eu_align=*/0);
          }
          // Add weight size
          uint32_t weight_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(ic_step, oc_step, kh, kw), args.tiu_fmt,
              /*eu_align=*/1);

          uint32_t ofmap_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(n_step, oc_step, oh_step, ow_step),
              args.tiu_fmt,
              /*eu_align=*/1);

          uint32_t ifmap_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(n_step, oc_step, ih_step, iw_step),
              args.tiu_fmt,
              /*eu_align=*/1);

          // Leaky relu need tl_neg, tl_relu.
          // tl_relu, tl_neg are not from tmda and not final output.
          // One copy is enough.
          uint32_t extra_size = scaleLutSize;
          if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
            extra_size += 2 * ofmap_size; // tl_relu + tl_neg
          }

          uint32_t total_needed =
              (ifmap_size + ofmap_size + weight_size + coeff_oc_step_size) *
                  bufferMultiplier +
              extra_size;

          tile_info.n_step = n_step;
          tile_info.oc_step = oc_step;
          tile_info.oh_step = oh_step;
          tile_info.ow_step = ow_step;
          tile_info.ih_step = ih_step;
          tile_info.iw_step = iw_step;
          tile_info.ic_step = ic_step;
          tile_info.total_needed = total_needed;
          tile_info.favor_dma = favor_dma;
          this->use_double_buffer = useDoubleBuffer;

          if (total_needed <= CV18xx::LMEM_BYTES && checkDmaPolicy(tile_info)) {
            tile_info.n_step = n_step;

            uint32_t total_size =
                ic * ih * iw + oc * ic * kh * kw + oc * oh * ow;
            uint32_t ut = (total_needed * 100) / CV18xx::LMEM_BYTES;
            bool is_tiled = ((oc != oc_step) || (oh != oh_step)) ? true : false;
            const char *ut_msg =
                (ut < 70 && is_tiled && !favor_dma) ? " => NG(dw)" : "";

            LLVM_DEBUG(
                llvm::errs() << llvm::format(
                    "  determineDwTileSize\n    "
                    "ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n    "
                    "kernel (%d, %d), pad (top=%d, bot=%d, left=%d, "
                    "right=%d)\n    "
                    "ins_h %d, ins_w %d, stride (%d, %d), dilation (%d, %d)\n",
                    input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh,
                    kw, pad_top, pad_bottom, pad_left, pad_right,
                    insert_height(), insert_width(), stride_h, stride_w,
                    dilation_h, dilation_w));
            LLVM_DEBUG(
                llvm::errs() << llvm::format(
                    "    Tile (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d"
                    ", ih_step=%d, iw_step=%d, ic_step=%d)\n    "
                    "inputSize %d, outputSize %d, weightSize %d"
                    ", biasSize %d, totalSizePerLane %d/%d"
                    ", totalSize %d/%d(%d), %d/%d(%d)%s\n    "
                    "tiled ifmap shape (%d, %d, %d, %d)\n    "
                    "tiled weight shape (%d, %d, %d, %d)\n    "
                    "tiled ofmap shape (%d, %d, %d, %d)\n    "
                    "use_double_buffer %d, favor_dma %d\n",
                    n_step, oc_step, oh_step, ow_step, ih_step, iw_step,
                    ic_step, ifmap_size * bufferMultiplier,
                    ofmap_size * bufferMultiplier,
                    weight_size * bufferMultiplier,
                    coeff_oc_step_size * bufferMultiplier, total_needed,
                    CV18xx::LMEM_BYTES, total_needed * CV18xx::NPU_NUM,
                    CV18xx::LMEM_BYTES * CV18xx::NPU_NUM,
                    (total_needed * 100) / CV18xx::LMEM_BYTES, total_needed,
                    total_size, (total_needed * 100) / total_size, ut_msg,
                    n_step, oc_step, ih_step, iw_step, oc_step, ic_step, kh, kw,
                    n_step, oc_step, oh_step, ow_step, use_double_buffer,
                    favor_dma));
            return true;
          }
        } // for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc)
      }   // for (tile_info.n = 1; tile_info.n < n; ++tile_info.n)
    }     // for (tile_info.h = 1; tile_info.h <= oh; ++tile_info.h)
  }       // for (tile_info.w = 1; tile_info.w <= ow; ++tile_info.ow)

  tile_info = {0};

  return false;
}

void Conv::dwConv() {
  allocateAllLocalMem();

  int first = 1;
  uint32_t flip = 0;
  uint32_t coeff_flip = 0;
  std::vector<uint32_t> gmOutputPoss[2];

  tilePolicy = ReuseActivationPolicyType;

  // tl_reg, tl_relu, tl_scale_lut
  if (args.do_scale_lut)
    enqueueLoadScaleLutTblCmd();

  enqueueDisParallelCmd();

  // split oc
  for (uint32_t oc_pos = 0; oc_pos < groups(); oc_pos += tile_info.oc_step) {
    std::vector<uint32_t> cur_weight_pos = {/*n_pos=*/0, 0, oc_pos,
                                            /*oh_pos=*/0, /*ow_pos=*/0};
    enqueueLoadBiasCmd(cur_weight_pos, coeff_flip);
    if (args.do_quant) {
      enqueueDisParallelCmd();
      enqueueLoadQuantCmd(cur_weight_pos, coeff_flip);
    }
    enqueueLoadWeightCmd(cur_weight_pos, coeff_flip);
    if (args.do_quant) {
      enqueueComputeQuantCmd(cur_weight_pos, coeff_flip);
    }

    // split n
    for (uint32_t n_pos = 0; n_pos < batch_size(); n_pos += tile_info.n_step) {
      // split h
      for (uint32_t oh_pos = 0; oh_pos < output_height();
           oh_pos += tile_info.oh_step) {
        // split w
        for (uint32_t ow_pos = 0; ow_pos < output_width();
             ow_pos += tile_info.ow_step) {
          gmOutputPoss[flip] = {n_pos, 0, oc_pos, oh_pos, ow_pos};

          if (args.do_scale_lut) {
            enqueueDisParallelCmd();
            enqueueLoadInputCmd(gmOutputPoss[flip], flip);
            enqueueComputeScaleLutCmd(gmOutputPoss[flip], flip);
          } else {
            enqueueLoadInputCmd(gmOutputPoss[flip], flip);
          }

          enqueueDisParallelCmd();
          enqueueEnParallelCmd();

          enqueueComputeCmd(gmOutputPoss[flip], {flip, coeff_flip, flip});

          if (first) {
            // postpone first result to next loop
            // loop0: LD0 TIU0
            // loop1: LD1 TIU1 SD0
            // loop2: LD2 TIU2 SD1
            first = 0;
          } else {
            uint32_t flip_back = 1 - flip;

            // Store back to global memory
            enqueueStoreOutputCmd(gmOutputPoss[flip_back], flip_back);
          }

          flip = 1 - flip;

        } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

      } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

    } // for (int n_i = 0; n_i < n; ni += n_step)

    coeff_flip = 1 - coeff_flip;

  } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

  enqueueDisParallelCmd();

  // Store back to global memory
  uint32_t flip_back = 1 - flip;
  enqueueStoreOutputCmd(gmOutputPoss[flip_back], flip_back);

  // Generate command
  generateCmd();

  deallocateAllLocalMem();
}

// No tiling, no parallel.
bool Conv::canNoTile() {
  // H/W does not support group convolution.
  if (args.groups > 1)
    return false;

  // Hardware limit
  if ((group_input_channels() > MAX_TIU_CHL) ||
      (group_output_channels() > MAX_TIU_CHL) ||
      (input_height() > MAX_HEIGHT) || (input_width() > MAX_WIDTH) ||
      (output_width() > MAX_WIDTH) || (output_height() > MAX_HEIGHT))
    return false;

  int input_n = args.input_n;
  int input_c = args.input_c;
  int input_h = args.input_h;
  int input_w = args.input_w;
  int groups = args.groups;
  int output_c = args.output_c;
  int do_bias = args.do_bias;
  // bool do_chl_quan = args.do_chl_quan;
  int do_activation = args.do_activation;
  float *activation_arg = args.activation_arg;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  uint16_t dilation_h = args.dilation_h;
  uint16_t dilation_w = args.dilation_w;
  uint8_t pad_top = args.pad_top;
  uint8_t pad_bottom = args.pad_bottom;
  uint8_t pad_left = args.pad_left;
  uint8_t pad_right = args.pad_right;
  uint8_t stride_h = args.stride_h;
  uint8_t stride_w = args.stride_w;

  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh =
      (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h +
      1;
  int ow =
      (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w +
      1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  uint32_t coeff_size = 0;

  if (args.tiu_fmt == CVK_FMT_I8) {
    // int8
    if (args.do_chl_quan) {
      // per-channel
      if (do_bias) {
        cvk_tl_shape_t coeff_shape_9byte = CV18xx::tl_shape_t4(1, oc, 1, 9);
        coeff_size +=
            CV18xx::lmem_tensor_to_size(coeff_shape_9byte, args.tiu_fmt,
                                        /*eu_align=*/0);
      } else {
        cvk_tl_shape_t coeff_shape_5byte = CV18xx::tl_shape_t4(1, oc, 1, 5);
        coeff_size +=
            CV18xx::lmem_tensor_to_size(coeff_shape_5byte, args.tiu_fmt,
                                        /*eu_align=*/0);
      }
    } else if (do_bias) {
      // per-tensor
      cvk_tl_shape_t coeff_shape = CV18xx::tl_shape_t4(2, oc, 1, 1);
      coeff_size += CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                                /*eu_align=*/0);
    }
  } else {
    // bf16
    if (do_bias) {
      cvk_tl_shape_t coeff_shape = CV18xx::tl_shape_t4(2, oc, 1, 1);
      coeff_size += CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                                /*eu_align=*/0);
    }
  }
  if (args.do_quant) {
    auto coeff_shape = CV18xx::tl_shape_t4(1, oc, 1, 1);
    coeff_size += 2 * CV18xx::lmem_tensor_to_size(coeff_shape, args.tiu_fmt,
                                                  /*eu_align=*/0);
  }
  // Add weight size
  uint32_t weight_size = CV18xx::lmem_tensor_to_size(
      CV18xx::tl_shape_t4(ic, oc, kh, kw), args.tiu_fmt, /*eu_align=*/0);

  uint32_t ofmap_size = CV18xx::lmem_tensor_to_size(
      CV18xx::tl_shape_t4(n, oc, oh, ow), args.tiu_fmt,
      /*eu_align=*/1);

  uint32_t ifmap_size = CV18xx::lmem_tensor_to_size(
      CV18xx::tl_shape_t4(n, ic, ih, iw), args.tiu_fmt,
      /*eu_align=*/1);

  uint32_t total_needed = ifmap_size + ofmap_size + weight_size + coeff_size;

  // Leaky relu need tl_neg, tl_relu.
  // tl_relu, tl_neg are not from tmda and not final output.
  if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
    total_needed += 2 * ofmap_size; // tl_relu + tl_neg
  }

  LLVM_DEBUG(llvm::dbgs() << "  canNoTile:\n"
                          << "    layer_id " << args.layer_id << "\n    "
                          << "total_needed " << static_cast<int>(total_needed)
                          << "\n    "
                          << "inputSize " << ifmap_size << ", outputSize "
                          << ofmap_size << ", weightSize " << weight_size
                          << ", biasSize " << coeff_size << "\n    "
                          << "ifmap shape (" << input_n << ", " << input_c
                          << ", " << input_h << ", " << input_w << ")\n    "
                          << "weight shape (oc=" << output_c << ", kh=" << kh
                          << ", kw=" << kw << ", ic=" << input_c << ")\n    "
                          << "ofmap shape (" << input_n << ", " << oc << ", "
                          << oh << ", " << ow << ")\n");

  if (total_needed <= CV18xx::LMEM_BYTES) {
    tile_info.n_step = n;
    tile_info.oc_step = oc;
    tile_info.oh_step = oh;
    tile_info.ow_step = ow;
    tile_info.ih_step = ih;
    tile_info.iw_step = iw;
    tile_info.ic_step = ic;
    tile_info.total_needed = total_needed;
    return true;
  }

  return false;
}

// No tiling, no parallel execution, maximized local memory utilization.
// 1. For activation compression.
// 2. For maximum TDMA load/store efficiency and local memory utilization
//    With tiu/tdma outstanding feature in 1822,
//    the backend does not need to split output channels to reduce TDMA
//    store latency.
//    Without it, the compiler has to implement inter-layer outstanding.
void Conv::convNoTile() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tg_fixed_conv_kernel w/ one tiu:\n"
                 "    layer_id %d\n"
                 "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                 "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                 "    kernel = (%d, %d), dilation = (%d, %d)\n"
                 "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
                 args.layer_id, args.ga_ifmap, args.ga_ofmap, args.ga_weight,
                 args.ga_bias, args.input_n, args.input_c, args.input_h,
                 args.input_w, args.groups, args.output_c, args.kh, args.kw,
                 args.dilation_h, args.dilation_w, args.pad_top,
                 args.pad_bottom, args.pad_left, args.pad_right, args.stride_h,
                 args.stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "    activation_gt_scale = %d, activation_gt_scale = %d\n"
                 "    activation_le_rshift = %d, activation_le_scale = %d\n"
                 "    do_activation = %d\n"
                 "    do_ic_alignment = %d\n",
                 args.activation_gt_scale, args.activation_gt_scale,
                 args.activation_le_rshift, args.activation_le_scale,
                 args.do_activation, args.do_ic_alignment));

  std::vector<uint32_t> poss = {0, 0, 0, 0, 0};
  std::vector<uint32_t> indexes = {0, 0, 0};

  // tl_reg, tl_relu, tl_scale_lut
  if (args.do_scale_lut)
    enqueueLoadScaleLutTblCmd();

  enqueueLoadBiasCmd(poss, indexes[0]);
  enqueueLoadQuantCmd(poss, indexes[0]);
  enqueueLoadInputCmd(poss, indexes[0]);

  if (args.do_scale_lut)
    enqueueComputeScaleLutCmd(poss, indexes[0]);

  enqueueLoadWeightCmd(poss, indexes[0]);
  if (args.do_quant) {
    enqueueComputeQuantCmd(poss, indexes[0]);
  }
  enqueueComputeCmd(poss, indexes);
  enqueueStoreOutputCmd(poss, indexes[0]);
}

// Straightforward tiling, no double buffer
// Modified from convReuseActivation
void Conv::convNaive() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "convNaive =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n"
                 "  do_bias %d, do_chl_quan %d\n"
                 "  Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, "
                 "ih_step=%d"
                 ", iw_step=%d, ic_step=%d)\n",
                 args.groups, args.input_n, args.input_c, args.input_h,
                 args.input_w, args.input_n, args.output_c, output_height(),
                 output_width(), args.kh, args.kw, args.pad_top,
                 args.pad_bottom, args.pad_left, args.pad_right, args.stride_h,
                 args.stride_w, args.dilation_h, args.dilation_w, args.do_bias,
                 args.do_chl_quan, tile_info.n_step, tile_info.oc_step,
                 tile_info.oh_step, tile_info.ow_step, tile_info.ih_step,
                 tile_info.ih_step, tile_info.ic_step));

  // tl_reg, tl_relu, tl_scale_lut
  if (args.do_scale_lut)
    enqueueLoadScaleLutTblCmd();

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    std::vector<uint32_t> gmOutputPoss;
    // split oc
    for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
         oc_pos += tile_info.oc_step) {
      // split n
      for (uint32_t n_pos = 0; n_pos < batch_size();
           n_pos += tile_info.n_step) {
        // split h
        for (uint32_t oh_pos = 0; oh_pos < output_height();
             oh_pos += tile_info.oh_step) {
          // split w
          for (uint32_t ow_pos = 0; ow_pos < output_width();
               ow_pos += tile_info.ow_step) {
            gmOutputPoss = {n_pos, ig, oc_pos, oh_pos, ow_pos};
            std::vector<uint32_t> cur_weight_pos = {/*n_pos=*/0, ig, oc_pos,
                                                    /*oh_pos=*/0,
                                                    /*ow_pos=*/0};

            enqueueLoadBiasCmd(cur_weight_pos, /*flip=*/0);
            enqueueLoadQuantCmd(cur_weight_pos, /*flip=*/0);

            for (uint32_t ic_pos = 0; ic_pos < group_input_channels();
                 ic_pos += tile_info.ic_step) {

              enqueueLoadInputCmd(
                  {n_pos, /*g_pos=*/0, /*oc_pos=*/0, oh_pos, ow_pos},
                  /*flip*/ 0, ic_pos);

              if (args.do_scale_lut)
                enqueueComputeScaleLutCmd(
                    {n_pos, /*g_pos=*/0, /*oc_pos=*/0, oh_pos, ow_pos},
                    /*flip*/ 0, ic_pos);

              enqueueLoadWeightCmd(cur_weight_pos, /*flip=*/0, ic_pos);
              if (args.do_quant) {
                enqueueComputeQuantCmd(cur_weight_pos, /*flip=*/0, ic_pos);
              }

              enqueueComputeCmd(gmOutputPoss,
                                {/*flip=*/0, /*flip=*/0, /*flip=*/0}, ic_pos);
            }

            enqueueStoreOutputCmd(gmOutputPoss, /*flip=*/0);

          } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

        } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      } // for (int n_i = 0; n_i < n; ni += n_step)

    } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

  } // for (int group_i = 0; group_i < groups; ++groups)

  LLVM_DEBUG(llvm::errs() << "<= convNaive\n");
}

void Conv::showCost(CostModel &cost) {
  LLVM_DEBUG(llvm::dbgs() << "total " << cost.totalRWSize << ", weight read "
                          << cost.wgtReadSize << ", act read "
                          << cost.actReadSize << ", act write "
                          << cost.actWriteSize << "\n");
}

void Conv::getCost(CostModel &cost) {
  for (uint32_t i = 0; i < cmdQueue.size(); ++i) {
    CmdDescriptor::CmdTypeEnum cmdType = cmdQueue[i]->getCmdType();
    std::vector<uint32_t> gmOutputPoss = cmdQueue[i]->getGmOutputPoss();
    std::vector<uint32_t> lmIndexes = cmdQueue[i]->getLmIndexes();
    uint32_t icPos = cmdQueue[i]->getIcPos();

    if (cmdType == CmdDescriptor::LoadBiasCmdType) {
    } else if (cmdType == CmdDescriptor::LoadQuantCmdType) {
    } else if (cmdType == CmdDescriptor::LoadInputCmdType) {
      // loadInput(gmOutputPoss, lmIndexes[0], i, icPos);
      std::vector<uint32_t> gmOutputPossShapes =
          getTiledGmShapesOfOutputForTiu(gmOutputPoss);

      std::vector<uint32_t> cur_gm_input_poss;
      std::vector<uint32_t> shapes;
      std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
      getTiledGmPossAndShapesOfInputForTiu(gmOutputPoss, gmOutputPossShapes,
                                           cur_gm_input_poss, shapes,
                                           cur_gm_input_paddings, icPos);

      auto count = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                                   std::multiplies<>());
      cost.totalRWSize += count;
      cost.actReadSize += count;
    } else if (cmdType == CmdDescriptor::LoadWeightCmdType) {
      // loadWeight(gmOutputPoss, lmIndexes[0], i, icPos);
      std::vector<uint32_t> shapes =
          getTiledGmShapesOfWeightForTdmaLoad(gmOutputPoss, icPos);
      auto count = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                                   std::multiplies<>());
      cost.totalRWSize += count;
      cost.wgtReadSize += count;
    } else if (cmdType == CmdDescriptor::LoadScaleLutTblCmdType) {

    } else if (cmdType == CmdDescriptor::ComputCmdType) {
      // compute(gmOutputPoss, lmIndexes, i, icPos);
    } else if (cmdType == CmdDescriptor::ComputeScaleLutCmdType) {

    } else if (cmdType == CmdDescriptor::ComputeQuantCmdType) {

    } else if (cmdType == CmdDescriptor::StoreOutputCmdType) {
      // storeOutput(gmOutputPoss, lmIndexes[0], i);

      std::vector<uint32_t> shapes =
          getTiledGmShapesOfOutputForTiu(gmOutputPoss);
      auto count = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                                   std::multiplies<>());
      cost.totalRWSize += count;
      cost.actWriteSize += count;
    } else if (cmdType == CmdDescriptor::ParallelCmdType) {
      // genParallCmd(i);
    } else {
      assert(0 && "Expect valid command");
    }
  }
}

bool Conv::isBetterCost(CostModel &from, CostModel &to) {
  // Since both reuse weight and reuse activation use double buffer,
  // use total read/write size as cost factor.
  if (to.totalRWSize < from.totalRWSize)
    return true;

  return false;
}

Conv::TilePolicy Conv::getReuseWgtOrActByCost() {
  CostModel wgtCost = {0}, actCost = {0};

  TilePolicy policy = ReuseWeightPolicyType;
  convReuseActivation();
  getCost(actCost);

  cmdQueue.clear();
  convReuseWeight();
  getCost(wgtCost);
  if (isBetterCost(wgtCost, actCost))
    policy = ReuseActivationPolicyType;

  cmdQueue.clear();

  return policy;
}

// Priority:
//   1. No tiling
//   2. Reuse weight w/ double buffer
//   3. Reuse activation w/ double buffer
//   4. Tile w/ single buffer
//   5. Tile+ps32 w/ single buffer
//
void Conv::determineTilePolicy() {
  if (canNoTile()) {
    // No tiling should be the best condition
    tilePolicy = NoTilePolicyType;

    // Update tiling again for ic alignment.
    initializeTile();
  } else if (determineTileSize(true, true)) {
    // Use single buffer to increase eu efficiency if output height*width is
    // too small.
    if (tile_info.ow_step < output_width() &&
        (tile_info.oh_step * tile_info.ow_step) <
            CV18xx::tiu_eu_num(args.tiu_fmt)) {
      determineTileSize(false, true);

      if (tile_info.ow_step < output_width() &&
          (tile_info.oh_step * tile_info.ow_step) <
              CV18xx::tiu_eu_num(args.tiu_fmt)) {
        // Use ps32 to increase eu efficiency if output height*width is too
        // small.
        determinePs32TileSize(false);
        tilePolicy = SingleBufferPs32PolicyType;
      } else
        tilePolicy = SingleBufferPolicyType;
    } else {
      use_double_buffer = true;
      tilePolicy = getReuseWgtOrActByCost();
    }
  } else if (determineTileSize(false, true)) {
    // Use ps32 to increase eu efficiency if output height*width is too small.
    if (tile_info.ow_step < output_width() &&
        (tile_info.oh_step * tile_info.ow_step) <
            CV18xx::tiu_eu_num(args.tiu_fmt)) {
      determinePs32TileSize(false);
      tilePolicy = SingleBufferPs32PolicyType;
    } else
      tilePolicy = SingleBufferPolicyType;
  } else if (determinePs32TileSize(false)) {
    tilePolicy = SingleBufferPs32PolicyType;
  } else {
    assert(0 && "Expect valid tile policy");
    tilePolicy = MaxTilePolicyType;
  }
}

bool Conv::compressWeight() {
  bool isBf16Flt = (args.tiu_fmt == CVK_FMT_BF16);
  int fltEltSize = isBf16Flt ? 2 : 1;
  int oc = group_output_channels();
  int ic = group_input_channels();
  int kh = kernel_height();
  int kw = kernel_width();
  int oc_step = tile_info.oc_step;
  int ic_step = tile_info.ic_step;

  if (ic != ic_step) {
    return false;
  }

  int totalSize = 0;
  int totalCompressedSize = 0;

  int maxPlainSize = oc_step * kh * kw * ic * fltEltSize;
  auto plainData = std::make_unique<std::vector<uint8_t>>(maxPlainSize);

  int maxComprSize = getCompressedDataSize(maxPlainSize, isBf16Flt ? 1 : 0);
  auto compressedData = std::make_unique<std::vector<uint8_t>>(maxComprSize);

  bool canCompress = true;
  int filterSize = args.filter->size();
  for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
    int cur_oc = std::min(oc - oc_pos, oc_step);
    int stepSize = cur_oc * kh * kw * ic * fltEltSize;
    int pos = oc_pos * kh * kw * ic * fltEltSize;

    // H/W constraint: must align 16B
    if (pos % 16) {
      canCompress = false;
      break;
    }

    std::memcpy(plainData->data(), args.filter->data() + pos, stepSize);

    // Calculate compress parameter first.
    CompressCommandInfo cmdInfo;
    std::memset(&cmdInfo, 0, sizeof(cmdInfo));
    cmdInfo.signedness = isBf16Flt ? 0 : 1;
    cmdInfo.is_bfloat16 = isBf16Flt ? 1 : 0;
    cmdInfo.bias0 = isBf16Flt ? 127 : 0;
    getCompressParameter(plainData->data(), stepSize, cmdInfo.signedness,
                         cmdInfo.is_bfloat16, &cmdInfo);

    int compressedSize = maxComprSize;
    if (isBf16Flt)
      compressBf16Data(plainData->data(), stepSize, compressedData->data(),
                       &compressedSize, &cmdInfo);
    else
      compressInt8Data(plainData->data(), stepSize, compressedData->data(),
                       &compressedSize, &cmdInfo);

    // Compress size must be less than tiled size.
    LLVM_DEBUG(llvm::dbgs()
               << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
               << ", stepSize " << stepSize << ", compressedSize "
               << compressedSize << ", pos " << pos << ", totalSize "
               << totalSize << ", totalCompressedSize " << totalCompressedSize
               << ", filterSize " << filterSize << "\n");

    if (compressedSize > stepSize) {
      llvm::errs() << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
                   << ", stepSize " << stepSize << ", compressedSize "
                   << compressedSize << ", SKIP\n";
      canCompress = false;
      break;
    } else {
      totalSize += stepSize;
      totalCompressedSize += compressedSize;
    }

    // Fill compressed data.
    assert(static_cast<uint32_t>(pos + compressedSize) <=
           args.new_filter->size());
    std::memcpy(args.new_filter->data() + pos, compressedData->data(),
                compressedSize);
  }
  return canCompress;
}

void Conv::doConvByTilePolicy() {
  // Pre-alloc maximum one-step size
  // The local memory release must be in reverse order.
  allocateAllLocalMem();

  configCModelDebug();

  switch (tilePolicy) {
  case NoTilePolicyType:
    convNoTile();
    break;

  case SingleBufferPolicyType:
    convNaive();
    break;

  case SingleBufferPs32PolicyType:
    convNaive();
    break;

  case ReuseWeightPolicyType:
    convReuseWeight();
    break;

  case ReuseActivationPolicyType:
    convReuseActivation();
    break;

  default:
    return;
  }

  if (args.do_load_cmpr_wgt) {
    if (!compressWeight()) {
      args.do_load_cmpr_wgt = false;
      args.new_filter->clear();
    }
  }

  // Channel should larger than CV18xx::NPU_NUM
  // Disable intra-cmd with reuse-weight
  //   need to reorder load input and weight
  // Disable intra-cmd with ps32
  //   need to separate load+compute, compute+store (e.g. ssd300 bf16)
  // Disable intra-cmd with compression
  //   h/w does not guarantee to work, failed in alphapose
  if (tile_info.ic_step > (uint32_t)CV18xx::NPU_NUM &&
      tile_info.oc_step > (uint32_t)CV18xx::NPU_NUM &&
      tilePolicy == ReuseActivationPolicyType &&
      tile_info.ic_step == group_input_channels()) {
    auto intraCmdAnalysis =
        std::make_unique<IntraCmdParallelAnalysis>(cmdQueue);
    intraCmdAnalysis->analyze();
    // intraCmdAnalysis->dumpStates();
  }

  // Generate command
  generateCmd();

  deallocateAllLocalMem();

  cModelDebug.dump();
}

void cvi_backend_tg_fixed_conv_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight,
    gaddr_t ga_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t insert_h, uint8_t insert_w, uint8_t stride_h,
    uint8_t stride_w, int do_bias, int do_activation, float activation_arg[],
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, int right_shift_width, bool do_chl_quan,
    bool do_ic_alignment, std::vector<uint8_t> *filter,
    std::vector<uint8_t> *new_filter, int pad_value, gaddr_t ga_scale_lut) {
  // this message is too long for llvm::format, so seperate it
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tg_fixed_conv_kernel:\n"
                 "    layer_id %d\n"
                 "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                 "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                 "    kernel = (%d, %d), dilation = (%d, %d)\n"
                 "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
                 layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n,
                 input_c, input_h, input_w, groups, output_c, kh, kw,
                 dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
                 pad_right, stride_h, stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "    activation_gt_rshift = %d, activation_gt_scale = %d\n"
                 "    activation_le_rshift = %d, activation_le_scale = %d\n"
                 "    do_activation = %d\n"
                 "    do_ic_alignment = %d\n"
                 "    ga_scale_lut 0x%lx\n",
                 activation_gt_rshift, activation_gt_scale,
                 activation_le_rshift, activation_le_scale, do_activation,
                 do_ic_alignment, ga_scale_lut));

  //
  // Convolution initialization
  //   Too many arguments come from pure-C api.
  //
  auto conv(std::make_unique<Conv>());
  conv->args.ga_ifmap = ga_ifmap;
  conv->args.ga_ofmap = ga_ofmap;
  conv->args.ga_weight = ga_weight;
  conv->args.ga_bias = ga_bias;
  conv->args.input_n = input_n;
  conv->args.input_c = input_c;
  conv->args.input_h = input_h;
  conv->args.input_w = input_w;
  conv->args.groups = groups;
  conv->args.output_c = output_c;
  conv->args.kh = kh;
  conv->args.kw = kw;
  conv->args.dilation_h = dilation_h;
  conv->args.dilation_w = dilation_w;
  conv->args.pad_top = pad_top;
  conv->args.pad_bottom = pad_bottom;
  conv->args.pad_left = pad_left;
  conv->args.pad_right = pad_right;
  conv->args.insert_h = insert_h;
  conv->args.insert_w = insert_w;
  conv->args.stride_h = stride_h;
  conv->args.stride_w = stride_w;
  conv->args.do_bias = static_cast<bool>(do_bias);
  conv->args.do_activation = static_cast<bool>(do_activation);
  conv->args.activation_arg = activation_arg;
  conv->args.activation_gt_scale = activation_gt_scale;
  conv->args.activation_gt_rshift = activation_gt_rshift;
  conv->args.activation_le_scale = activation_le_scale;
  conv->args.activation_le_rshift = activation_le_rshift;
  conv->args.right_shift_width = right_shift_width;
  conv->args.do_chl_quan = do_chl_quan;
  conv->args.layer_id = layer_id;
  conv->args.do_ic_alignment = do_ic_alignment;
  conv->args.pad_value = pad_value;
  conv->args.do_quant = false;
  conv->args.ga_scale_lut = ga_scale_lut;
  conv->args.do_scale_lut = ga_scale_lut != GA_INVALID ? true : false;
  // Mix-precision tdma load/store from dialect
  // E.g. input int8 -> tiu bf16 -> output fp32
  conv->args.input_fmt = CVK_FMT_I8;
  conv->args.output_fmt = CVK_FMT_I8;
  conv->args.tiu_fmt = CVK_FMT_I8;
  conv->args.filter = filter;
  conv->args.new_filter = new_filter;
  conv->args.do_load_cmpr_wgt = false;
  if (filter != nullptr && !filter->empty()) {
    assert(new_filter && !new_filter->empty());
    conv->args.do_load_cmpr_wgt = true;
  }

  // Global memory region from dialect
  conv->initializeGlobalMem();

  conv->initializeFusedActivation();
  conv->initializeTile();

  // For tdma
  CV18xx::set_layer_id(layer_id);

  // Try depthwise convolution.
  if (conv->isDwConv()) {
    if (conv->determineDwTileSize(true, false))
      return conv->dwConv();
    else
      assert(0 && "DwConv does not support single buffer yet");
  }

  // For double convolution, weight and output already altered.
  // But the input is still unchanged and needs to restore original channels.
  // In dialect, ifmap tensor<1x3x85x85xi8>, weight tensor<64x4x5x5xi8>
  if (do_ic_alignment && (input_c % 2 != 0)) {
    assert(input_c >= 1);
    conv->args.input_c = input_c + 1;
  }

  conv->determineTilePolicy();
  conv->doConvByTilePolicy();
}

void Conv::configCModelDebug() {
  // WZC-0, batch 8
  // [ig=0][oc_pos=736][n_pos=6][oh_pos=31][ow_pos=0][ic_pos=0]
  // cModelDebug.assignOutput(21, {6, 0, 736, 31, 0});

  // onnx
  // cModelDebug.assignOutput(1, {6, 0, 736, 31, 0});

  // WZC-6, batch 4 in onnx
  // [ig=0][oc_pos=1152][n_pos=2][oh_pos=15][ow_pos=34][ic_pos=0]
  // cModelDebug.assignOutput(1, {2, 0, 1152, 15, 34});
}

void cvi_backend_tg_bf16_conv_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight,
    gaddr_t ga_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t ins_h, uint8_t ins_w, uint8_t stride_h,
    uint8_t stride_w, int do_bias, int do_activation, bool fp32_output,
    std::vector<uint8_t> *filter, std::vector<uint8_t> *new_filter,
    bool do_quant, gaddr_t ga_scale, gaddr_t ga_zeropoint) {

  // this message is too long for llvm::format, so separate it
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tg_bf16_conv_kernel:\n"
                 "    layer_id %d\n"
                 "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                 "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                 "    kernel = (%d, %d), dilation = (%d, %d)\n"
                 "    pad = (%d, %d, %d, %d), ins=(%d, %d) stride = (%d, %d)\n",
                 layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n,
                 input_c, input_h, input_w, groups, output_c, kh, kw,
                 dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
                 pad_right, ins_h, ins_w, stride_h, stride_w));
  LLVM_DEBUG(
      llvm::errs() << llvm::format("    do_activation = %d\n", do_activation));

  //
  // Convolution initialization
  //   Too many arguments come from pure-C api.
  //
  auto conv(std::make_unique<Conv>());
  conv->args.ga_ifmap = ga_ifmap;
  conv->args.ga_ofmap = ga_ofmap;
  conv->args.ga_weight = ga_weight;
  conv->args.ga_bias = ga_bias;
  conv->args.ga_scale = ga_scale;
  conv->args.ga_zeropoint = ga_zeropoint;
  conv->args.input_n = input_n;
  conv->args.input_c = input_c;
  conv->args.input_h = input_h;
  conv->args.input_w = input_w;
  conv->args.groups = groups;
  conv->args.output_c = output_c;
  conv->args.kh = kh;
  conv->args.kw = kw;
  conv->args.dilation_h = dilation_h;
  conv->args.dilation_w = dilation_w;
  conv->args.pad_top = pad_top;
  conv->args.pad_bottom = pad_bottom;
  conv->args.pad_left = pad_left;
  conv->args.pad_right = pad_right;
  conv->args.insert_h = ins_h;
  conv->args.insert_w = ins_w;
  conv->args.stride_h = stride_h;
  conv->args.stride_w = stride_w;
  conv->args.do_bias = static_cast<bool>(do_bias);
  conv->args.do_activation = static_cast<bool>(do_activation);
  conv->args.do_quant = do_quant;
  conv->args.layer_id = layer_id;

  // Mix-precision tdma load/store from dialect
  // E.g. input int8 -> tiu bf16 -> output fp32
  conv->args.input_fmt = CVK_FMT_BF16;
  conv->args.output_fmt = CVK_FMT_BF16;
  conv->args.tiu_fmt = CVK_FMT_BF16;
  conv->args.ps32_output = fp32_output;
  conv->args.filter = filter;
  conv->args.new_filter = new_filter;
  conv->args.do_load_cmpr_wgt = false;
  if (filter != nullptr && !filter->empty()) {
    assert(new_filter && !new_filter->empty());
    conv->args.do_load_cmpr_wgt = true;
  }

  // Global memory region from dialect
  conv->initializeGlobalMem();

  conv->initializeFusedActivation();
  conv->initializeTile();

  // For tdma
  CV18xx::set_layer_id(layer_id);

  // Try depthwise convolution.
  if (conv->isDwConv()) {
    if (conv->determineDwTileSize(true, false)) {
      return conv->dwConv();
    } else
      assert(0 && "DwConv does not support single buffer yet");
  }

  conv->determineTilePolicy();
  conv->doConvByTilePolicy();
}
} // namespace backend
} // namespace tpu_mlir
