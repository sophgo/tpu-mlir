//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {
class OpDividePass : public OpDivideBase<OpDividePass> {
public:
  typedef struct {
    Operation *op;
    std::set<Operation *> op_set;
    int64_t max_size;
  } main_op_t;

  typedef struct {
    int out_h_start;
    int out_h;
    int in_h_start;
    int in_h;
    int num_backward;
    int num_forward;
    Operation *op; // op after slice;
  } h_slice_t;

  typedef struct {
    std::vector<h_slice_t> slice;
    int num_uses;
    int num_input;
    std::vector<int64_t> shape; // output shape
  } op_info_t;

  OpDividePass() {}

  void set_op_locs(FuncOp &fn) {
    uint16_t loc = 0;
    fn.walk([&](Operation *op) { ops_loc[op] = loc; });
  }

  static bool support(Operation *op) {
    if (op->getNumResults() == 0) {
      return false;
    }
    if (module::isUniformQuantized(op->getResult(0))) {
      if (isa<tpu::Conv2DOp>(op)) {
        return true;
      } else if (isa<tpu::AddOp>(op)) {
        // should be eltwise
        auto input1_shape = module::getShape(op->getOperand(0));
        auto input2_shape = module::getShape(op->getOperand(1));
        if (input1_shape.size() != input2_shape.size()) {
          return false;
        }
        for (int i = 0; i < input1_shape.size(); i++) {
          if (input1_shape[i] != input2_shape[i]) {
            return false;
          }
        }
        return true;
      } else if (isa<tpu::Pool2DOp>(op)) {
        auto poolOp = dyn_cast<tpu::Pool2DOp>(op);
        if (poolOp.getPoolMode() == tpu::PoolMode::Max) {
          return true;
        }
        return false;
      }
    }
    return false;
  }

  // static unsigned getOpLayerId(Operation *op) {
  //   if (op->getDialect()->getNamespace() == "tpu") {
  //     op->dump();
  //     auto loc = op->getLoc().dyn_cast<FileLineColLoc>();
  //     //auto loc = op->getLoc().cast<FileLineColLoc>();
  //     return loc.getLine();
  //   } else {
  //     std::string errorMsg = std::string(__func__) + " failed, Op " +
  //                           op->getName().getStringRef().str() + "\n";
  //     llvm_unreachable(errorMsg.c_str());
  //   }
  // }

  static int64_t getTensorSize(Value value) {
    // get int tensor size
    std::vector<int64_t> shape = value.getType().cast<TensorType>().getShape();
    return std::accumulate(std::begin(shape), std::end(shape), 1,
                           std::multiplies<>());
  }

  bool update_box(std::set<Operation *> &op_box, main_op_t &main_op) {
    if (op_box.size() == 1) {
      return true;
    }
    if (op_box.empty()) {
      return false;
    }
    Operation *update_op = *op_box.begin();
    uint32_t min_layer_id = ops_loc[update_op];
    for (auto op : op_box) {
      auto id = ops_loc[op];
      if (id < min_layer_id) {
        min_layer_id = id;
        update_op = op;
      }
    }
    op_box.erase(update_op);
    for (auto &use : update_op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      if (!support(sub_op)) {
        return false;
      }
      op_box.insert(sub_op);
      main_op.op_set.insert(sub_op);
      auto size = getTensorSize(sub_op->getResult(0));
      if (main_op.max_size < size) {
        main_op.max_size = size;
      }
    }
    return update_box(op_box, main_op);
  }

  static inline std::string getOpName(Operation *op) {
    return module::getName(op->getResult(0)).str();
  }

  inline bool start_slice() { return slice_idx == 0; }
  inline bool end_slice() { return slice_idx == num_slice - 1; }

  Operation *getNextMainOp(Operation *op, main_op_t &main_op) {
    std::set<Operation *> op_box;
    for (auto &use : op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      if (!support(sub_op)) {
        return nullptr;
      }
      op_box.insert(sub_op);
      main_op.op_set.insert(sub_op);
      auto size = getTensorSize(sub_op->getResult(0));
      if (main_op.max_size < size) {
        main_op.max_size = size;
      }
    }
    bool ret = update_box(op_box, main_op);
    if (ret == false) {
      return nullptr;
    }
    return *op_box.begin();
  }

  bool init_main_op(FuncOp &fn) {
    Operation *in_op = nullptr;
    bool multi_input = false;
    fn.walk([&](top::InputOp inputOp) {
      if (in_op == nullptr) {
        in_op = inputOp.getOperation();
      } else if (!multi_input) {
        // not support multi inpult
        multi_input = true;
      }
    });
    if (multi_input) {
      return false;
    }
    Operation *next_op = in_op;
    do {
      next_op = module::getNextOp(next_op);
    } while (next_op != nullptr && !isa<ReturnOp>(next_op) &&
             false == support(next_op));
    if (next_op == nullptr || isa<ReturnOp>(next_op)) {
      return false;
    }

    auto size = getTensorSize(next_op->getResult(0));
    main_op_t info = {
        .op = next_op,
        .op_set = {next_op},
        .max_size = size,
    };
    main_ops.clear();
    max_size = 0;
    do {
      if (max_size < info.max_size) {
        max_size = info.max_size;
      }
      main_ops.push_back(info);
      info.op = nullptr;
      info.max_size = 0;
      info.op_set.clear();
      next_op = getNextMainOp(next_op, info);
      info.op = next_op;
    } while (next_op != nullptr);
    // make sure max_size is really too large
    int64_t npu_memory = tpu_mlir::backend::CV18xx::NPU_NUM *
                         tpu_mlir::backend::CV18xx::LMEM_BYTES;
    if (max_size <= npu_memory) {
      return false;
    }
    if (main_ops.size() < 2) {
      return false;
    }
    start_idx = 0;
    end_idx = main_ops.size() - 1;

    init_last_size(fn);
    if (false == update_op_set()) {
      return false;
    }
    for (auto op : op_set) {
      op_info_t op_info;
      op_info.num_input = 0;
      op_info.num_uses = 0;
      op_info.slice.clear();
      op_info.shape = module::getShape(op->getResult(0));
      if (op_info.shape.size() < 3) {
        return false;
      }
      for (auto &use : op->getResult(0).getUses()) {
        if (op_set.find(use.getOwner()) != op_set.end()) {
          op_info.num_uses++;
        }
      }
      for (auto input : op->getOperands()) {
        if (op_set.find(input.getDefiningOp()) != op_set.end()) {
          op_info.num_input++;
        }
      }
      op_h_map[op] = op_info;
    }
    return true;
  }

  bool update_op_set() {
    auto start_size = main_ops[start_idx].max_size;
    if (start_size <= last_size) {
      while (start_idx < end_idx &&
             main_ops[start_idx + 1].max_size <= last_size) {
        start_idx++;
      }
    }
    auto end_size = main_ops[end_idx].max_size;
    if (end_size <= last_size) {
      while (start_idx < end_idx &&
             main_ops[end_idx - 1].max_size <= last_size) {
        end_idx--;
      }
    }
    if (start_idx >= end_idx) {
      return false;
    }
    op_set.clear();
    for (int i = start_idx; i <= end_idx; i++) {
      auto ops = main_ops[i].op_set;
      op_set.insert(ops.begin(), ops.end());
    }
    return true;
  }

  void init_last_size(FuncOp &fn) {
    op_set.clear();
    for (auto &info : main_ops) {
      op_set.insert(info.op_set.begin(), info.op_set.end());
    }
    last_size = getTensorSize(main_ops[end_idx].op->getResult(0));
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect()->getNamespace() != "tpu" ||
          isa<top::WeightOp>(op) || isa<top::NoneOp>(op) ||
          isa<top::InputOp>(op) || isa<ReturnOp>(op) || isa<FuncOp>(op)) {
      } else if (op_set.find(op) != op_set.end()) {
      } else {
        auto size = getTensorSize(op->getResult(0));
        if (last_size < size) {
          last_size = size;
        }
      }
    });
  }

  bool backward(Operation *op, int h_start, int h) {
    h_slice_t slice = {.out_h_start = h_start,
                       .out_h = h,
                       .in_h_start = h_start,
                       .in_h = h,
                       .num_backward = 1,
                       .num_forward = 0,
                       .op = nullptr};
    if (op_set.find(op) == op_set.end()) {
      // input
      return true;
    }
    auto &op_info = op_h_map[op];
    if (op_info.shape[2] < h * 2) {
      return false;
    }
    bool exist = (op_info.slice.size() > slice_idx);
    if (!exist) {
      op_info.slice.push_back(slice);
    }
    auto &s = op_info.slice[slice_idx];
    if (exist) {
      if (s.out_h < h) {
        s.out_h = h;
      }
      if (s.out_h_start > h_start) {
        s.out_h_start = h_start;
      }
      s.num_backward++;
    }
    // check all sub ops has backward, then do backward
    if (s.num_backward < op_info.num_uses) {
      return true;
    }

    // do backward
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::AddOp>(op)) {
      bool do_early_stride = false;
      if (cast_op.getDoEarlyStride().has_value()) {
        do_early_stride = cast_op.getDoEarlyStride().value();
      }
      uint32_t h_stride = 1;
      if (cast_op.getEarlyStrideH().has_value()) {
        h_stride = cast_op.getEarlyStrideH().value();
      }
      if (do_early_stride) {
        s.in_h_start = s.out_h_start * h_stride;
        s.in_h = s.out_h * h_stride;
      } else {
        s.in_h_start = s.out_h_start;
        s.in_h = s.out_h;
      }
      for (auto input : cast_op.getInputs()) {
        if (false == backward(input.getDefiningOp(), s.in_h_start, s.in_h)) {
          return false;
        }
      }
      return true;
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
      auto p = cast_op.parseParam();

      if (p.dh > 1) {
        p.kh = p.dh * (p.kh - 1) + 1;
      }
      s.in_h_start = (start_slice() ? 0 : s.out_h_start * p.sh - p.pht);
      s.in_h_start = std::max(s.in_h_start, 0);
      if (start_slice()) {
        s.in_h = (s.out_h - 1) * p.sh + p.kh - p.pht;
      } else if (end_slice()) {
        s.in_h = p.ih - s.in_h_start;
      } else {
        s.in_h = (s.out_h - 1) * p.sh + p.kh;
      }
      s.in_h = std::min(s.in_h, (int)p.ih);
      if (s.in_h_start + s.in_h > p.ih) {
        return false;
      }
      return backward(cast_op.getInput().getDefiningOp(), s.in_h_start, s.in_h);
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
      auto p = cast_op.parseParam();
      s.in_h_start = (start_slice() ? 0 : s.out_h_start * p.sh - p.pad_h);
      s.in_h_start = std::max(s.in_h_start, 0);
      if (start_slice()) {
        s.in_h = (s.out_h - 1) * p.sh + p.kh - p.pad_h;
      } else if (end_slice()) {
        s.in_h = p.ih - s.in_h_start;
      } else {
        s.in_h = (s.out_h - 1) * p.sh + p.kh;
      }
      s.in_h = std::min(s.in_h, (int)p.ih);
      return backward(cast_op.getInput().getDefiningOp(), s.in_h_start, s.in_h);
    }
    return false;
  }

  bool do_backward(int num_slice_, int last_h) {
    auto last_op = main_ops[end_idx].op;
    num_slice = num_slice_;
    if (num_slice < 2) {
      return false;
    }
    int h_step = (last_h + num_slice - 1) / num_slice;
    slice_idx = 0;
    for (int h_pos = 0; h_pos < last_h; h_pos += h_step, ++slice_idx) {
      auto h = std::min(h_step, last_h - h_pos);
      if (false == backward(last_op, h_pos, h)) {
        return false;
      }
    }
    for (auto &op : op_set) {
      if (op_h_map[op].slice.size() != num_slice) {
        // make sure all ops backward
        return false;
      }
    }
    return true;
  }

  bool do_backward() {
    auto last_op = main_ops[end_idx].op;
    auto shape = module::getShape(last_op->getResult(0));
    int last_h = shape[2];
    uint32_t max_slice = (max_size + last_size - 1) / last_size;
    max_slice = std::min(32u, std::min(max_slice, (uint32_t)(last_h / 3)));
    for (auto s = max_slice; s >= 2; s--) {
      if (do_backward(s, last_h)) {
        return true;
      }
      for (auto &op_info : op_h_map) {
        op_info.second.slice.clear();
      }
    }
    return false;
  }

  bool do_slice(FuncOp &fn) {
    while (do_backward() == false) {
      auto size = main_ops[end_idx].max_size;
      end_idx--;
      if (last_size < size) {
        last_size = size;
        if (last_size * 2 > max_size) {
          return false;
        }
        if (false == update_op_set()) {
          return false;
        }
      }
      if (start_idx >= end_idx) {
        return false;
      }
      for (auto &op_info : op_h_map) {
        op_info.second.slice.clear();
      }
    }
    return true;
  }

  Value create_slice_op(OpBuilder &builder, Operation *op, int h_start,
                        int h_slice) {
    auto op_out = op->getResult(0);
    // builder.setInsertionPointAfterValue(op_out);
    auto shape = module::getShape(op_out);
    std::vector<int64_t> slice_shape(shape.begin(), shape.end());
    slice_shape[2] = h_slice;
    std::string name = getOpName(op) + "_tod_crop_" + std::to_string(slice_idx);
    auto loc = NameLoc::get(builder.getStringAttr(name));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> offset(shape.size(), 0);
    offset[2] = h_start;
    attrs.emplace_back(
        builder.getNamedAttr("offset", builder.getI64ArrayAttr(offset)));
    attrs.emplace_back(
        builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1, 1, 1})));
    attrs.emplace_back(builder.getNamedAttr(
        "ends", builder.getI64ArrayAttr({-1, -1, -1, -1})));
    std::vector<Value> operands;
    operands.emplace_back(op_out);
    auto none = module::getNoneOp(op);
    operands.emplace_back(none);
    operands.emplace_back(none);
    operands.emplace_back(none);
    operands.emplace_back(none);
    auto quant_type = module::getUniformQuantizedType(op_out);
    auto type = RankedTensorType::get(slice_shape, quant_type);
    auto sliceOp = builder.create<tpu::SliceOp>(loc, type, operands, attrs);
    return sliceOp.getResult();
  }

  Value adjust_input(OpBuilder &builder, Operation *op, Value input,
                     h_slice_t &s) {
    auto input_op = input.getDefiningOp();
    auto shape = module::getShape(input_op->getResult(0));
    if (op_set.find(input_op) == op_set.end() || op == main_ops[start_idx].op) {
      if (shape[2] == s.in_h) {
        return input;
      }
      return create_slice_op(builder, input_op, s.in_h_start, s.in_h);
    }
    auto &s_in = op_h_map[input_op].slice[slice_idx];
    if (s.in_h == s_in.out_h && s.in_h_start == s_in.out_h_start) {
      return s_in.op->getResult(0);
    }
    return create_slice_op(builder, s_in.op, s.in_h_start - s_in.out_h_start,
                           s.in_h);
  }

  void concat_all(FuncOp &fn, OpBuilder &builder) {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto last_op = main_ops[end_idx].op;
    auto &slice = op_h_map[last_op].slice;
    for (auto &s : slice) {
      operands.push_back(s.op->getResult(0));
    }
    attrs.emplace_back(
        builder.getNamedAttr("axis", builder.getSI32IntegerAttr(2)));
    std::vector<int64_t> multipliers(slice.size(), 1);
    std::vector<int64_t> rshifts(slice.size(), 0);
    attrs.emplace_back(builder.getNamedAttr(
        "multipliers", builder.getI64ArrayAttr(multipliers)));
    attrs.emplace_back(
        builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshifts)));
    auto new_op = builder.create<tpu::ConcatOp>(
        last_op->getLoc(), last_op->getResult(0).getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    last_op->replaceAllUsesWith(new_op.getOperation());
    op_to_erase.insert(op_to_erase.end(), weight_set.begin(), weight_set.end());
    for (auto &op : op_to_erase) {
      op->erase();
    }
  }

  // Value copyFilter(Value ori_filter) {
  //   auto filterOp = cast<top::WeightOp>(ori_filter.getDefiningOp());
  //   auto filter_data = filterOp.read<int8_t>();
  //   auto filter_type = ori_filter.getType().cast<RankedTensorType>();
  //   auto new_filter = top::WeightOp::create(op, "filter_i8", *filter_data,
  //   filter_type);
  // }

  void forward(OpBuilder &builder, Operation *op) {
    if (op_set.find(op) == op_set.end()) {
      return;
    }
    auto &op_info = op_h_map[op];
    auto &s = op_info.slice[slice_idx];
    s.num_forward++;
    if (s.num_forward < op_info.num_input && op != main_ops[start_idx].op) {
      return;
    }

    auto origin_shape = module::getShape(op->getResult(0));
    std::vector<int64_t> output_shape(origin_shape.begin(), origin_shape.end());
    output_shape[2] = s.out_h;
    auto quant_type = module::getUniformQuantizedType(op->getResult(0));
    auto type = RankedTensorType::get(output_shape, quant_type);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    std::string op_name = getOpName(op);
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::AddOp>(op)) {
      for (auto input : cast_op.getInputs()) {
        auto in = adjust_input(builder, op, input, s);
        operands.push_back(in);
      }
      std::string name = op_name + "_tod_" + std::to_string(slice_idx);
      for (auto &attr : op->getAttrs()) {
        attrs.emplace_back(attr);
      }
      auto loc = NameLoc::get(builder.getStringAttr(name));
      auto newOp =
          builder.create<tpu::AddOp>(loc, type, ArrayRef<Value>{operands},
                                     ArrayRef<NamedAttribute>{attrs});
      s.op = newOp.getOperation();
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
      auto in = adjust_input(builder, op, cast_op.getInput(), s);
      operands.push_back(in);
      // operands.push_back(cast_op.getFilter());
      // operands.push_back(cast_op.getBias());
      // copy filter
      auto ori_filter = cast_op.getFilter();
      auto filterOp = cast<top::WeightOp>(ori_filter.getDefiningOp());
      auto filter_data = filterOp.read<int8_t>();
      auto filter_type = ori_filter.getType().cast<RankedTensorType>();
      auto new_filter = top::WeightOp::create(
          op, "_tod_" + std::to_string(slice_idx), *filter_data, filter_type);
      operands.push_back(new_filter);
      // copy bias
      auto biasOp = cast_op.getBias().getDefiningOp();
      if (!isa<top::NoneOp>(biasOp)) {
        auto bias_weight_op = cast<top::WeightOp>(biasOp);
        // because weight reorder, bias_data is int8_t
        auto bias_data = bias_weight_op.read<int8_t>();
        auto bias_type = cast_op.getBias().getType().cast<RankedTensorType>();
        auto new_bias = top::WeightOp::create(
            op, "_tod_" + std::to_string(slice_idx), *bias_data, bias_type);
        operands.push_back(new_bias);
      } else {
        operands.push_back(module::getNoneOp(op));
      }
      std::string name = op_name + "_tod_" + std::to_string(slice_idx);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      for (auto &attr : op->getAttrs()) {
        attrs.emplace_back(attr);
      }
      auto newOp =
          builder.create<tpu::Conv2DOp>(loc, type, ArrayRef<Value>{operands},
                                        ArrayRef<NamedAttribute>{attrs});
      auto pads = module::getI64Array(newOp.getPads()); // top,left,bottom,right
      std::vector<int64_t> new_pads(pads->begin(), pads->end());
      if (start_slice() == false && origin_shape[2] != s.out_h) {
        new_pads[0] = 0;
      }
      if (end_slice() == false && origin_shape[2] != s.out_h) {
        new_pads[2] = 0;
      }
      newOp.setPadsAttr(builder.getI64ArrayAttr(new_pads));
      s.op = newOp.getOperation();
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
      for (auto &attr : op->getAttrs()) {
        attrs.emplace_back(attr);
      }
      auto in = adjust_input(builder, op, cast_op.getInput(), s);
      operands.push_back(in);
      std::string name = op_name + "_tod_" + std::to_string(slice_idx);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      auto newOp =
          builder.create<tpu::Pool2DOp>(loc, type, ArrayRef<Value>{operands},
                                        ArrayRef<NamedAttribute>{attrs});
      auto pads = module::getI64Array(newOp.getPads()); // top,left,bottom,right
      std::vector<int64_t> new_pads(pads->begin(), pads->end());
      if (start_slice() == false && origin_shape[2] != s.out_h) {
        new_pads[0] = 0;
      }
      if (end_slice() == false && origin_shape[2] != s.out_h) {
        new_pads[2] = 0;
      }
      newOp.setPadsAttr(builder.getI64ArrayAttr(new_pads));
      s.op = newOp.getOperation();
    }
    for (auto &use : op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      forward(builder, sub_op);
    }
    if (end_slice()) {
      op_to_erase.push_back(op);
    }
  }

  void do_process(FuncOp &fn, OpBuilder &builder) {
    llvm::errs() << "============ tg op divide ===========================\n";
    for (int i = start_idx; i <= end_idx; i++) {
      auto &info = main_ops[i];
      llvm::errs() << "op:" << getOpName(info.op) << ", size: " << info.max_size
                   << "\n";
    }
    llvm::errs() << "max_size: " << max_size << ", last_size: " << last_size
                 << "\ndivide to [" << num_slice << "] pieces\n";
    builder.setInsertionPointAfter(main_ops[end_idx].op);
    for (slice_idx = 0; slice_idx < num_slice; slice_idx++) {
      forward(builder, main_ops[start_idx].op);
    }
    concat_all(fn, builder);
  }

  void runOnOperation() override {
    if (!module::isCV18xx()) {
      return;
    }
    auto *context = &getContext();
    auto builder = OpBuilder(context);
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      auto fn = module::getMainFuncOp(s);
      set_op_locs(fn);
      if (init_main_op(fn) == false) {
        llvm::errs() << "tg-op-divide op set failed\n";
        return;
      }
      if (do_slice(fn) == false) {
        llvm::errs() << "tg-op-divide slice failed\n";
        return;
      }
      do_process(fn, builder);
    }
  }

private:
  uint32_t num_slice;
  uint32_t slice_idx;
  std::map<Operation *, uint32_t> ops_loc;
  std::vector<main_op_t> main_ops;
  std::map<Operation *, op_info_t> op_h_map;
  std::set<Operation *> op_set;
  std::set<Operation *> weight_set;
  std::vector<Operation *> op_to_erase;
  int start_idx;
  int end_idx;
  int64_t max_size;
  int64_t last_size;
};

std::unique_ptr<OperationPass<ModuleOp>> createOpDividePass() {
  return std::make_unique<OpDividePass>();
}
} // namespace tpu
} // namespace tpu_mlir
