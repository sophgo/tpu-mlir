//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Support/ModuleEnum.h.inc"
#include "tpu_mlir/Support/Logger.h"

using namespace mlir;
using namespace mlir::func;
using namespace tpu_mlir;

namespace tpu_mlir {

typedef enum {
  /* 1. 3D group if this group has CONV3D/DECONV3D/POOL3D
   * for 1684 float32, data in local memory storage as {d * n, c, h, w}
   * for 1684 int8, data in local memory storage as {n, d * c, h, w}
   * for 1684X, data in local memory storage as {d * n, c, h, w}
   * 2. data in global memory always storage as {n, c, d, h, w}
   * 3. GROUP_SMALL_C: move h to c-dim, and merge cd-dim to n-dim
   *    1) case1: {n, c, h, w} --> {n * c, h, w, 1}
   *    2) case2: {n, c, d, h, w} --> {n * c * d, h, w, 1}
   * 4. GROUP_MM_INT4: for INT4 matrix multiplication
   * 5. GROUP_MM: split N/C/H in order if this group has  matrix multiplication
   *    1) for MM2_NN, split left matrix C-dim
   *    2) for MM2_NT, split left/right matrix C-dim
   *    3) for MM2_TT, split right matrix C-dim
   * group_type < 8, because 1684 dynamic compile reserved `3bit` for group_type
   */
  GROUP_NORMAL = 0,
  GROUP_3D = 1,
  GROUP_SMALL_C = 2,
  GROUP_MM_INT4 = 3,
  GROUP_MM = 4,
  GROUP_UNSUPPORT
} group_type_t;

//-----------------------------------------------------------------
// Types
//-----------------------------------------------------------------
typedef std::shared_ptr<std::vector<int32_t>> i32_array_t;
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
typedef std::shared_ptr<std::vector<double>> f64_array_t;
namespace module {

// init module by ModuleOp in init pass
void init(ModuleOp module);
void init_loglevel(int32_t log_level);
void setWeightInMemFlag(bool enable);
bool getWeightInMemFlag();

extern std::unordered_map<std::string, int> patternMatchCounts;
extern std::mutex patternMatchCountsMutex;

//-----------------------------------------------------------------
// Helper for debug information
//-----------------------------------------------------------------
void assert_with_dump(bool cond, Operation *op, const char *info,
                      const char *file = nullptr, unsigned line = 0);
void unreachable(const char *info, Operation *op = nullptr,
                 const char *file = nullptr, unsigned line = 0);

//-----------------------------------------------------------------
// Helper for get/set Attributes
//-----------------------------------------------------------------
int64_t getCoreNum();
void setCoreNum(int64_t core_num = 1);
int64_t getDeviceNum();
void setDeviceNum(int64_t device_num = 1);

Chip getChip();
void setChip(Chip chip);
bool isChip(Chip chip);
Mode getMode();
bool getTrain();
void setMode(Mode mode);
State getState();
void setState(State state);
bool isState(State state);
bool isSubnetDividedState();
void setAddrMode(AddrMode mode);
AddrMode getAddrMode();
bool isAddrMode(AddrMode mode);
void setTopRunMode(TopRunMode mode);
TopRunMode getTopRunMode();
bool isDynamic();
bool isTrain();
void setTrain(bool is_train);
bool isDebugCmdEnable(std::string cmd_str);
void setInputs(ArrayRef<StringRef> inputs);
std::shared_ptr<std::vector<StringRef>> getInputs();
void setOutputs(ArrayRef<StringRef> outputs);
std::shared_ptr<std::vector<StringRef>> getOutputs();
bool isBF16Modes();
bool isF16Modes();
bool isF8Modes();

Platform getPlatform();
bool isPlatform(Platform plt);

int64_t getFLOPs();
void setFLOPs(int64_t flops);
bool isAsymmetric();
void setAsymmetric(bool is_asymmetric);
int getQuantGroupSize();
void setQuantGroupSize(int q_group_size);
llvm::StringRef getPostprocess();
void setPostprocess(StringRef post);

//-----------------------------------------------------------------
// Helper Functions for ModuleOp
//-----------------------------------------------------------------

ModuleOp getModuleOp();
ModuleOp getModuleOp(Operation *op); // get moduleop that op belong to
Location getLoc();
MLIRContext *getCtx();

top::NoneOp getNoneOp(Operation *op);
Value getOriValue(Value v);
Operation *getNextOp(Operation *op, int i = 0);
Value getOperand(Operation *op, int i);
bool isSameOp(Operation *op0, Operation *op1);
void updateModuleTypes();
void removeUnusedOp();
int64_t getAddress(Value v);
void setAddress(Value v, int64_t addr);
void set8chAddress(Value v, size_t index, int64_t offset, int64_t addr);
void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align = true);
void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align = true);
void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, group_type_t group_type);
void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             group_type_t group_type);
void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h,
              int64_t &w, group_type_t group_type);
double getDtypeSize(Value v);
size_t getBytes(Value v);
int64_t getNumElements(Value v);
Type getStorageType(Value v); // storage type
Type getStorageType(Type type);
Type getElementType(Value v);
RankedTensorType getTypeLike(Value v, llvm::ArrayRef<int64_t> shape);

void setShape(Value v, llvm::ArrayRef<int64_t> shape);
llvm::ArrayRef<int64_t> getShape(Value v);
std::vector<int64_t> getShapeVec(Value v);
void getGlobalShape(Value v, int *shape, int dim = 4);
void getLocalShape(Value v, int64_t n_step, int64_t h_step, int *shape);
void getLocalShape(Operation *op, int64_t n_step, int64_t h_step, int *shape);
void get128BtyeAlignedStrideForNBit(int *stride, int *shape, int npu_num,
                                    int bit);
void getCompactStride(int *stride, int *shape, int npu_num);
void getContinousStride(int *stride, int *shape);
bool isUnranked(Value v);
void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape);
bool isSign(Value v);
bool isWeight(Value v);
bool isActive(Value v);
bool isDynWeight(Value v);
bool isShapeRelatedOp(Value v);
bool isAllWeight(Operation *op);
bool isNone(Value v);
bool isGlobalBuffer(Value v);
FuncOp getMainFuncOp(ModuleOp module);
i32_array_t getI32Array(ArrayAttr arrayAttr);
i32_array_t getI32Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value);
i64_array_t getI64Array(ArrayAttr arrayAttr);
i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value);
f64_array_t getF64Array(ArrayAttr arrayAttr);
f64_array_t getF64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value);
bool isOpInGroup(Operation *Op, int64_t *group_type = nullptr);
bool isOpInCoreMatch(Operation *Op);
bool isOpInCoreParallel(Operation *Op);
bool isOpInDevParallel(Operation *Op);
bool isOpInBlock(Operation *op);
FuncOp getFuncOp(ModuleOp module, StringRef func_name);
func::CallOp getCallOp(FuncOp func);
llvm::StringRef getName(Operation *op, int index = 0);
llvm::StringRef getName(Value v);
uint32_t getIdx(Value v);
NameLoc getLoc(Value v);
NameLoc getLocLike(Operation *op, llvm::StringRef suffix);
NameLoc getLocLike(Value v, llvm::StringRef suffix);
void setLocSuffix(Operation *op, llvm::StringRef suffix);
void setLoc(Value v, NameLoc loc);
void getInputsOutputs(ModuleOp submodule, std::vector<Value> &inputs,
                      std::vector<Value> &outputs);
void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs);
void removeAttr(mlir::Operation *op, std::string attr_name);

bool isTpuOp(Operation *op);
bool isInt4Op(Operation *op);
bool isCV18xx();
bool isBM1684Family();
bool isBM1684XFamily();
bool isBM1690Family();
bool isSG2380();
bool isBM1688();
bool isBM1684X();
bool isMARS3();

//-----------------------------------------------------------------
// Helper Functions for submodule
//-----------------------------------------------------------------
int getNumSubModule();
std::shared_ptr<std::vector<ModuleOp>> getAllModules();
void setSubModuleId(ModuleOp submodule, int64_t device_id, int64_t step);
void getSubModuleId(ModuleOp submodule, int64_t &device_id, int64_t &step);

int64_t getNeuronSize(ModuleOp submodule);
void setNeuronSize(ModuleOp submodule, int64_t size);
int64_t getNeuronAddr(ModuleOp submodule);
void setNeuronAddr(ModuleOp submodule, int64_t addr);

int64_t getIOSize(ModuleOp submodule);
void setIOSize(ModuleOp submodule, int64_t size);
int64_t getIOAddr(ModuleOp submodule);
void setIOAddr(ModuleOp submodule, int64_t addr);

int64_t getCoeffSize(ModuleOp submodule);
void setCoeffSize(ModuleOp submodule, int64_t size);
int64_t getCoeffAddr(ModuleOp submodule);
void setCoeffAddr(ModuleOp submodule, int64_t addr);
int64_t getDynamicOffset(ModuleOp submodule);
void setDynamicOffset(ModuleOp submodule, int64_t size);

int64_t getGmemPrivateSize(ModuleOp submodule);
void setGmemPrivateSize(ModuleOp submodule, int64_t size);
//-----------------------------------------------------------------
// Helper Functions for op translate
//-----------------------------------------------------------------
mlir::Value opSliceAxis(PatternRewriter &rewriter, mlir::Value v, int64_t axis,
                        int64_t offset, int64_t length,
                        std::string mode = "default");

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
mlir::TensorFile &weightFile();
void setWeightFileName(const std::string &name);
void saveWeight();
void detachWeightFile();
void setWeightFileAttr(const std::string &name);
llvm::StringRef getWeightFileAttr();

//-----------------------------------------------------------------
// Helper Functions for apply pattern only once
//-----------------------------------------------------------------
template <typename T> void applyPatternOnce(ModuleOp m) {
  auto ctx = m.getContext();
  RewritePatternSet patterns(ctx);
  auto config = GreedyRewriteConfig();
  config.maxIterations = 1; // apply each pattern only once.
  patterns.add<T>(ctx);
  applyPatternsAndFoldGreedily(m, std::move(patterns), config);
}

//-----------------------------------------------------------------
// Helper Functions for quantization
//-----------------------------------------------------------------
bool isCalibratedType(Type type);
bool isCalibratedType(Value v);
bool isUniformQuantized(Type type);
bool isUniformQuantized(Value v);
template <typename... Args> bool isCalibratedType(Value v, Args... args) {
  return isCalibratedType(v) && isCalibratedType(args...);
}
template <typename... Args> bool isUniformQuantized(Value v, Args... args) {
  return isUniformQuantized(v) && isUniformQuantized(args...);
}

quant::CalibratedQuantizedType getCalibratedType(Value v);
quant::CalibratedQuantizedType getCalibratedType(Type t);
quant::UniformQuantizedType getUniformQuantizedType(Value v);
quant::UniformQuantizedType getUniformQuantizedType(Type t);
double getThreshold(Value v);

// for symmetric
double getScale(double threshold, bool sign, int bitwidth = 8);
// for asymmetric
void getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                          int64_t &zeroPoint, int bitwidth = 8);
void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool asymmetric, int bitwidth = 8);
void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool &sign, bool asymmetric, int bitwidth = 8);

// for Value with shape [1] but means scalar
bool isScalar(mlir::Operation *op);

//-----------------------------------------------------------------
// Helper for shape op inference
//-----------------------------------------------------------------
class ShapeHelper {
private:
  ShapeHelper(){};
  ~ShapeHelper(){};
  ShapeHelper(const ShapeHelper &);
  ShapeHelper &operator=(const ShapeHelper &);

public:
  static ShapeHelper &getInstance() {
    static ShapeHelper instance;
    return instance;
  }

  void bindShapeInfo(const Value &v, const std::vector<int64_t> &shape);
  std::vector<int64_t> getShapeInfo(const Value &v);
  bool isShape(const Value &v);

private:
  llvm::DenseMap<Value, std::vector<int64_t>> _shape_info;
};

void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape);
std::vector<int64_t> getShapeTensorValue(const Value &v);
bool isShape(const Value &v);
std::vector<int64_t>
commonShapeValInfer(mlir::Operation *op,
                    const std::vector<std::vector<int64_t>> &in_shapes_v,
                    const std::vector<int64_t> &out_shape);
bool startsWith(const std::string& fullString, const std::string& startingSubstring);
bool endsWith(const std::string& fullString, const std::string& suffix);
} // namespace module

#define ASSERT_OP(COND, OP)                                                    \
  module::assert_with_dump(COND, OP, #COND, __FILE__, __LINE__)
#define ASSERT_THIS(COND) ASSERT_OP(COND, this->getOperation()) // in Op inner functions
#define UNREACHABLE_OP(INFO, OP) module::unreachable(INFO, OP, __FILE__, __LINE__)
#define UNREACHABLE_THIS(INFO) UNREACHABLE_OP(INFO, this->getOperation())

} // namespace tpu_mlir
