# 设计随笔

## 要求

* 用vscode编辑代码，提交前都格式化一下

* 所有文档用markdown编写

* td文件的注视要完整，以便自动生成文档

* 所有对外的工具都用python，（runtime工具除外）

* 尽可能减少git submodule;第三方优先直接在docker里面安装(如pybind11)，不能安装的则使用编译后的库（如llvm,oneDNN)

* 将通用代码和专用代码(与TPU型号绑定的代码)严格分开。专用代码尽可能按规格参数转换成通用代码。

* 所有内部写的代码，都要加上CopyRight开头。

## 设计

* 用mlir自带的文档生成工具生成td文档

* 保证calibration高效：

  * operation的资源申请与forward过程分离；
  * interpet能用oneDNN的，用oneDNN；不能用oneDNN的，尽量用OMP
  * oneDNN做int8推理的时候也用fp32运算，避免其他类型运算难以定位

* mlir默认整型是INT64，默认浮点型是F64。 attribute的类型尽量用默认类型。好处如下：

``` mlir
# 用INT32
%4 = "top.Conv"(%1, %2, %3) {kernel_shape = [3 : i32, 3 : i32], name = "output_Conv"}
# 用INT64
%4 = "top.Conv"(%1, %2, %3) {kernel_shape = [3, 3], name = "output_Conv"}
```

* 为了可读性，头文件尽量不用宏定义成员或方法，源文件可以用宏定义

* regression除了basic测试时会跑calibation，其余测试均使用calibration之后的文件，避免测试时间过长

#### 调试方法
``` shell
gdb --args python /workspace/tpu-mlir/python/tools/model_runner.py --input resnet18_in_f32.npz --model resnet18_opt.mlir --output resnet18_out.npz
```

## 待解决问题

* docker需要配置成用普通用户权限

* dnnl pool与conv参考matmul优化

* 目前所有op都认为是一个输出，后续支持多输出op ?

* top的pass只有一个：Fuse Relu到conv和add，其他pass需要补充完整

* 目前shape还不支持动态shape和shape推导，后续支持

* vscode的pybind11路径配置没有配好，Python.h也没有定位好

* tpuc-opt为什么编译时间这么久，需要研究一下

* 为了省时间，直接用晶视的npz_tool/calibation_tool，后续再做优化。其中cali的min和max直接取最小值和最大值，是否也应该做kld算法求min/max？另外tune没有加入，作用很大，后续再加入。

* resnet18.onnx (40多M，目前用于测试，后续删掉)

* install内容很多，后期再整理一下

* 自定义ModuleOp，支持当前的属性，mlir.state用enum类型表达

* 当return的op类型被改动后，需要手写代码把FuncOp的类型同步，不然出错；应该有自动的方法。待优化（如ImportCalibrationTablePass)

* mlir近期Verifier.cpp加入了一个OpTrait::IsTerminator的判断，会触发assert，目前现象来看像是mlir自身的bug。先删掉这个判断，不会照成任何影响。后期再升级llvm，再看会不会触发这个问题。

* 目前1684用batch 4验证，避免在4N存储的细节上花费太多时间；后续需要补上4N的逻辑

* BM前缀是否都需要改成SG前缀？

* conv各种3ic的优化 ？

## 一些思考

#### 是否第一层用TOSA Dialect ?
不用：
如果由客户转，增加客户工作量；
tosa定义的op还是太少，没有lrn、layernorm等等op；
自己定义top dialect，可以完全掌控

#### caffe需要前端转吗？
需要：
目前caffe转onnx没有组织维护，支持的算子特别少

#### 开源后的亮点

1 top层和quant层采用oneDNN和omp推导结果，做验证
2 工具全部用python，方便使用和维护，以及与第三方应用对接
3 可以随时更新到llvm最新版本

#### 是否要支持预处理？以及将预处理合并到tpu layer?

预处理事情比较多，先不做预处理的考虑，假定用户的输入是预处理后的数据；
后续时间充足再考虑预处理的事情，边端处理器还是需要的，用户从vpss出来后就需要直接导入到模型，
预处理只能有tpu完成。

#### backend库是动态加载还是静态加载？

1684和1684x的库的函数名都是重复的，静态加载会有符号冲突。所以cv18xx和bm16xx都用动态加载
