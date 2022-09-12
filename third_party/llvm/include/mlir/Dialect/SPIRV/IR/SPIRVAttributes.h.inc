/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace spirv {
class AddressingModelAttr;
class ImageArrayedInfoAttr;
class BuiltInAttr;
class CapabilityAttr;
class CooperativeMatrixPropertiesNVAttr;
class DecorationAttr;
class ImageDepthInfoAttr;
class DeviceTypeAttr;
class DimAttr;
class EntryPointABIAttr;
class ExecutionModeAttr;
class ExecutionModelAttr;
class ExtensionAttr;
class FunctionControlAttr;
class GroupOperationAttr;
class ImageFormatAttr;
class ImageOperandsAttr;
class JointMatrixPropertiesINTELAttr;
class LinkageTypeAttr;
class LoopControlAttr;
class MatrixLayoutAttr;
class MemoryAccessAttr;
class MemoryModelAttr;
class MemorySemanticsAttr;
class OpcodeAttr;
class ResourceLimitsAttr;
class ImageSamplerUseInfoAttr;
class ImageSamplingInfoAttr;
class ScopeAttr;
class SelectionControlAttr;
class StorageClassAttr;
class VendorAttr;
class VersionAttr;
namespace detail {
struct AddressingModelAttrStorage;
} // namespace detail
class AddressingModelAttr : public ::mlir::Attribute::AttrBase<AddressingModelAttr, ::mlir::Attribute, detail::AddressingModelAttrStorage> {
public:
  using Base::Base;
public:
  static AddressingModelAttr get(::mlir::MLIRContext *context, ::mlir::spirv::AddressingModel value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"addressing_model"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::AddressingModel getValue() const;
};
namespace detail {
struct ImageArrayedInfoAttrStorage;
} // namespace detail
class ImageArrayedInfoAttr : public ::mlir::Attribute::AttrBase<ImageArrayedInfoAttr, ::mlir::Attribute, detail::ImageArrayedInfoAttrStorage> {
public:
  using Base::Base;
public:
  static ImageArrayedInfoAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageArrayedInfo value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_arrayed_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageArrayedInfo getValue() const;
};
namespace detail {
struct BuiltInAttrStorage;
} // namespace detail
class BuiltInAttr : public ::mlir::Attribute::AttrBase<BuiltInAttr, ::mlir::Attribute, detail::BuiltInAttrStorage> {
public:
  using Base::Base;
public:
  static BuiltInAttr get(::mlir::MLIRContext *context, ::mlir::spirv::BuiltIn value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"built_in"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::BuiltIn getValue() const;
};
namespace detail {
struct CapabilityAttrStorage;
} // namespace detail
class CapabilityAttr : public ::mlir::Attribute::AttrBase<CapabilityAttr, ::mlir::Attribute, detail::CapabilityAttrStorage> {
public:
  using Base::Base;
public:
  static CapabilityAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Capability value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"capability"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Capability getValue() const;
};
namespace detail {
struct CooperativeMatrixPropertiesNVAttrStorage;
} // namespace detail
class CooperativeMatrixPropertiesNVAttr : public ::mlir::Attribute::AttrBase<CooperativeMatrixPropertiesNVAttr, ::mlir::Attribute, detail::CooperativeMatrixPropertiesNVAttrStorage> {
public:
  using Base::Base;
public:
  static CooperativeMatrixPropertiesNVAttr get(::mlir::MLIRContext *context, int m_size, int n_size, int k_size, mlir::Type a_type, mlir::Type b_type, mlir::Type c_type, mlir::Type result_type, mlir::spirv::ScopeAttr scope);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"coop_matrix_props"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getMSize() const;
  int getNSize() const;
  int getKSize() const;
  mlir::Type getAType() const;
  mlir::Type getBType() const;
  mlir::Type getCType() const;
  mlir::Type getResultType() const;
  mlir::spirv::ScopeAttr getScope() const;
};
namespace detail {
struct DecorationAttrStorage;
} // namespace detail
class DecorationAttr : public ::mlir::Attribute::AttrBase<DecorationAttr, ::mlir::Attribute, detail::DecorationAttrStorage> {
public:
  using Base::Base;
public:
  static DecorationAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Decoration value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"decoration"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Decoration getValue() const;
};
namespace detail {
struct ImageDepthInfoAttrStorage;
} // namespace detail
class ImageDepthInfoAttr : public ::mlir::Attribute::AttrBase<ImageDepthInfoAttr, ::mlir::Attribute, detail::ImageDepthInfoAttrStorage> {
public:
  using Base::Base;
public:
  static ImageDepthInfoAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageDepthInfo value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_depth_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageDepthInfo getValue() const;
};
namespace detail {
struct DeviceTypeAttrStorage;
} // namespace detail
class DeviceTypeAttr : public ::mlir::Attribute::AttrBase<DeviceTypeAttr, ::mlir::Attribute, detail::DeviceTypeAttrStorage> {
public:
  using Base::Base;
public:
  static DeviceTypeAttr get(::mlir::MLIRContext *context, ::mlir::spirv::DeviceType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"device_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::DeviceType getValue() const;
};
namespace detail {
struct DimAttrStorage;
} // namespace detail
class DimAttr : public ::mlir::Attribute::AttrBase<DimAttr, ::mlir::Attribute, detail::DimAttrStorage> {
public:
  using Base::Base;
public:
  static DimAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Dim value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dim"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Dim getValue() const;
};
namespace detail {
struct EntryPointABIAttrStorage;
} // namespace detail
class EntryPointABIAttr : public ::mlir::Attribute::AttrBase<EntryPointABIAttr, ::mlir::Attribute, detail::EntryPointABIAttrStorage> {
public:
  using Base::Base;
public:
  static EntryPointABIAttr get(::mlir::MLIRContext *context, DenseIntElementsAttr local_size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"entry_point_abi"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DenseIntElementsAttr getLocalSize() const;
};
namespace detail {
struct ExecutionModeAttrStorage;
} // namespace detail
class ExecutionModeAttr : public ::mlir::Attribute::AttrBase<ExecutionModeAttr, ::mlir::Attribute, detail::ExecutionModeAttrStorage> {
public:
  using Base::Base;
public:
  static ExecutionModeAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ExecutionMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"execution_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ExecutionMode getValue() const;
};
namespace detail {
struct ExecutionModelAttrStorage;
} // namespace detail
class ExecutionModelAttr : public ::mlir::Attribute::AttrBase<ExecutionModelAttr, ::mlir::Attribute, detail::ExecutionModelAttrStorage> {
public:
  using Base::Base;
public:
  static ExecutionModelAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ExecutionModel value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"execution_model"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ExecutionModel getValue() const;
};
namespace detail {
struct ExtensionAttrStorage;
} // namespace detail
class ExtensionAttr : public ::mlir::Attribute::AttrBase<ExtensionAttr, ::mlir::Attribute, detail::ExtensionAttrStorage> {
public:
  using Base::Base;
public:
  static ExtensionAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Extension value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ext"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Extension getValue() const;
};
namespace detail {
struct FunctionControlAttrStorage;
} // namespace detail
class FunctionControlAttr : public ::mlir::Attribute::AttrBase<FunctionControlAttr, ::mlir::Attribute, detail::FunctionControlAttrStorage> {
public:
  using Base::Base;
public:
  static FunctionControlAttr get(::mlir::MLIRContext *context, ::mlir::spirv::FunctionControl value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"function_control"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::FunctionControl getValue() const;
};
namespace detail {
struct GroupOperationAttrStorage;
} // namespace detail
class GroupOperationAttr : public ::mlir::Attribute::AttrBase<GroupOperationAttr, ::mlir::Attribute, detail::GroupOperationAttrStorage> {
public:
  using Base::Base;
public:
  static GroupOperationAttr get(::mlir::MLIRContext *context, ::mlir::spirv::GroupOperation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"group_operation"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::GroupOperation getValue() const;
};
namespace detail {
struct ImageFormatAttrStorage;
} // namespace detail
class ImageFormatAttr : public ::mlir::Attribute::AttrBase<ImageFormatAttr, ::mlir::Attribute, detail::ImageFormatAttrStorage> {
public:
  using Base::Base;
public:
  static ImageFormatAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageFormat value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_format"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageFormat getValue() const;
};
namespace detail {
struct ImageOperandsAttrStorage;
} // namespace detail
class ImageOperandsAttr : public ::mlir::Attribute::AttrBase<ImageOperandsAttr, ::mlir::Attribute, detail::ImageOperandsAttrStorage> {
public:
  using Base::Base;
public:
  static ImageOperandsAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageOperands value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_operands"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageOperands getValue() const;
};
namespace detail {
struct JointMatrixPropertiesINTELAttrStorage;
} // namespace detail
class JointMatrixPropertiesINTELAttr : public ::mlir::Attribute::AttrBase<JointMatrixPropertiesINTELAttr, ::mlir::Attribute, detail::JointMatrixPropertiesINTELAttrStorage> {
public:
  using Base::Base;
public:
  static JointMatrixPropertiesINTELAttr get(::mlir::MLIRContext *context, int m_size, int n_size, int k_size, mlir::Type a_type, mlir::Type b_type, mlir::Type c_type, mlir::Type result_type, mlir::spirv::ScopeAttr scope);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"joint_matrix_props"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getMSize() const;
  int getNSize() const;
  int getKSize() const;
  mlir::Type getAType() const;
  mlir::Type getBType() const;
  mlir::Type getCType() const;
  mlir::Type getResultType() const;
  mlir::spirv::ScopeAttr getScope() const;
};
namespace detail {
struct LinkageTypeAttrStorage;
} // namespace detail
class LinkageTypeAttr : public ::mlir::Attribute::AttrBase<LinkageTypeAttr, ::mlir::Attribute, detail::LinkageTypeAttrStorage> {
public:
  using Base::Base;
public:
  static LinkageTypeAttr get(::mlir::MLIRContext *context, ::mlir::spirv::LinkageType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"linkage_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::LinkageType getValue() const;
};
namespace detail {
struct LoopControlAttrStorage;
} // namespace detail
class LoopControlAttr : public ::mlir::Attribute::AttrBase<LoopControlAttr, ::mlir::Attribute, detail::LoopControlAttrStorage> {
public:
  using Base::Base;
public:
  static LoopControlAttr get(::mlir::MLIRContext *context, ::mlir::spirv::LoopControl value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"loop_control"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::LoopControl getValue() const;
};
namespace detail {
struct MatrixLayoutAttrStorage;
} // namespace detail
class MatrixLayoutAttr : public ::mlir::Attribute::AttrBase<MatrixLayoutAttr, ::mlir::Attribute, detail::MatrixLayoutAttrStorage> {
public:
  using Base::Base;
public:
  static MatrixLayoutAttr get(::mlir::MLIRContext *context, ::mlir::spirv::MatrixLayout value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matrixLayout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::MatrixLayout getValue() const;
};
namespace detail {
struct MemoryAccessAttrStorage;
} // namespace detail
class MemoryAccessAttr : public ::mlir::Attribute::AttrBase<MemoryAccessAttr, ::mlir::Attribute, detail::MemoryAccessAttrStorage> {
public:
  using Base::Base;
public:
  static MemoryAccessAttr get(::mlir::MLIRContext *context, ::mlir::spirv::MemoryAccess value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_access"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::MemoryAccess getValue() const;
};
namespace detail {
struct MemoryModelAttrStorage;
} // namespace detail
class MemoryModelAttr : public ::mlir::Attribute::AttrBase<MemoryModelAttr, ::mlir::Attribute, detail::MemoryModelAttrStorage> {
public:
  using Base::Base;
public:
  static MemoryModelAttr get(::mlir::MLIRContext *context, ::mlir::spirv::MemoryModel value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_model"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::MemoryModel getValue() const;
};
namespace detail {
struct MemorySemanticsAttrStorage;
} // namespace detail
class MemorySemanticsAttr : public ::mlir::Attribute::AttrBase<MemorySemanticsAttr, ::mlir::Attribute, detail::MemorySemanticsAttrStorage> {
public:
  using Base::Base;
public:
  static MemorySemanticsAttr get(::mlir::MLIRContext *context, ::mlir::spirv::MemorySemantics value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_semantics"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::MemorySemantics getValue() const;
};
namespace detail {
struct OpcodeAttrStorage;
} // namespace detail
class OpcodeAttr : public ::mlir::Attribute::AttrBase<OpcodeAttr, ::mlir::Attribute, detail::OpcodeAttrStorage> {
public:
  using Base::Base;
public:
  static OpcodeAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Opcode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"opcode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Opcode getValue() const;
};
namespace detail {
struct ResourceLimitsAttrStorage;
} // namespace detail
class ResourceLimitsAttr : public ::mlir::Attribute::AttrBase<ResourceLimitsAttr, ::mlir::Attribute, detail::ResourceLimitsAttrStorage> {
public:
  using Base::Base;
public:
  static ResourceLimitsAttr get(::mlir::MLIRContext *context, int max_compute_shared_memory_size, int max_compute_workgroup_invocations, ArrayAttr max_compute_workgroup_size, int subgroup_size, ArrayAttr cooperative_matrix_properties_nv);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"resource_limits"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getMaxComputeSharedMemorySize() const;
  int getMaxComputeWorkgroupInvocations() const;
  ArrayAttr getMaxComputeWorkgroupSize() const;
  int getSubgroupSize() const;
  ArrayAttr getCooperativeMatrixPropertiesNv() const;
};
namespace detail {
struct ImageSamplerUseInfoAttrStorage;
} // namespace detail
class ImageSamplerUseInfoAttr : public ::mlir::Attribute::AttrBase<ImageSamplerUseInfoAttr, ::mlir::Attribute, detail::ImageSamplerUseInfoAttrStorage> {
public:
  using Base::Base;
public:
  static ImageSamplerUseInfoAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageSamplerUseInfo value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_sampler_use_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageSamplerUseInfo getValue() const;
};
namespace detail {
struct ImageSamplingInfoAttrStorage;
} // namespace detail
class ImageSamplingInfoAttr : public ::mlir::Attribute::AttrBase<ImageSamplingInfoAttr, ::mlir::Attribute, detail::ImageSamplingInfoAttrStorage> {
public:
  using Base::Base;
public:
  static ImageSamplingInfoAttr get(::mlir::MLIRContext *context, ::mlir::spirv::ImageSamplingInfo value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"image_sampling_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::ImageSamplingInfo getValue() const;
};
namespace detail {
struct ScopeAttrStorage;
} // namespace detail
class ScopeAttr : public ::mlir::Attribute::AttrBase<ScopeAttr, ::mlir::Attribute, detail::ScopeAttrStorage> {
public:
  using Base::Base;
public:
  static ScopeAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Scope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Scope getValue() const;
};
namespace detail {
struct SelectionControlAttrStorage;
} // namespace detail
class SelectionControlAttr : public ::mlir::Attribute::AttrBase<SelectionControlAttr, ::mlir::Attribute, detail::SelectionControlAttrStorage> {
public:
  using Base::Base;
public:
  static SelectionControlAttr get(::mlir::MLIRContext *context, ::mlir::spirv::SelectionControl value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"selection_control"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::SelectionControl getValue() const;
};
namespace detail {
struct StorageClassAttrStorage;
} // namespace detail
class StorageClassAttr : public ::mlir::Attribute::AttrBase<StorageClassAttr, ::mlir::Attribute, detail::StorageClassAttrStorage> {
public:
  using Base::Base;
public:
  static StorageClassAttr get(::mlir::MLIRContext *context, ::mlir::spirv::StorageClass value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"storage_class"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::StorageClass getValue() const;
};
namespace detail {
struct VendorAttrStorage;
} // namespace detail
class VendorAttr : public ::mlir::Attribute::AttrBase<VendorAttr, ::mlir::Attribute, detail::VendorAttrStorage> {
public:
  using Base::Base;
public:
  static VendorAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Vendor value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"vendor"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Vendor getValue() const;
};
namespace detail {
struct VersionAttrStorage;
} // namespace detail
class VersionAttr : public ::mlir::Attribute::AttrBase<VersionAttr, ::mlir::Attribute, detail::VersionAttrStorage> {
public:
  using Base::Base;
public:
  static VersionAttr get(::mlir::MLIRContext *context, ::mlir::spirv::Version value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"version"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::spirv::Version getValue() const;
};
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::AddressingModelAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageArrayedInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::BuiltInAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::CapabilityAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::CooperativeMatrixPropertiesNVAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::DecorationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageDepthInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::DeviceTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::DimAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::EntryPointABIAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ExecutionModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ExecutionModelAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ExtensionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::FunctionControlAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::GroupOperationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageFormatAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageOperandsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::JointMatrixPropertiesINTELAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::LinkageTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::LoopControlAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::MatrixLayoutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::MemoryAccessAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::MemoryModelAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::MemorySemanticsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::OpcodeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ResourceLimitsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageSamplerUseInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ImageSamplingInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ScopeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::SelectionControlAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::StorageClassAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::VendorAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::VersionAttr)

#endif  // GET_ATTRDEF_CLASSES

