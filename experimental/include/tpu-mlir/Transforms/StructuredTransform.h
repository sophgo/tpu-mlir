//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include <llvm/ADT/ilist.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

namespace tpu_mlir {
using namespace mlir;

struct ComputePattern {
  SmallVector<AffineMap> indexingMaps;
  SmallVector<utils::IteratorType> iteratorTypes;

  bool operator==(const ComputePattern &rhs) const {
    if (this->indexingMaps.size() != rhs.indexingMaps.size())
      return false;
    for (auto [a, b] : llvm::zip(this->indexingMaps, rhs.indexingMaps)) {
      if (a != b)
        return false;
    }
    for (auto [a, b] : llvm::zip(this->iteratorTypes, rhs.iteratorTypes)) {
      if (a != b)
        return false;
    }
    return true;
  }

  bool operator!=(const ComputePattern &rhs) const { return !(*this == rhs); }
};

class Transform {
public:
  enum TransformKind {
    TS_Unroll = 0,
    TS_DropSymbol = 1,
    TS_MergeDim = 2,
    TS_Permutation = 3,
    TS_ExpandDim = 4,
    TS_DecomposeExpr = 5,
  };

  virtual std::optional<ComputePattern> run(const ComputePattern &) = 0;
  virtual StringLiteral name() = 0;
  virtual void dump() = 0;
  virtual ~Transform() = default;

private:
  const TransformKind Kind;

public:
  TransformKind getKind() const { return Kind; }
  Transform(TransformKind K) : Kind(K) {}
};

struct Unroll : public Transform {
  const AffineDimExpr dimention;
  StringLiteral name() override { return "Unroll"; };
  void dump() override {
    llvm::errs() << name() << "(";
    dimention.print(llvm::errs());
    llvm::errs() << ")\n";
  };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  Unroll(AffineDimExpr dimention)
      : Transform(TS_Unroll), dimention(dimention) {}
  static bool classof(const Transform *T) { return T->getKind() == TS_Unroll; }
};

struct DropSymbol : public Transform {
  const AffineSymbolExpr symbol;
  StringLiteral name() override { return "DropSymbol"; };
  void dump() override {
    llvm::errs() << name() << "(";
    symbol.print(llvm::errs());
    llvm::errs() << ")\n";
  };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  DropSymbol(AffineSymbolExpr symbol)
      : Transform(TS_DropSymbol), symbol(symbol) {}
  static bool classof(const Transform *T) {
    return T->getKind() == TS_DropSymbol;
  }
};

struct MergeDims : public Transform {
  const AffineDimExpr dim1, dim2;
  StringLiteral name() override { return "MergeDims"; };
  void dump() override {
    llvm::errs() << name() << ":(";
    dim1.print(llvm::errs());
    llvm::errs() << ", ";
    dim2.print(llvm::errs());
    llvm::errs() << ")\n";
  };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  MergeDims(AffineDimExpr dim1, AffineDimExpr dim2)
      : Transform(TS_MergeDim), dim1(dim1), dim2(dim2) {}
  static bool classof(const Transform *T) {
    return T->getKind() == TS_MergeDim;
  }
};

struct Permutation : public Transform {
  const SmallVector<bool> runOnItems;
  const SmallVector<int64_t, 2> permuteDims;
  StringLiteral name() override { return "Permutation"; };
  void dump() override {
    llvm::errs() << name() << "(dim[" << permuteDims[0] << ", "
                 << permuteDims[1] << "])[";
    interleaveComma(runOnItems, llvm::errs());
    llvm::errs() << "]\n";
  };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  Permutation(ArrayRef<int64_t> dims, ArrayRef<bool> runOn)
      : Transform(TS_Permutation), permuteDims(dims), runOnItems(runOn) {}
  static bool classof(const Transform *T) {
    return T->getKind() == TS_Permutation;
  }
};

struct ExpandDims : public Transform {
  // expand  dim for each item in indexingMaps
  const SmallVector<bool> runOnItems;
  const utils::IteratorType iteratorType;
  StringLiteral name() override { return "ExpandDims"; };
  void dump() override {
    auto iType = (iteratorType == utils::IteratorType::parallel) ? "P" : "R";
    llvm::errs() << name() << ":" << iType << "(";
    interleaveComma(runOnItems, llvm::errs());
    llvm::errs() << ")\n";
  };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  ExpandDims(utils::IteratorType iType, ArrayRef<bool> runOn)
      : Transform(TS_ExpandDim), runOnItems(runOn), iteratorType(iType) {}
  static bool classof(const Transform *T) {
    return T->getKind() == TS_ExpandDim;
  }
};

struct DecomposeExpr : public Transform {
  // decompose add expression
  int64_t position;
  StringLiteral name() override { return "DecomposeExpr"; };
  void dump() override { llvm::errs() << name() << ":" << position << "\n"; };
  std::optional<ComputePattern> run(const ComputePattern &source) override;
  DecomposeExpr(int64_t position)
      : Transform(TS_DecomposeExpr), position(position) {}
  static bool classof(const Transform *T) {
    return T->getKind() == TS_DecomposeExpr;
  }
};

struct Transforms
    : public llvm::ilist_node_with_parent<Transforms, Transforms> {

  Transforms(std::unique_ptr<Transform> transform, Transforms *parent)
      : transform(std::move(transform)), parent(parent){};

  Transforms() : transform(nullptr), parent(nullptr){};

  Transforms(Transforms &&other) {
    transform = std::move(other.transform);
    children = std::move(other.children);
    parent = other.parent;
    // Correct the move-copy semantics of the class; update the root address to
    // maintain the correct parent address when a move-copy operation is
    // performed.
    if (!parent) {
      for (auto &child : children)
        child.parent = this;
    }
  };

  template <typename T, typename... Args>
  void add(Args... args) {
    children.push_back(new Transforms(std::make_unique<T>(args...), this));
  }

  template <typename T>
  void add(T &ts) {
    children.push_back(new Transforms(std::make_unique<T>(ts), this));
  }

  Transform &getTransfom() { return *transform; };
  // remove this node from the tree.
  void erase() {
    if (parent) {
      auto &sibling = parent->children;
      sibling.erase(this);
    }
  }

  void dump() { dump(llvm::errs(), 0); }

  size_t size() {
    if (children.empty() && parent)
      return 1;
    size_t count = 0;
    for (auto &i : children) {
      count += i.size();
    }
    return count;
  }
  bool isRoot() { return parent == nullptr; }
  Transforms *getParent() const { return parent; }

  using TsListType = llvm::ilist<Transforms>;

  TsListType &getChildren() { return children; };

  using iterator = TsListType::iterator;
  using reverse_iterator = TsListType::reverse_iterator;

  iterator begin() { return children.begin(); }
  iterator end() { return children.end(); }
  reverse_iterator rbegin() { return children.rbegin(); }
  reverse_iterator rend() { return children.rend(); }

  bool empty() { return children.empty(); }

  Transforms &back() { return children.back(); }
  Transforms &front() { return children.front(); }

private:
  Transforms(const Transforms &);
  // print transforms as a tree structure.
  void dump(raw_ostream &os, int indent = 0) {
    if (transform) { // remove root
      if (&(parent->children.front()) != this)
        os.indent((indent - 1) * 15);
      os << llvm::format("%-15s", transform->name().str().c_str());
      if (children.empty())
        os << "\n";
    }
    for (auto &child : children) {
      child.dump(os, indent + 1);
    }
  }
  std::unique_ptr<Transform> transform;
  Transforms *parent = nullptr;
  TsListType children;
};

struct Solver {

  Solver(ComputePattern &target, int64_t depth = 5)
      : target(target), maxDepth(depth) {
    context = target.indexingMaps[0].getContext();
  };

  // (transform*)source (linalg) -> target (SIMD)
  // transform := merge | drop d/s | permutation | expandDim | decomposeExpr
  // merge : (same iterator type) reshape
  // drop d/s := unroll | drop symbol dim
  // permutation := permute the access index
  // expandDim := insert rank 1 dim
  // decomposeExpr := decompose add expression
  Transforms solve(const ComputePattern source);
  size_t getAllPath() { return allPath; }

private:
  void driver(const ComputePattern &source, Transforms &transforms,
              int64_t depth = 0);

  template <typename T,
            typename = std::enable_if_t<std::is_base_of<Transform, T>::value>>
  void driver(T transform, const ComputePattern &source, Transforms &transforms,
              int64_t depth) {
    if (auto out = transform.run(source)) {
      transforms.add(transform);
      if (out.value() == target) {
        allPath += 1;
        return;
      }
      return driver(out.value(), transforms.back(), depth + 1);
    }
    allPath += 1;
  };

  MLIRContext *context;

  const ComputePattern target;
  // the paths have been tried.
  size_t allPath = 0;
  // This control the depth of the transform.
  int64_t maxDepth;
};

} // namespace tpu_mlir

// remove this code when we upgrade llvm to
// (1609f1c2a5ecc0e0e28f433ec9205122478ddaa3)
namespace llvm {
/// Add support for llvm style casts. We provide a cast between To and From if
/// From is mlir::AffineExpr or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_same_v<mlir::AffineExpr,
                                                std::remove_const_t<From>> ||
                                 std::is_base_of_v<mlir::AffineExpr, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {

  static inline bool isPossible(mlir::AffineExpr expr) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return expr.isa<To>();
    }
  }
  static inline To doCast(mlir::AffineExpr expr) {
    return To((mlir::AffineExpr::ImplType *)(expr.getAsOpaquePointer()));
  }
};
} // namespace llvm
