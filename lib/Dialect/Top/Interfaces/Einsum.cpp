//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::EinsumOp::getFLOPs() {
  llvm_unreachable("GetFLOPs Not Implemented");
  return 0;
}

LogicalResult top::EinsumOp::init(InferenceParameter &p) { return success(); }

void top::EinsumOp::deinit(InferenceParameter &p) {}

LogicalResult top::EinsumOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return failure();
}

void top::EinsumOp::shape_inference() {
  auto mode = getMode().str();
  ASSERT_THIS(getInputs().size() == 2 || getInputs().size() == 3);
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  if (mode == "a,b->ab") { // outer product
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], rhs_shape[0]});
  } else if (mode == "ab,ab->a") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0]});
  } else if (mode == "ab,acb->ac") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], rhs_shape[1]});
  } else if (mode == "ab,cdb->acd") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], rhs_shape[0], rhs_shape[1]});
  } else if (mode == "abc,db->adc") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], rhs_shape[0], lhs_shape[2]});
  } else if (mode == "abcd,cde->abe") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], lhs_shape[1], rhs_shape[2]});
  } else if (mode == "abcd,acd->abc") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], lhs_shape[1], lhs_shape[2]});
  } else if (mode == "abcd,bed->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[1]});
  } else if (mode == "abcd,ced->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[0], rhs_shape[1]});
  } else if (mode == "abcd,abed->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[2]});
  } else if (mode == "abcd,abde->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[3]});
  } else if (mode == "abcd,abce->acde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[2], lhs_shape[3], rhs_shape[3]});
  } else if (mode == "abc,adc->abd") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], lhs_shape[1], rhs_shape[1]});
  } else if (mode == "abc,adc->adb") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], rhs_shape[1], lhs_shape[1]});
  } else if (mode == "abc,abd->acd") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], lhs_shape[2], rhs_shape[2]});
  } else if (mode == "abcd,cde->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[2]});
  } else if (mode == "abc,acde->abde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[2], rhs_shape[3]});
  } else if (mode == "abc,abde->acde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[2], rhs_shape[2], rhs_shape[3]});
  } else if (mode == "abc,bd->abcd") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[1]});
  } else if (mode == "abc,abdc,abc->abcd") {
    module::setShapeOrVerify(
        getOutput(), {rhs_shape[0], rhs_shape[1], rhs_shape[3], rhs_shape[2]});
  } else if (mode == "abcd,acde,abc->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[3]});
  } else if (mode == "abc,abc->ab") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], lhs_shape[1]});
  } else if (mode == "abcd,aecd->aeb") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], rhs_shape[1], lhs_shape[1]});
  } else if (mode == "abcd,aecd->abec") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[1], rhs_shape[2]});
  } else if (mode == "abcd,aecd->acbe") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[2], lhs_shape[1], rhs_shape[1]});
  } else if (mode == "abcd,acde->abde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[2], rhs_shape[3]});
  } else if (mode == "abc,cde->abde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[1], rhs_shape[2]});
  } else if (mode == "abcd,aeb->aecd") {
    module::setShapeOrVerify(
        getOutput(), {rhs_shape[0], rhs_shape[1], lhs_shape[2], lhs_shape[3]});
  } else if (mode == "abcde,afbc->abdef") {
    module::setShapeOrVerify(
        getOutput(),
        {lhs_shape[0], lhs_shape[1], lhs_shape[3], lhs_shape[4], rhs_shape[1]});
  } else if (mode == "ab,cbdef->cadef") {
    module::setShapeOrVerify(
        getOutput(),
        {rhs_shape[0], lhs_shape[0], rhs_shape[2], rhs_shape[3], rhs_shape[4]});
  } else if (mode == "abcd,cebd->abce" || mode == "abcd,ecbd->abec") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[0], rhs_shape[1]});
  } else if (mode == "abcd,cdbe->abce") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[3]});
  } else if (mode == "abcde,abfge->abcdfg") {
    module::setShapeOrVerify(getOutput(),
                             {lhs_shape[0], lhs_shape[1], lhs_shape[2],
                              lhs_shape[3], rhs_shape[2], rhs_shape[3]});
  } else if (mode == "abcd,abef->acdef") {
    module::setShapeOrVerify(
        getOutput(),
        {lhs_shape[0], lhs_shape[2], lhs_shape[3], rhs_shape[2], rhs_shape[3]});
  } else if (mode == "abcd,abce->abde") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[3], rhs_shape[3]});
  } else if (mode == "abcd,adbe->acbe") {
    module::setShapeOrVerify(
        getOutput(), {lhs_shape[0], lhs_shape[2], lhs_shape[1], rhs_shape[3]});
  } else if (mode == "ab,abc->ac") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], rhs_shape[2]});
  } else {
    llvm_unreachable("Not support now.");
  }
}
