// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- AdjustIntegerWidthPass.cpp ------------------------------*- C++//-*-===//
//
// Pass to adjust integer widths of operations.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pass to
/// 1) Legalize 64-bit integer values to 32-bit integers values.
/// 2) Legalize !spv.array containing i1 type to !spv.array of i32 types.
/// 1) Legalize 8-bit integer values to 32-bit integers values.
/// TODO(b/144743561): Use Int8 capability after it is well-supported.
struct AdjustIntegerWidthPass
    : public PassWrapper<AdjustIntegerWidthPass, OperationPass<>> {
  void runOnOperation() override;
};

// Returns true if the type contains any IntegerType of the width specified by
// `widths`
bool hasIntTypeOfWidth(Type type, ArrayRef<int64_t> widths) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return llvm::is_contained(widths, intType.getWidth());
  } else if (auto structType = type.dyn_cast<spirv::StructType>()) {
    for (int64_t i = 0, e = structType.getNumElements(); i != e; ++i) {
      if (hasIntTypeOfWidth(structType.getElementType(i), widths)) return true;
    }
    return false;
  } else if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    return hasIntTypeOfWidth(arrayType.getElementType(), widths);
  } else if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    return hasIntTypeOfWidth(ptrType.getPointeeType(), widths);
  }
  return false;
}

// Returns true if the type contains i1, i8, i16, or i64.
bool hasSupportedIntegerType(Type type) {
  return hasIntTypeOfWidth(type, {1, 8, 16, 64});
}

// Legalizes all the integer types in struct to i32.
Type legalizeIntegerType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return IntegerType::get(32, intType.getContext());
  } else if (auto structType = type.dyn_cast<spirv::StructType>()) {
    SmallVector<Type, 1> elementTypes;
    for (auto i : llvm::seq<unsigned>(0, structType.getNumElements())) {
      elementTypes.push_back(legalizeIntegerType(structType.getElementType(i)));
    }
    // TODO(ravishankarm): Use ABI attributes to legalize the struct type.
    return VulkanLayoutUtils::decorateType(
        spirv::StructType::get(elementTypes));
  } else if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    return spirv::ArrayType::get(
        legalizeIntegerType(arrayType.getElementType()),
        arrayType.getNumElements());
  } else if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    return spirv::PointerType::get(
        legalizeIntegerType(ptrType.getPointeeType()),
        ptrType.getStorageClass());
  }
  return type;
}

/// Rewrite access chain operations where the pointee type contains i1 or i64
/// types.
struct AdjustAccessChainOp : public OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern<spirv::AccessChainOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::AccessChainOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasSupportedIntegerType(op.component_ptr().getType())) {
      return failure();
    }
    ValueRange indices(op.indices());
    Type newType = legalizeIntegerType(op.component_ptr().getType());
    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(op, newType,
                                                      op.base_ptr(), indices);
    return success();
  }
};

/// Rewrite address of operations which refers to global variables that contain
/// i1 or i64 types.
struct AdjustAddressOfOp : public OpRewritePattern<spirv::AddressOfOp> {
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::AddressOfOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasSupportedIntegerType(op.pointer().getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, legalizeIntegerType(op.pointer().getType()),
        SymbolRefAttr::get(op.variable(), rewriter.getContext()));
    return success();
  }
};

/// Rewrite global variable ops that contains i1, i8, i16 and i64 types to i32
/// type.
struct AdjustGlobalVariableWidth
    : public OpRewritePattern<spirv::GlobalVariableOp> {
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::GlobalVariableOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasSupportedIntegerType(op.type())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, legalizeIntegerType(op.type()), op.sym_name(),
        op.getAttr("descriptor_set").cast<IntegerAttr>().getInt(),
        op.getAttr("binding").cast<IntegerAttr>().getInt());
    return success();
  }
};

// Returns an adjusted spirv::AccessChainOp to access corresponding i32
// elements. One element was a `bits`-bit integer. The method adjust the last
// index to make it access the corresponding i32 element. Note that this only
// works for a scalar or 1-D tensor.
Value convertToI32AccessChain(spirv::AccessChainOp op, int bits,
                              PatternRewriter &rewriter) {
  assert(32 % bits == 0);
  const auto loc = op.getLoc();
  auto i32Type = rewriter.getIntegerType(32);
  auto idx = rewriter.create<spirv::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(32 / bits));
  auto lastDim = op.getOperation()->getOperand(op.getNumOperands() - 1);
  SmallVector<Value, 4> indices;
  for (auto it : op.indices()) {
    indices.push_back(it);
  }
  if (indices.size() > 1) {
    indices.back() = rewriter.create<spirv::SDivOp>(loc, lastDim, idx);
  }
  Type t = legalizeIntegerType(op.component_ptr().getType());
  return rewriter.create<spirv::AccessChainOp>(loc, t, op.base_ptr(), indices);
}

// Returns the offset of input value in i32 representation. For example, if
// `bits` equals to 8, the x-th element is located at (x % 4) * 8. Because there
// are four elements in one i32, and one element has 8 bits.
Value getOffsetOfInt(spirv::AccessChainOp op, int bits,
                     PatternRewriter &rewriter) {
  assert(32 % bits == 0);
  const auto loc = op.getLoc();
  Type i32Type = rewriter.getIntegerType(32);
  auto idx = rewriter.create<spirv::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(32 / bits));
  auto num = rewriter.create<spirv::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(bits));
  auto lastDim = op.getOperation()->getOperand(op.getNumOperands() - 1);
  auto m = rewriter.create<spirv::SModOp>(loc, lastDim, idx);
  return rewriter.create<spirv::IMulOp>(loc, i32Type, m, num);
}

Value rewriteIntForLoadOp(spirv::LoadOp op, PatternRewriter &rewriter) {
  const auto loc = op.getLoc();
  Type valueType = op.value().getType();
  Type i32Type = rewriter.getIntegerType(32);
  Value result;
  auto accessChainOp = cast<spirv::AccessChainOp>(op.ptr().getDefiningOp());
  // Only support for scalar and 1-D tensor. The first element in indices is
  // index, the remaining elements map to other dimensions.
  if (accessChainOp.indices().size() > 2) {
    return nullptr;
  }

  int bits = hasIntTypeOfWidth(valueType, {1, 8}) ? 8 : 16;
  Value i32AccessChainOp =
      convertToI32AccessChain(accessChainOp, bits, rewriter);
  Value loadOp = rewriter.create<spirv::LoadOp>(
      loc, i32Type, i32AccessChainOp,
      op.getAttrOfType<IntegerAttr>(
          spirv::attributeName<spirv::MemoryAccess>()),
      op.getAttrOfType<IntegerAttr>("alignment"));

  // If it is a scalar, use the loading value directly. Otherwise, extract
  // corresponding bits out. If it is a scalar, the indices only contains one
  // element (which is index).
  if (accessChainOp.indices().size() == 1) {
    result = loadOp;
  } else {
    Value offset = getOffsetOfInt(accessChainOp, bits, rewriter);
    result = rewriter.create<spirv::ShiftRightArithmeticOp>(loc, i32Type,
                                                            loadOp, offset);
  }

  auto intMax = rewriter.create<spirv::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr((1 << bits) - 1));
  result = rewriter.create<spirv::BitwiseAndOp>(loc, i32Type, result, intMax);

  // If this is a load of a i1, replace it with a load of i8, and truncate the
  // result. Use INotEqualOp because SConvert doesn't work for i1.
  if (hasIntTypeOfWidth(valueType, {1})) {
    Type newType = legalizeIntegerType(valueType);
    auto zero = spirv::ConstantOp::getZero(newType, loc, &rewriter);
    result = rewriter.create<spirv::INotEqualOp>(loc, valueType, result, zero)
                 .getResult();
  }

  return result;
}

/// Rewrite loads from !spv.ptr<i64,..> to load from !spv.ptr<i32,...>
/// Rewrite loads from !spv.ptr<i1,...> to load from !spv.ptr<i32,...> followed
/// by a truncate to i1 type.
/// Rewrite loads from !spv.ptr<i8,...> to load from !spv.ptr<i32,...> followed
/// by an extraction.
/// Rewrite loads from !spv.ptr<i16,...> to load from !spv.ptr<i32,...> followed
/// by an extraction.
struct AdjustLoadOp : public OpRewritePattern<spirv::LoadOp> {
  using OpRewritePattern<spirv::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Type valueType = op.value().getType();
    if (!hasSupportedIntegerType(valueType)) {
      return failure();
    }

    Type newType = legalizeIntegerType(valueType);
    const auto loc = op.getLoc();
    Value result;
    if (hasIntTypeOfWidth(valueType, {1, 8, 16})) {
      result = rewriteIntForLoadOp(op, rewriter);
    } else {
      auto loadOp = rewriter.create<spirv::LoadOp>(
          loc, newType, op.ptr(),
          op.getAttrOfType<IntegerAttr>(
              spirv::attributeName<spirv::MemoryAccess>()),
          op.getAttrOfType<IntegerAttr>("alignment"));
      result = loadOp.getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Returns the shifted 32-bit value with the given offset.
Value shiftStoreValue(spirv::StoreOp op, const Value &offset, const Value &mask,
                      PatternRewriter &rewriter) {
  Type valueType = op.value().getType();
  Type i32Type = rewriter.getIntegerType(32);
  const auto loc = op.getLoc();

  Value storeVal = op.value();
  if (hasIntTypeOfWidth(valueType, {1})) {
    Value zero =
        spirv::ConstantOp::getZero(i32Type, loc, &rewriter).getResult();
    Value one = spirv::ConstantOp::getOne(i32Type, loc, &rewriter).getResult();
    storeVal =
        rewriter.create<spirv::SelectOp>(loc, storeVal, one, zero).getResult();
  } else {
    storeVal = rewriter.create<spirv::SConvertOp>(loc, i32Type, storeVal);
    storeVal = rewriter.create<spirv::BitwiseAndOp>(loc, storeVal, mask);
  }
  return rewriter.create<spirv::ShiftLeftLogicalOp>(loc, i32Type, storeVal,
                                                    offset);
}

// Rewrites store operation that contains i1, i8 and i16 types to i32 type.
// Since there are multi threads in the processing, atomic operations are
// required. E.g., if the loading value is i8, rewrite the StoreOp to
// 1) load a 32-bit integer
// 2) clear 8 bits in the loading value
// 3) store 32-bit value back
// 4) load a 32-bit integer
// 5) modify 8 bits in the loading value
// 6) store 32-bit value back
// The step 1 to step 3 are done by AtomicAnd, and the step 4 to
// step 6 are done by AtomicOr.
LogicalResult rewriteIntForStoreOp(spirv::StoreOp op,
                                   PatternRewriter &rewriter) {
  Type valueType = op.value().getType();
  Type i32Type = rewriter.getIntegerType(32);
  const auto loc = op.getLoc();
  auto accessChainOp = cast<spirv::AccessChainOp>(op.ptr().getDefiningOp());

  // Only support for scalar and 1-D tensor. The first element in indices is
  // index, the remaining elements map to other dimensions.
  if (accessChainOp.indices().size() > 2) {
    return failure();
  }

  int bits = hasIntTypeOfWidth(valueType, {1, 8}) ? 8 : 16;
  auto offset = getOffsetOfInt(accessChainOp, bits, rewriter);

  // Create a mask to clear the destination. E.g., if it is the second i8 in
  // i32, 0xFFFF00FF is created.
  auto mask = rewriter.create<spirv::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr((1 << bits) - 1));
  Value clear8BitMask =
      rewriter.create<spirv::ShiftLeftLogicalOp>(loc, i32Type, mask, offset);
  clear8BitMask = rewriter.create<spirv::NotOp>(loc, i32Type, clear8BitMask);

  Value storeVal = shiftStoreValue(op, offset, mask, rewriter);
  Value i32AccessChainOp =
      convertToI32AccessChain(accessChainOp, bits, rewriter);
  Value result = rewriter.create<spirv::AtomicAndOp>(
      loc, i32Type, i32AccessChainOp, spirv::Scope::Device,
      spirv::MemorySemantics::AcquireRelease, clear8BitMask);
  result = rewriter.create<spirv::AtomicOrOp>(
      loc, i32Type, i32AccessChainOp, spirv::Scope::Device,
      spirv::MemorySemantics::AcquireRelease, storeVal);

  // The AtomicOrOp has no side effect. Since it is already inserted, we can
  // just remove the original StoreOp. Note that rewriter.replaceOp()
  // doesn't work because it only accepts that the numbers of result are the
  // same.
  rewriter.eraseOp(op);

  return success();
}

/// Rewrite store operation that contain i1, i8 and i64 types to i32 type.
struct AdjustStoreOp : public OpRewritePattern<spirv::StoreOp> {
  using OpRewritePattern<spirv::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Type valueType = op.value().getType();
    if (!hasSupportedIntegerType(valueType)) {
      return failure();
    }

    if (hasIntTypeOfWidth(valueType, {1, 8, 16})) {
      if (failed(rewriteIntForStoreOp(op, rewriter))) return failure();
    } else {
      const auto loc = op.getLoc();
      auto i32Type = rewriter.getIntegerType(32);
      auto value = rewriter.create<spirv::SConvertOp>(loc, i32Type, op.value());
      rewriter.replaceOpWithNewOp<spirv::StoreOp>(
          op, op.ptr(), value,
          op.getAttrOfType<IntegerAttr>(
              spirv::attributeName<spirv::MemoryAccess>()),
          op.getAttrOfType<IntegerAttr>("alignment"));
    }

    return success();
  }
};

/// Some Adjust* OpRewritePattern will generate useless SConvert operations,
/// which are invalid operations. Remove the SConvert operation if this is an
/// nop, i.e., if the source type and destination type are the same, remove the
/// op. It relies on the furthur finialization to remove the op, and propagate
/// right operands to other operations.
struct RemoveNopSConvertOp : public OpRewritePattern<spirv::SConvertOp> {
  using OpRewritePattern<spirv::SConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::SConvertOp op,
                                PatternRewriter &rewriter) const override {
    Type t1 = op.operand().getType();
    Type t2 = op.result().getType();
    if (t1 != t2) return failure();
    auto zero = spirv::ConstantOp::getZero(t1, op.getLoc(), &rewriter);
    rewriter.replaceOpWithNewOp<spirv::IAddOp>(op, op.operand(), zero);
    return success();
  }
};

/// Rewrite SConvert operation that the target type is i8, i16 or i64.
struct AdjustSConvertOp : public OpRewritePattern<spirv::SConvertOp> {
  using OpRewritePattern<spirv::SConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::SConvertOp op,
                                PatternRewriter &rewriter) const override {
    Type t = op.result().getType();
    if (!hasIntTypeOfWidth(t, {8, 16, 64})) {
      return failure();
    }
    Type i32Type = rewriter.getIntegerType(32);
    rewriter.replaceOpWithNewOp<spirv::SConvertOp>(op, i32Type, op.operand());
    return success();
  }
};

/// Rewrite i64 constants to i32 constants.
struct AdjustConstantOp : public OpRewritePattern<spirv::ConstantOp> {
  using OpRewritePattern<spirv::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::ConstantOp op,
                                PatternRewriter &rewriter) const {
    Type constantType = op.getType();
    if (!hasIntTypeOfWidth(constantType, {8, 16, 64})) {
      return failure();
    }

    Value i32cst;
    if (auto attr = op.value().dyn_cast<IntegerAttr>()) {
      Type i32Type = rewriter.getIntegerType(32);
      auto i32Attr = IntegerAttr::get(i32Type, attr.getInt());
      i32cst =
          rewriter.create<spirv::ConstantOp>(op.getLoc(), i32Type, i32Attr);
    } else {
      llvm_unreachable("only support splat constant");
    }

    rewriter.replaceOpWithNewOp<spirv::SConvertOp>(op, constantType, i32cst);
    return success();
  }
};

/// Rewrite integer arithmetic operations that operate on 64-bit integers to
/// operate on 32-bit integers.
template <typename OpTy>
struct AdjustIntegerArithmeticOperations : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const {
    Type resultType = op.result().getType();
    if (!hasIntTypeOfWidth(resultType, {8, 16, 64})) {
      return failure();
    }
    Type newType = legalizeIntegerType(op.getResult().getType());
    ValueRange operands(op.getOperation()->getOperands());
    rewriter.replaceOpWithNewOp<OpTy>(op, newType, operands, op.getAttrs());
    return success();
  }
};

/// Rewrite SelectOp so that result is a 32-bit integer.
struct AdjustSelectOp : public OpRewritePattern<spirv::SelectOp> {
  using OpRewritePattern<spirv::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spirv::SelectOp op,
                                PatternRewriter &rewriter) const {
    Type resultType = op.result().getType();
    if (!hasSupportedIntegerType(resultType)) {
      return failure();
    }
    Type newType = legalizeIntegerType(op.getResult().getType());
    ValueRange operands(op.getOperation()->getOperands());
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, newType, operands,
                                                 op.getAttrs());
    return success();
  }
};

void AdjustIntegerWidthPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<
      // Arithmetic ops:
      AdjustIntegerArithmeticOperations<spirv::GLSLSAbsOp>,
      AdjustIntegerArithmeticOperations<spirv::GLSLSMaxOp>,
      AdjustIntegerArithmeticOperations<spirv::GLSLSMinOp>,
      AdjustIntegerArithmeticOperations<spirv::GLSLSSignOp>,
      AdjustIntegerArithmeticOperations<spirv::IAddOp>,
      AdjustIntegerArithmeticOperations<spirv::ISubOp>,
      AdjustIntegerArithmeticOperations<spirv::IMulOp>,
      AdjustIntegerArithmeticOperations<spirv::SDivOp>,
      AdjustIntegerArithmeticOperations<spirv::SModOp>,
      AdjustIntegerArithmeticOperations<spirv::SRemOp>,
      AdjustIntegerArithmeticOperations<spirv::UDivOp>,
      AdjustIntegerArithmeticOperations<spirv::UModOp>,
      // Control flow ops:
      AdjustSelectOp,
      // Structure ops:
      AdjustConstantOp,
      // Others:
      AdjustAccessChainOp, AdjustAddressOfOp, AdjustGlobalVariableWidth,
      AdjustLoadOp, AdjustStoreOp, RemoveNopSConvertOp, AdjustSConvertOp>(
      &getContext());
  Operation *op = getOperation();
  applyPatternsAndFoldGreedily(op->getRegions(), patterns);
}

static PassRegistration<AdjustIntegerWidthPass> pass(
    "iree-spirv-adjust-integer-width",
    "Adjust integer width from all integer types to i32 type");

}  // namespace

std::unique_ptr<Pass> createAdjustIntegerWidthPass() {
  return std::make_unique<AdjustIntegerWidthPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
