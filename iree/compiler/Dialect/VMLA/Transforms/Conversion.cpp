// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Patterns.h"
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/Conversion/HALToVMLA/ConvertHALToVMLA.h"
#include "iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/ConvertHLOToVMLA.h"
#include "iree/compiler/Dialect/VMLA/Conversion/StandardToVMLA/ConvertStandardToVMLA.h"
#include "iree/compiler/Dialect/VMLA/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

static LogicalResult insertInterfacesToEntryPoints(mlir::ModuleOp moduleOp) {
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (SymbolTable::getSymbolVisibility(funcOp) !=
        SymbolTable::Visibility::Public) {
      continue;
    }
    auto originalType = funcOp.getType();
    if (originalType.getNumInputs() != 0 || originalType.getNumResults() != 0) {
      return funcOp.emitError() << "exported functions must have no I/O";
    }
    auto interfaceType = IREE::VMLA::InterfaceType::get(moduleOp.getContext());
    auto newType =
        FunctionType::get({interfaceType}, {}, moduleOp.getContext());
    funcOp.setType(newType);
    funcOp.front().addArgument(interfaceType);
  }
  return success();
}

// Runs conversion with registered input dialects.
class ConversionPass
    : public PassWrapper<ConversionPass, OperationPass<mlir::ModuleOp>> {
 public:
  void runOnOperation() override {
    // First insert vmla.interface arguments to all exported functions.
    // The conversions require that the interface argument is present in order
    // to properly retrieve buffer bindings.
    if (failed(insertInterfacesToEntryPoints(getOperation()))) {
      return signalPassFailure();
    }

    auto *context = &getContext();
    VMLATypeConverter typeConverter;
    VMLAConversionTarget conversionTarget(context, typeConverter);

    // Ensure all input dialects go away.
    conversionTarget.addIllegalDialect<xla_hlo::XlaHloDialect>();
    conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();

    OwningRewritePatternList conversionPatterns;
    populateStandardToVMLAPatterns(context, conversionPatterns, typeConverter);
    populateHLOToVMLAPatterns(context, conversionPatterns, typeConverter);
    populateHALToVMLAPatterns(context, conversionPatterns, typeConverter);

    // Ensure FuncOp signatures are updated.
    populateFuncOpTypeConversionPattern(conversionPatterns, context,
                                        typeConverter);

    // We allow the shape dialect to persist, making specific dim queries
    // illegal (which allows them to fold away). These patterns allow dimension
    // queries to convert properly, but they do not allow the introduction
    // of new shaped tensors.
    Shape::populateFoldConversionPatterns(&getContext(), conversionPatterns);
    conversionTarget.addLegalDialect<ShapeDialect>();
    // Since all inputs are converted to buffers, must trigger the TieShape
    // type conversion if the result type is illegal.
    conversionTarget.addDynamicallyLegalOp<Shape::TieShapeOp>(
        [](Shape::TieShapeOp op) {
          return op.result().getType().isa<BufferType>();
        });
    conversionTarget.addIllegalOp<Shape::RankedDimOp>();
    conversionTarget.addIllegalOp<Shape::RankedDimsOp>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      conversionPatterns, &typeConverter))) {
      getOperation().emitError() << "conversion to the VMLA dialect failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass() {
  return std::make_unique<ConversionPass>();
}

static PassRegistration<ConversionPass> pass(
    "iree-vmla-conversion",
    "Converts from various dialects to the VMLA dialect");

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
