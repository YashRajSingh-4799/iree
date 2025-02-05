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

#include "iree/compiler/Translation/CodegenPasses/Passes.h"

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

void addHLOToLinalgOnBuffersPasses(OpPassManager &pm) {
  pm.addPass(createHLOToLinalgOnTensorsPass());
  pm.addPass(createLinalgOnTensorsFusionPass());
  pm.addPass(createHLOToLinalgOnBuffersPass());
}

}  // namespace iree_compiler
}  // namespace mlir
