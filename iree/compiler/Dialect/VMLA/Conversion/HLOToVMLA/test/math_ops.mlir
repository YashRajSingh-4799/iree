// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @abs_scalar
func @abs_scalar(%arg0 : tensor<f32>) -> tensor<f32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 4
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: vmla.abs(%arg0, [[BUF]]) : f32
  %0 = "xla_hlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @abs_tensor
func @abs_tensor(%arg0 : tensor<4xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 16
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: vmla.abs(%arg0, [[BUF]]) : f32
  %0 = "xla_hlo.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp
func @clamp(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 16
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: vmla.clamp(%arg0, %arg1, %arg2, [[BUF]]) : f32
  %0 = "xla_hlo.clamp"(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<4xf32>
}
