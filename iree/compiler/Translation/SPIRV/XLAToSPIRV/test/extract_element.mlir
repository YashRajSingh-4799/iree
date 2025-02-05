// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK-LABEL: spv.func @extract_element
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
  func @extract_element(%arg0: memref<i1>, %arg1: memref<i1>)
    attributes {iree.dispatch_fn_name = "extract_element"} {
    %0 = "iree.load_input"(%arg0) : (memref<i1>) -> tensor<i1>
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0]]{{\[}}[[ZERO1]]{{\]}}
    %1 = "std.extract_element"(%0) : (tensor<i1>) -> i1
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1]]{{\[}}[[ZERO2]]{{\]}}
    "iree.store_output"(%1, %arg1) : (i1, memref<i1>) -> ()
    "std.return"() : () -> ()
  }
}
