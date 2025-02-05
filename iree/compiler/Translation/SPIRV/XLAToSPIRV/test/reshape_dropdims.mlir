// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @reshape_4D_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  func @reshape_4D_3D(%arg0: memref<12x42x1xi32>, %arg1: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "reshape_4D_3D"} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42x1xi32>) : tensor<12x42x1xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42x1xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @reshape_4D_2D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  func @reshape_4D_2D(%arg0: memref<12x42x1x1xi32>, %arg1: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "reshape_4D_2D"} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42x1x1xi32>) : tensor<12x42x1x1xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42x1x1xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @reshape_2D_4D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  func @reshape_2D_4D(%arg0: memref<12x42xi32>, %arg1: memref<12x42x1x1xi32>)
  attributes {iree.dispatch_fn_name = "reshape_2D_4D"} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42xi32>) -> tensor<12x42x1x1xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42x1x1xi32>, %arg1 : memref<12x42x1x1xi32>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @reshape_2D_4D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  func @reshape_2D_4D(%arg0: memref<12x42xi32>, %arg1: memref<12x1x1x42xi32>)
  attributes {iree.dispatch_fn_name = "reshape_2D_4D"} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42xi32>) -> tensor<12x1x1x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x1x1x42xi32>, %arg1 : memref<12x1x1x42xi32>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @reshape_2D_4D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  func @reshape_2D_4D(%arg0: memref<12x1x1x42xi32>, %arg1: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "reshape_2D_4D"} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x1x1x42xi32>) : tensor<12x1x1x42xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x1x1x42xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    return
  }
}
