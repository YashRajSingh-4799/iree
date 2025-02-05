// TODO(hanchung): Remove the test once vulkan Linalg path support reshape op
// for general cases.
func @reshape_2D_3D() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]> : tensor<2x6xi32>
  %result = "xla_hlo.reshape"(%input) : (tensor<2x6xi32>) -> tensor<2x1x6xi32>
  check.expect_eq_const(%result, dense<[[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12]]]> : tensor<2x1x6xi32>) : tensor<2x1x6xi32>
  return
}
