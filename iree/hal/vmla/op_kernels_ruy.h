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

#ifndef IREE_HAL_VMLA_OP_KERNELS_RUY_H_
#define IREE_HAL_VMLA_OP_KERNELS_RUY_H_

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "ruy/context.h"
#include "ruy/ruy.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

// TODO(benvanik): something more clever for making this shareable.
// Maybe a factory fn based on the impl selected?
struct MatMul::RuntimeState {
  // TODO(benvanik): share the thread pool but keep context per-fiber?
  ruy::Context context;
};

inline std::unique_ptr<MatMul::RuntimeState> MatMul::CreateRuntimeState() {
  return absl::make_unique<RuntimeState>();
}

template <typename T, typename ACC>
Status MatMul::Execute(RuntimeState* runtime_state,
                       const Buffers<T, ACC>& buffers) {
  ruy::Matrix<T> lhs;
  lhs.set_data(buffers.lhs_buffer.data());
  ruy::MakeSimpleLayout(buffers.lhs_shape[0], buffers.lhs_shape[1],
                        ruy::Order::kRowMajor, lhs.mutable_layout());

  ruy::Matrix<T> rhs;
  rhs.set_data(buffers.rhs_buffer.data());
  ruy::MakeSimpleLayout(buffers.rhs_shape[1], buffers.rhs_shape[0],
                        ruy::Order::kColMajor, rhs.mutable_layout());

  ruy::Matrix<T> dst;
  dst.set_data(buffers.dst_buffer.data());
  ruy::MakeSimpleLayout(buffers.dst_shape[1], buffers.dst_shape[0],
                        ruy::Order::kColMajor, dst.mutable_layout());

  ruy::MulParams<ACC, T> mul_params;
  mul_params.set_bias(buffers.bias_buffer.data());

  if (buffers.multiplier_mantissa_buffer.size() == 1) {
    mul_params.set_multiplier_fixedpoint(buffers.multiplier_mantissa_buffer[0]);
    mul_params.set_multiplier_exponent(buffers.multiplier_exponent_buffer[0]);
  } else {
    mul_params.set_multiplier_fixedpoint_perchannel(
        buffers.multiplier_mantissa_buffer.data());
    mul_params.set_multiplier_exponent_perchannel(
        buffers.multiplier_exponent_buffer.data());
  }

  ruy::Mul(lhs, rhs, mul_params, &runtime_state->context, &dst);

  return OkStatus();
}

}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VMLA_OP_KERNELS_RUY_H_
