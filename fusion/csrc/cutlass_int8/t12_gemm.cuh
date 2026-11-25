// CUTLASS 2.x (Sm80 tensor-op) INT8 GEMM with epilogue-visitor-tree (EVT) epilogues.
//
//   A: INT8 [L*T, K] row-major (activations),  B: INT8 [N, K] (weights, = column-major K x N)
//   y = sa[row] * (acc * sb[col]) + bias[col]                         (EPI plain)
//   D = r[row, col] + gate[img, col] * y                              (EPI residual)
// The token axis is run as a batched GEMM with L = number of images and T tokens per image, so the
// per-image adaLN gate row is an ordinary row broadcast with a batch stride. Output FP16.
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace t12 {
using namespace cute;
namespace tbe = cutlass::epilogue::threadblock;
using Half = cutlass::half_t;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

template <int BM, int BN, int BK, int WM, int WN, int Stages, bool RES>
struct Kernel {
  using TB = cutlass::gemm::GemmShape<BM, BN, BK>;
  using WS = cutlass::gemm::GemmShape<WM, WN, BK>;
  using IS = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int kEpiStages = 1;
  using TM = tbe::OutputTileThreadLayout<TB, WS, Half, 8, kEpiStages>;

  using Acc = tbe::VisitorAccFetch;
  using ColF = tbe::VisitorColBroadcast<TM, float, Stride<_1, _0, int64_t>>;
  using RowF = tbe::VisitorRowBroadcast<TM, float, Stride<_0, _1, int64_t>>;
  using RowH = tbe::VisitorRowBroadcast<TM, Half, Stride<_0, _1, int64_t>>;
  using AuxH = tbe::VisitorAuxLoad<TM, Half, Stride<int64_t, _1, int64_t>>;
  using MulF = tbe::VisitorCompute<cutlass::multiplies, float, float, kRound>;
  using FmaF = tbe::VisitorCompute<cutlass::multiply_add, float, float, kRound>;
  using FmaH = tbe::VisitorCompute<cutlass::multiply_add, Half, float, kRound>;

  using EvtAccB = tbe::Sm80EVT<MulF, Acc, RowF>;                                    // acc * sb
  using EvtY = tbe::Sm80EVT<std::conditional_t<RES, FmaF, FmaH>, ColF, EvtAccB, RowF>; // sa * . + bias
  using EvtZ = std::conditional_t<RES, tbe::Sm80EVT<FmaH, RowH, EvtY, AuxH>, EvtY>; // gate * y + r
  using Store = tbe::VisitorAuxStore<TM, Half, kRound, Stride<int64_t, _1, int64_t>>;
  using EvtD = tbe::Sm80EVT<Store, EvtZ>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      Half, cutlass::layout::RowMajor, 8,
      int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      TB, WS, IS, EvtD, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      Stages, cutlass::arch::OpMultiplyAddSaturate, kEpiStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Returns 0 on success, 1 = cannot implement, 2 = initialize failed, 3 = launch failed.
template <class K>
int run(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa, const torch::Tensor& sb,
        const torch::Tensor& bias, bool has_bias, torch::Tensor& d, const torch::Tensor& r,
        const torch::Tensor& gate, int64_t g_off, int64_t g_stride, int64_t tokens) {
  using Gemm = typename K::Gemm;
  const int64_t M = a.size(0), Kd = a.size(1), N = b.size(0);
  const int64_t T = tokens, L = M / tokens;
  const float* sbp = sb.data_ptr<float>();
  const float* biasp = has_bias ? bias.data_ptr<float>() : nullptr;

  typename K::EvtY::Arguments y_args{
      {sa.data_ptr<float>(), 0.f, {_1{}, _0{}, T}},             // sa  [M] (batch stride T)
      {{}, {sbp, 0.f, {_0{}, _1{}, int64_t(0)}}, {}},           // acc * sb
      {biasp, 0.f, {_0{}, _1{}, int64_t(0)}},                   // bias
      {}};
  typename K::Store::Arguments store_args{reinterpret_cast<Half*>(d.data_ptr()), {N, _1{}, T * N}};
  typename K::EvtD::Arguments evt_args;
  if constexpr (std::is_same_v<typename K::EvtZ, typename K::EvtY>) {
    evt_args = typename K::EvtD::Arguments{y_args, store_args};
  } else {
    typename K::EvtZ::Arguments z_args{
        {reinterpret_cast<const Half*>(gate.data_ptr()) + g_off, Half(0), {_0{}, _1{}, g_stride}},
        y_args,
        {reinterpret_cast<Half*>(r.data_ptr()), Half(0), {N, _1{}, T * N}},
        {}};
    evt_args = typename K::EvtD::Arguments{z_args, store_args};
  }
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched,
                                {int(T), int(N), int(Kd)}, int(L), evt_args,
                                a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), nullptr, nullptr,
                                T * Kd, 0, 0, T * N, Kd, Kd, N, N);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  size_t ws = Gemm::get_workspace_size(args);
  torch::Tensor workspace;
  void* wsp = nullptr;
  if (ws) {
    workspace = torch::empty({int64_t(ws)}, a.options().dtype(torch::kUInt8));
    wsp = workspace.data_ptr();
  }
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  if (gemm.initialize(args, wsp, stream) != cutlass::Status::kSuccess) return 2;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return 3;
  return 0;
}

#define T12_ARGS const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,        \
                 const torch::Tensor& sb, const torch::Tensor& bias, bool has_bias, torch::Tensor& d, \
                 const torch::Tensor& r, const torch::Tensor& gate, int64_t g_off, int64_t g_stride,  \
                 int64_t tokens
#define T12_CALL a, b, sa, sb, bias, has_bias, d, r, gate, g_off, g_stride, tokens
}  // namespace t12
