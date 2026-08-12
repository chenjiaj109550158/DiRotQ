#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

// This is the single, pinned public E2M1 x E2M1 baseline kernel family used by
// the unsupported operand-format-bit research prototype.  Its configuration is
// derived from CUTLASS v4.0.0 example 79a; output is FP32 so correctness can be
// checked before a separately measured runtime cast.
namespace blackwell_optimized_fp4 {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;  // 32 FP4 elements == 16 bytes.
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 1;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

static_assert(cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 16);
static_assert(cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 8);
static_assert(cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 64);
static_assert(ScaleConfig::SFVecSize == 16);

}  // namespace blackwell_optimized_fp4
