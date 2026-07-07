#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#define __CLION_IDE__

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE — full implementation
// ----------------------------------------------------------------------------
// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (1 or 2, totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * L2 normally finishes with the COMBINE step (top-k reduction in BF16).
//     The split-graph path can skip that reduction in K2 and run it as K3.
// ============================================================================

enum class MegaMoEPhaseKind {
    Linear1,
    Linear2
};

template <MegaMoEPhaseKind kKind>
struct MegaMoEPhasePolicy {
    static constexpr bool is_linear1_only = kKind == MegaMoEPhaseKind::Linear1;
    static constexpr bool is_linear2_only = kKind == MegaMoEPhaseKind::Linear2;
    static constexpr bool runs_linear1 = is_linear1_only;
    static constexpr bool runs_linear2 = is_linear2_only;
    static constexpr bool needs_dispatch_pull = runs_linear1;
    static constexpr bool needs_combine_barrier = runs_linear2;
    static constexpr bool needs_combine_reduce = runs_linear2;

    template <typename Scheduler, typename Func>
    CUTLASS_DEVICE static void for_each_selected_block(Scheduler& scheduler, Func&& func) {
        if constexpr (is_linear1_only) {
            scheduler.template for_each_phase_block<sched::BlockPhase::Linear1>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(sched::BlockPhase::Linear1, local_expert_idx,
                         num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
            scheduler.template for_each_phase_block<sched::BlockPhase::Linear2>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(sched::BlockPhase::Linear2, local_expert_idx,
                         num_k_blocks, m_block_idx, n_block_idx);
                });
        }
    }
};

using MegaMoELinear1Phase = MegaMoEPhasePolicy<MegaMoEPhaseKind::Linear1>;
using MegaMoELinear2Phase = MegaMoEPhasePolicy<MegaMoEPhaseKind::Linear2>;

struct MegaMoELinear2NoReducePhase : MegaMoEPhasePolicy<MegaMoEPhaseKind::Linear2> {
    static constexpr bool needs_combine_reduce = false;
};

#define DG_SM90_FP8_MOE_TEMPLATE_PARAMS \
    uint32_t kNumMaxTokensPerRank, \
    uint32_t kHidden, uint32_t kIntermediateHidden, \
    uint32_t kNumExperts, uint32_t kNumTopk, \
    uint32_t kNumExpertsPerWave, \
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, \
    uint32_t kNumMaxPoolTokens, \
    uint32_t kNumPaddedSFPoolTokens, \
    uint32_t kSFPoolStrideTokens, \
    uint32_t kNumStages, \
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads, \
    uint32_t kNumEpilogueThreads, \
    uint32_t kClusterSize, \
    uint32_t kNumSMs, uint32_t kNumRanks, \
    float kActivationClamp, \
    bool kFastMath, \
    bool kDirectL2ScatterRequested = false, \
    bool kPhaseProfileRequested = false, \
    bool kL2NMajorScheduleRequested = false, \
    bool kOneWarpCleanupRequested = false, \
    uint32_t kDispatchNumSMs = kNumSMs, \
    bool kSplitSFATMARequested = false, \
    bool kL2DualAccumRequested = false, \
    bool kL1DualKAccumRequested = false, \
    bool kL2SFA2RowTMARequested = false, \
    bool kL2PrecomputeScaleRequested = false, \
    bool kL2RatioAccumRequested = false, \
    bool kL2WeightSFLDURequested = false, \
    bool kL2PrefetchWeightSFRequested = false, \
    bool kL2FastWeightSFNRequested = false, \
    bool kL2HoistWeightSFBaseRequested = false, \
    bool kL2DelaySFALoadRequested = false, \
    bool kL2SplitSFALoadRequested = false, \
    bool kL2SkipPostWGMMAFenceRequested = false, \
    bool kL2FirstAccumAssignRequested = false, \
    bool kL2N64InterleaveRequested = false, \
    bool kL2HalfKPipelineRequested = false, \
    bool kAsyncL1TMAStoreRequested = false, \
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2, \
    uint32_t L1_SHAPE_K = kHidden, \
    uint32_t L2_SHAPE_N = kHidden, \
    uint32_t L2_SHAPE_K = kIntermediateHidden, \
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32, \
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32, \
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32, \
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4, \
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, \
    uint32_t kNumTokensPerWarp = 32 / kNumTopk, \
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks

#define DG_SM90_FP8_MOE_KERNEL_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const layout::SymBuffer<kNumRanks>& sym_buffer, \
    const cute::TmaDescriptor& tensor_map_l1_acts, \
    const cute::TmaDescriptor& tensor_map_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const cute::TmaDescriptor& tensor_map_l1_output, \
    const cute::TmaDescriptor& tensor_map_l2_acts, \
    const cute::TmaDescriptor& tensor_map_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf

#define DG_SM90_FP8_MOE_KERNEL_ARGS \
    y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, \
    tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, \
    l1_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, \
    tensor_map_l2_acts_sf, tensor_map_l2_weights, l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS(PhasePolicy) \
    PhasePolicy, \
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, \
    kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, kNumMaxPoolTokens, \
    kNumPaddedSFPoolTokens, kSFPoolStrideTokens, kNumStages, kNumDispatchThreads, \
    kNumNonEpilogueThreads, kNumEpilogueThreads, kClusterSize, kNumSMs, \
    kNumRanks, kActivationClamp, kFastMath, kDirectL2ScatterRequested, \
    kPhaseProfileRequested, kL2NMajorScheduleRequested, kOneWarpCleanupRequested, \
    kDispatchNumSMs, kSplitSFATMARequested, kL2DualAccumRequested, \
    kL1DualKAccumRequested, kL2SFA2RowTMARequested, kL2PrecomputeScaleRequested, kL2RatioAccumRequested, kL2WeightSFLDURequested, \
    kL2PrefetchWeightSFRequested, kL2FastWeightSFNRequested, kL2HoistWeightSFBaseRequested, kL2DelaySFALoadRequested, kL2SplitSFALoadRequested, kL2SkipPostWGMMAFenceRequested, kL2FirstAccumAssignRequested, kL2N64InterleaveRequested, kL2HalfKPipelineRequested, kAsyncL1TMAStoreRequested, L1_SHAPE_N, L1_SHAPE_K, L2_SHAPE_N, \
    L2_SHAPE_K, kNumDispatchWarps, \
    kNumMMANonEpilogueWarps, kNumEpilogueWarps, kNumEpilogueWarpgroups, \
    kNumThreads, kNumTokensPerWarp, kNumExpertsPerRank

template <typename MegaMoEPhase, DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_DEVICE void
sm90_fp8_mega_moe_core(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
#if defined(DG_SM90_MOE_SCALE_FREE_L1_K2)
    constexpr bool kScaleFreeL1K2 = true;
#else
    constexpr bool kScaleFreeL1K2 = false;
#endif
#if defined(DG_SM90_MOE_SCALE_FREE_DROP_SFA)
    constexpr bool kDropScaleFreeSFA = true;
#else
    constexpr bool kDropScaleFreeSFA = false;
#endif
#if defined(DG_SM90_MOE_FP8_COMPRESSED_COMBINE)
    constexpr bool kFP8CompressedCombineRequested = true;
#else
    constexpr bool kFP8CompressedCombineRequested = false;
#endif
    DG_STATIC_ASSERT(!kDropScaleFreeSFA || kScaleFreeL1K2,
                     "Dropping L1-to-K2 SFA requires the scale-free path");

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 0 or kNumDispatchThreads == 64 or kNumDispatchThreads % 128 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps (2 or 4 warps expected)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(kNumSMs % kClusterSize == 0, "SM count must be divisible by cluster size");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0,
                     "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128 or BLOCK_N == 256, "BLOCK_N must be 64/128/256 for this SM90 path");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Prefetch the TMA descriptors used by this split phase.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        if constexpr (MegaMoEPhase::runs_linear1) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        }
        if constexpr (MegaMoEPhase::runs_linear2) {
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
            if constexpr (!kDropScaleFreeSFA)
                cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        }
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Per-64 K float SF (SM90 only): 4 bytes per per-64 group => `kIntermediateHidden / 16` bytes/token
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
    DG_STATIC_ASSERT(kSFPoolStrideTokens <= kNumPaddedSFPoolTokens,
                     "Logical SF pool stride must fit in the allocated SF pool capacity");

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());
    constexpr uint32_t kL2NGroupSize = 4;
    constexpr uint32_t kL2NumNGroups = (L2_SHAPE_N / BLOCK_N) / kL2NGroupSize;
    // Keep the retained group-of-four schedule while rebalancing the exact
    // scale-free split; its persistent block stride already uses kNumSMs.
    constexpr bool kL2GroupedNSchedule =
        MegaMoEPhase::is_linear2_only && kNumRanks == 8 &&
        kHidden == 3072 && kIntermediateHidden == 1536 &&
        kNumExperts == 128 && kNumTopk == 16 &&
        (kNumSMs == 62 || kNumSMs == 54 || kScaleFreeL1K2) &&
        BLOCK_M == 192 && BLOCK_N == 128 && BLOCK_K == 128;
    constexpr bool kFP8CompressedCombine =
        kFP8CompressedCombineRequested && MegaMoEPhase::is_linear2_only &&
        !MegaMoEPhase::needs_combine_reduce && kNumRanks == 8 &&
        kNumMaxTokensPerRank >= 7000 && kHidden == 3072 &&
        kIntermediateHidden == 1536 && kNumExperts == 128 && kNumTopk == 16 &&
        BLOCK_M == 192 && BLOCK_N == 128 && BLOCK_K == 128 &&
        kNumEpilogueWarpgroups == 3;
    DG_STATIC_ASSERT(!kL2GroupedNSchedule ||
                     L2_SHAPE_N / BLOCK_N == kL2NGroupSize * kL2NumNGroups,
                     "Grouped K2 schedule must cover every output-N tile");

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    DG_STATIC_ASSERT(!kScaleFreeL1K2 ||
                     (kNumMaxTokensPerRank >= 7000 && kHidden == 3072 &&
                      kIntermediateHidden == 1536 && kNumExperts == 128 &&
                      kNumTopk == 16 && kNumRanks == 8 && BLOCK_M == 192 &&
                      BLOCK_N == 128 && BLOCK_K == 128),
                     "Scale-free L1-to-K2 must remain scoped to the approved exact shape");
    using a_dtype_t = std::conditional_t<
        kScaleFreeL1K2 && MegaMoEPhase::runs_linear2,
        cutlass::float_e5m2_t, cutlass::float_e4m3_t>;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kSplitNWarpgroups =
        BLOCK_N == 256 && (kNumEpilogueWarpgroups == 2 || kNumEpilogueWarpgroups == 4);
    constexpr bool kSerialNWarpgroups = false;
    constexpr bool kWideNWarpgroups =
        BLOCK_N == 256 && kNumEpilogueWarpgroups == 1;
    constexpr uint32_t WG_BLOCK_M = kSplitNWarpgroups ? BLOCK_M : BLOCK_M / kNumEpilogueWarpgroups;
    constexpr uint32_t WG_BLOCK_N = kSplitNWarpgroups ? BLOCK_N / kNumEpilogueWarpgroups :
                                    (kSerialNWarpgroups ? BLOCK_N / 2 : BLOCK_N);
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr bool kAsyncL1TMAStore = kAsyncL1TMAStoreRequested &&
        MegaMoEPhase::runs_linear1 && kSplitNWarpgroups;
    DG_STATIC_ASSERT(!kSplitSFATMARequested || kNumMMANonEpilogueWarps >= 3,
                     "Split SFA TMA requires a third non-epilogue TMA warp");
    constexpr bool kSplitSFATMA = kSplitSFATMARequested;
    constexpr bool kDropL2SFA = kDropScaleFreeSFA && MegaMoEPhase::runs_linear2;
    constexpr bool kDirectL2Scatter = kDirectL2ScatterRequested && MegaMoEPhase::runs_linear2 &&
        (!kSerialNWarpgroups) && (WG_BLOCK_N == 64 || WG_BLOCK_N == 128);
    constexpr bool kL2DualAccum = kL2DualAccumRequested && MegaMoEPhase::runs_linear2;
    constexpr bool kL2RatioAccum = kL2RatioAccumRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2DualAccum;
    constexpr bool kL2PrecomputeScale = kL2PrecomputeScaleRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2DualAccum && !kL2RatioAccum;
    constexpr bool kL2WeightSFLDU = kL2WeightSFLDURequested && MegaMoEPhase::runs_linear2;
#if defined(DG_SM90_MOE_L2_WEIGHT_SF_NC)
    constexpr bool kL2WeightSFNC = MegaMoEPhase::runs_linear2;
#else
    constexpr bool kL2WeightSFNC = false;
#endif
    constexpr bool kL2PrefetchWeightSF = kL2PrefetchWeightSFRequested && MegaMoEPhase::runs_linear2 &&
        !kL2DualAccum;
    constexpr bool kL2FastWeightSFN = kL2FastWeightSFNRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && BLOCK_N == 256 && kSplitNWarpgroups;
    constexpr bool kL2HoistWeightSFBase = kL2HoistWeightSFBaseRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2FastWeightSFN;
    constexpr bool kL2DelaySFALoad = kL2DelaySFALoadRequested && MegaMoEPhase::runs_linear2 &&
        !kL2DualAccum && !kL2PrefetchWeightSF;
    constexpr bool kL2SplitSFALoad = kL2SplitSFALoadRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2DualAccum && !kL2RatioAccum && !kL2DelaySFALoad;
    constexpr bool kL2SkipPostWGMMAFence =
        kL2SkipPostWGMMAFenceRequested && MegaMoEPhase::runs_linear2 && !kL2DualAccum;
    constexpr bool kL2FirstAccumAssign =
        kL2FirstAccumAssignRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2DualAccum && !kL2RatioAccum;
    constexpr bool kL2N64Interleave =
        kL2N64InterleaveRequested && MegaMoEPhase::runs_linear2 &&
        WG_BLOCK_N == 128 && !kL2DualAccum && !kL2RatioAccum &&
        !kL2FirstAccumAssign && !kL2DelaySFALoad && !kL2SplitSFALoad &&
        !kL2PrefetchWeightSF;
    constexpr bool kL2HalfKPipeline =
        kL2HalfKPipelineRequested && MegaMoEPhase::is_linear2_only &&
        !MegaMoEPhase::needs_combine_reduce && WG_BLOCK_N == 128 && BLOCK_K == 128 &&
        !kL2DualAccum && !kL2RatioAccum && !kL2N64Interleave &&
        !kL2PrecomputeScale && !kL2PrefetchWeightSF && !kL2FastWeightSFN &&
        !kL2HoistWeightSFBase && !kL2DelaySFALoad && !kL2SplitSFALoad &&
        !kL2SkipPostWGMMAFence && !kL2FirstAccumAssign;
    constexpr bool kL1DualKAccum = kL1DualKAccumRequested && MegaMoEPhase::runs_linear1 &&
        ((kHidden / BLOCK_K) % 2 == 0);
    constexpr bool kL2SFA2RowTMA = kL2SFA2RowTMARequested && MegaMoEPhase::runs_linear2 &&
        (L2_SHAPE_K % BLOCK_K == 0);
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
#if defined(DG_SM90_MOE_SCALE_FREE_L1_K2)
    using L2WGMMA   = typename mma::sm90::FP8E5M2E4M3MMASelector<WG_BLOCK_N>::type;
#else
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
#endif
    using L2N64WGMMA = typename mma::sm90::FP8MMASelector<64>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or (BLOCK_M == 64 and (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64 or M64N128 WGMMA consumers");

    // A is always CTA-local.  When kClusterSize=2 the scheduler pairs adjacent
    // M blocks with identical expert/N/K coordinates so the B TMA can multicast.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Combine reuses the pre-barrier SMEM region, so split L2 keeps this
    // dispatch scratch capacity even though it does not run dispatch pull.
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE = kNumDispatchWarps > 0 ?
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment) : 0u;
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE = kNumDispatchWarps > 0 ?
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment) : 0u;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and L2
    // (two per-64-K halves). Each TMA destination must be 128B aligned.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = 2 * kL2SFAHalfStride * sizeof(float);
    // Block (128, 128) weight SF: 1 float per (BLOCK_N, BLOCK_K) tile for L2,
    // 2 floats (gate/up) for L1. Loaded by math warpgroup directly from global,
    // so no SMEM is needed.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = 0;

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_ACCUM_SIZE = 0u;
    constexpr uint32_t SMEM_CD_L1_SIZE = MegaMoEPhase::runs_linear1 ?
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_L2_SIZE = (!MegaMoEPhase::runs_linear2 || kDirectL2Scatter) ? 0u :
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_L1_ASYNC_ELEMS =
        kNumEpilogueWarpgroups * WG_BLOCK_M * L1_OUT_BLOCK_N;
    constexpr uint32_t SMEM_CD_L1_ASYNC_SIZE = kAsyncL1TMAStore ?
        2 * SMEM_CD_L1_ASYNC_ELEMS * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE > SMEM_CD_L1_ASYNC_SIZE ? SMEM_CD_OUTPUT_BASE_SIZE : SMEM_CD_L1_ASYNC_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_ACCUM_SIZE + SMEM_CD_OUTPUT_SIZE;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    constexpr uint32_t kCombineHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kCombineChunkSlots = 3;
    constexpr uint32_t kCombineMaxRegistersForBuffer = 128;
    constexpr uint32_t kCombineNumChunks =
        (kCombineChunkSlots * kNumEpilogueWarps * kCombineHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
         and kHidden <= 32 * kCombineMaxRegistersForBuffer) ? 1 : 2;
    constexpr uint32_t kCombineChunkBytes = kCombineHiddenBytes / kCombineNumChunks;
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE = MegaMoEPhase::needs_combine_reduce
        ? kCombineChunkSlots * kNumEpilogueWarps * kCombineChunkBytes : 0u;
    DG_STATIC_ASSERT(kHidden % kCombineNumChunks == 0, "Hidden must be divisible by number of combine chunks");
    DG_STATIC_ASSERT(SMEM_COMBINE_ALIAS_SIZE <= SMEM_BEFORE_BARRIER_SIZE,
                     "Combine SMEM alias exceeds the pre-barrier scratch region");

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    auto smem_cd_base = math::advance_ptr<uint8_t>(smem_gemm_base, SMEM_CD_ACCUM_SIZE);
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    // Barriers live after SF (SFB is loaded directly from global, no SMEM)
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });

    // =====================================================================
    // Initialization
    // =====================================================================
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        if constexpr (kNumDispatchWarps > 0) {
            #pragma unroll
            for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
                ptx::st_shared(smem_expert_count + i, 0u);
        }
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        if constexpr (kNumDispatchWarps > 0) {
            #pragma unroll
            for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
                dispatch_barriers[i]->init(1);
            cutlass::arch::fence_barrier_init();
        }
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B, or A + B + SFA when
                // split-SFA uses an otherwise idle TMA warp.
                full_barriers[i]->init(kSplitSFATMA && !kDropL2SFA ? 3 : 2);
                // With cluster multicast the leader CTA's TMA warp waits on peer
                // empty barriers too, so every math warp releases both CTAs.
                empty_barriers[i]->init(kClusterSize * kNumEpilogueWarps);
            }
            if constexpr (MegaMoEPhase::needs_combine_reduce) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                    combine_barriers[i]->init(1);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kClusterSize > 1) {
        cute::cluster_sync();
    } else {
        __syncthreads();
    }

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, kClusterSize, kL2NMajorScheduleRequested, false,
        kDispatchNumSMs>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread case (block_m=128, 2 math WGs):
    //   128*48 + 128*40 + 256*208 = 64512 exactly.
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr bool kCompactFrontendWarpgroup =
        kNumDispatchWarps + kNumMMANonEpilogueWarps == 4;
    constexpr uint32_t kNumNonEpilogueRegisters = kCompactFrontendWarpgroup ? kNumDispatchRegisters : 40;
#ifndef DG_SM90_MOE_EPILOGUE_REGS
#define DG_SM90_MOE_EPILOGUE_REGS 0
#endif
    constexpr uint32_t kDefaultEpilogueRegisters =
        kNumEpilogueThreads == 384 ? (kCompactFrontendWarpgroup ? 152 : 136) :
        ((kSplitNWarpgroups && WG_BLOCK_N == 64) ? 112 :
         ((kSerialNWarpgroups or kWideNWarpgroups) ? 256 : 208));
    constexpr uint32_t kRequestedEpilogueRegisters = DG_SM90_MOE_EPILOGUE_REGS;
    constexpr uint32_t kNumEpilogueRegisters =
        kRequestedEpilogueRegisters > 0 ? kRequestedEpilogueRegisters : kDefaultEpilogueRegisters;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");
    DG_STATIC_ASSERT(kNumTopk <= 64 and kNumExpertsPerRank <= 1024 and kNumMaxTokensPerRank <= 65536,
                     "Route entry bit-packing constraints");
    constexpr bool kEnableSingleRankRouteDispatch = kNumRanks == 1 and kNumSMs == 76;
    constexpr bool kNeedsCombineBarrier = MegaMoEPhase::needs_combine_barrier;

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    constexpr uint32_t kProfileDispatchTotal = 0;
    constexpr uint32_t kProfileDispatchPull = 1;
    constexpr uint32_t kProfileMathLoop = 2;
    constexpr uint32_t kProfileCombineBarrier = 3;
    constexpr uint32_t kProfileCombineReduce = 4;
    constexpr uint32_t kProfileGemmCore = 5;
    constexpr uint32_t kProfileL1Epilogue = 6;
    constexpr uint32_t kProfileL2Epilogue = 7;
    constexpr uint32_t kProfileL1ArrivalWait = 8;
    constexpr uint32_t kProfileL2ArrivalWait = 9;
    constexpr uint32_t kProfileWorkspaceCleanup = 10;
    constexpr uint32_t kProfileFullBarrierWait = 11;
    constexpr uint32_t kProfileL1WgmmaGroup = 12;
    constexpr uint32_t kProfileL1AccumScale = 13;
    constexpr uint32_t kProfileL2WgmmaLo = 14;
    constexpr uint32_t kProfileL2ScaleLo = 15;
    constexpr uint32_t kProfileL2WgmmaHi = 16;
    constexpr uint32_t kProfileL2ScaleHi = 17;
    constexpr uint32_t kProfileL1TMAStoreWait = 18;
    constexpr uint32_t kProfileL1ReadyMask = 19;
    constexpr uint32_t kProfileL1MathLoop = 20;
    constexpr uint32_t kProfileL2MathLoop = 21;
    constexpr uint32_t kProfileL2ScaleFactorLoad = 22;
    constexpr uint32_t kProfileL2ActivationScaleLoad = 23;
    constexpr uint32_t kProfileL2WeightScaleLoad = 24;
    constexpr uint32_t kProfileL2WgmmaLoIssue = 25;
    constexpr uint32_t kProfileL2WgmmaLoWait = 26;
    constexpr uint32_t kProfileL2WgmmaHiIssue = 27;
    constexpr uint32_t kProfileL2WgmmaHiWait = 28;
    constexpr uint32_t kProfileL2EmptyBarrierArrive = 29;
    constexpr uint32_t kProfileL2IterCore = 30;
    constexpr uint32_t kNumProfileMetrics = 31;
    constexpr uint32_t kPhaseProfileIntSlots = 192;
    constexpr uint32_t kPhaseTraceIntSlots = 64;
    constexpr uint32_t kTraceK2DispatchEnter = 0;
    constexpr uint32_t kTraceK2DispatchAfterFetch = 1;
    constexpr uint32_t kTraceK2DispatchAfterSync1 = 2;
    constexpr uint32_t kTraceK2DispatchAfterSync2 = 3;
    constexpr uint32_t kTraceK2DispatchReturnNoReduce = 4;
    constexpr uint32_t kTraceK2TmaABeforeEmptyWait = 5;
    constexpr uint32_t kTraceK2TmaAAfterEmptyWait = 6;
    constexpr uint32_t kTraceK2TmaABeforeArrivalWait = 7;
    constexpr uint32_t kTraceK2TmaAAfterArrivalWait = 8;
    constexpr uint32_t kTraceK2TmaAAfterFullArrive = 9;
    constexpr uint32_t kTraceK2TmaBBeforeEmptyWait = 10;
    constexpr uint32_t kTraceK2TmaBAfterEmptyWait = 11;
    constexpr uint32_t kTraceK2TmaBAfterFullArrive = 12;
    constexpr uint32_t kTraceK2MathAfterDispatchSync = 13;
    constexpr uint32_t kTraceK2MathBlockStart = 14;
    constexpr uint32_t kTraceK2MathBeforeFullWait = 15;
    constexpr uint32_t kTraceK2MathAfterFullWait = 16;
    constexpr uint32_t kTraceK2MathAfterGemmLoop = 17;
    constexpr uint32_t kTraceK2MathBeforeScatterSync = 18;
    constexpr uint32_t kTraceK2MathAfterScatterSync = 19;
    constexpr uint32_t kTraceK2MathAfterBlock = 20;
    constexpr uint32_t kTraceK2MathLoopEnd = 21;
    constexpr uint32_t kTraceK2TmaAEmptyTimeout = 22;
    constexpr uint32_t kTraceK2TmaAArrivalTimeout = 23;
    constexpr uint32_t kTraceK2TmaBEmptyTimeout = 24;
    constexpr uint32_t kTraceK2MathFullTimeout = 25;
    constexpr uint32_t kTraceK2FetchBegin = 26;
    constexpr uint32_t kTraceK2FetchDone = 27;
    constexpr uint32_t kTraceK2FetchTimeout = 28;
    constexpr uint32_t kTraceK2FetchOnlyReturn = 29;
    constexpr uint32_t kTraceK2KernelEntry = 30;
    constexpr uint32_t kTraceK2DispatchRoleEntry = 31;
    constexpr uint32_t kTraceK2TmaARoleEntry = 32;
    constexpr uint32_t kTraceK2TmaBRoleEntry = 33;
    constexpr uint32_t kTraceK2IdleRoleEntry = 34;
    constexpr uint32_t kTraceK2MathRoleEntry = 35;
    constexpr uint32_t kTraceK2EntryOnlyReturn = 36;
    constexpr uint32_t kTraceK2MathBeforeRemoteScatter = 37;
    constexpr uint32_t kTraceK2MathAfterRemoteScatter = 38;
    constexpr uint32_t kTraceK2MathBeforeFinalEpiSync = 39;
    constexpr uint32_t kTraceK2MathAfterFinalEpiSync = 40;
    constexpr uint32_t kTraceK2MathBeforeEmptyArrive = 41;
    constexpr uint32_t kTraceK2MathAfterEmptyArrive = 42;
    constexpr uint32_t kTraceK2MathAfterDefaultGemmLoop = 43;
    constexpr uint32_t kTraceK2MathWGAfterFullWaitBase = 44;
    constexpr uint32_t kTraceK2MathWGAfterDefaultGemmLoopBase = 48;
    constexpr uint32_t kTraceK2MathWGAfterFinalEpiSyncBase = 52;
    constexpr uint32_t kTraceK2MathWGAfterBlockBase = 56;
    constexpr uint32_t kTraceK2MathWGLoopEndBase = 60;
    const auto phase_profile_clock = [&]() -> unsigned long long {
        if constexpr (kPhaseProfileRequested) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            return t;
        } else {
            return 0ull;
        }
    };
    const auto phase_profile_record = [&](const uint32_t& metric, const unsigned long long& cycles) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and cycles > 0) {
                constexpr uint32_t kProfileIntOffset = (kNumExpertsPerRank + 1u) & ~1u;
                auto profile = reinterpret_cast<unsigned long long*>(
                    cumulative_local_expert_recv_stats + kProfileIntOffset);
                atomicAdd(profile + metric, cycles);
                atomicMax(profile + kNumProfileMetrics + metric, cycles);
                atomicAdd(profile + 2 * kNumProfileMetrics + metric, 1ull);
            }
        }
    };
    const auto phase_trace_mark = [&](const uint32_t& slot) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and lane_idx == 0 and slot < kPhaseTraceIntSlots) {
                constexpr uint32_t kProfileIntOffset = (kNumExpertsPerRank + 1u) & ~1u;
                ptx::red_add(cumulative_local_expert_recv_stats + kProfileIntOffset + kPhaseProfileIntSlots + slot, 1);
            }
        }
    };
    const auto phase_trace_mark_any = [&](const uint32_t& slot) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and slot < kPhaseTraceIntSlots) {
                constexpr uint32_t kProfileIntOffset = (kNumExpertsPerRank + 1u) & ~1u;
                ptx::red_add(cumulative_local_expert_recv_stats + kProfileIntOffset + kPhaseProfileIntSlots + slot, 1);
            }
        }
    };
    constexpr unsigned long long kPhaseTraceWaitTimeoutNs = 2000000000ull;
    const auto phase_trace_wait_barrier = [&](auto* barrier, const uint32_t& wait_phase,
                                              const uint32_t& timeout_slot) -> bool {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            const unsigned long long wait_start = phase_profile_clock();
            while (!barrier->try_wait(wait_phase)) {
                if (phase_profile_clock() - wait_start > kPhaseTraceWaitTimeoutNs) {
                    phase_trace_mark(timeout_slot);
                    return false;
                }
            }
        } else {
            barrier->wait(wait_phase);
        }
        return true;
    };
    const auto phase_trace_wait_l2_arrival = [&](auto ptr, const uint64_t& expected_mask,
                                                 const uint32_t& timeout_slot) -> bool {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            const unsigned long long wait_start = phase_profile_clock();
            while ((ptx::ld_acq_gpu(ptr) & expected_mask) != expected_mask) {
                if (phase_profile_clock() - wait_start > kPhaseTraceWaitTimeoutNs) {
                    phase_trace_mark(timeout_slot);
                    return false;
                }
            }
        } else {
            while ((ptx::ld_acq_gpu(ptr) & expected_mask) != expected_mask);
        }
        return true;
    };

    const auto phase_trace_fetch_expert_recv_count = [&]() -> bool {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2FetchBegin);
            constexpr uint32_t kTraceNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
            #pragma unroll
            for (uint32_t i = 0; i < kTraceNumExpertsPerLane; ++ i) {
                const auto expert_idx = i * 32 + lane_idx;
                uint64_t value = 0;
                const unsigned long long wait_start = phase_profile_clock();
                while (true) {
                    bool ready = true;
                    bool timed_out = false;
                    if (expert_idx < kNumExpertsPerRank) {
                        const auto ptr = workspace.get_expert_recv_count_sum_ptr(expert_idx);
                        value = ptx::ld_volatile(ptr);
                        ready = static_cast<uint32_t>(value >> 32) == kDispatchNumSMs * kNumRanks;
                        if (!ready and phase_profile_clock() - wait_start > kPhaseTraceWaitTimeoutNs) {
                            timed_out = true;
                            ready = true;
                        }
                    }
                    if (__any_sync(0xffffffffu, timed_out)) {
                        if (timed_out)
                            phase_trace_mark_any(kTraceK2FetchTimeout);
                        scheduler.stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
                        return false;
                    }
                    if (__all_sync(0xffffffffu, ready))
                        break;
                }
                scheduler.stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
            }
            __syncwarp();
            phase_trace_mark(kTraceK2FetchDone);
            return true;
        } else {
            scheduler.fetch_expert_recv_count();
            return true;
        }
    };

    const auto for_each_selected_block = [&](auto&& func) {
        if constexpr (kL2GroupedNSchedule) {
            // Keep four adjacent N tiles on the same CTA to reuse the same
            // activation M tile through the cache hierarchy.
            DG_STATIC_ASSERT(kClusterSize == 1,
                             "Grouped K2 schedule requires single-CTA clusters");
            scheduler.fetch_expert_recv_count();
            scheduler.set_expert_idx(0);
            while (scheduler.current_local_expert_idx < kNumExpertsPerRank) {
                const uint32_t num_m_units = scheduler.get_current_num_m_blocks();
                const uint32_t num_grouped_blocks = num_m_units * kL2NumNGroups;
                if (scheduler.block_idx < num_grouped_blocks) {
                    const uint32_t m_block_idx = scheduler.block_idx / kL2NumNGroups;
                    const uint32_t n_group_idx = scheduler.block_idx -
                        m_block_idx * kL2NumNGroups;
                    scheduler.block_idx += kNumSMs;
                    #pragma unroll
                    for (uint32_t n_in_group = 0; n_in_group < kL2NGroupSize;
                         ++ n_in_group) {
                        scheduler.m_block_idx = m_block_idx;
                        scheduler.n_block_idx =
                            n_group_idx * kL2NGroupSize + n_in_group;
                        func(sched::BlockPhase::Linear2,
                             scheduler.current_local_expert_idx,
                             L2_SHAPE_K / BLOCK_K,
                             scheduler.m_block_idx, scheduler.n_block_idx);
                    }
                } else {
                    scheduler.block_idx -= num_grouped_blocks;
                    scheduler.advance_expert_idx();
                }
            }
        } else if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            if (!phase_trace_fetch_expert_recv_count())
                return;
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2FetchOnlyReturn);
                return;
            }
            scheduler.set_expert_idx(0);
            while (scheduler.current_local_expert_idx < kNumExpertsPerRank) {
                const bool found_block = scheduler.fetch_next_l2_block();
                if (found_block) {
                    scheduler.block_idx += kNumSMs / kClusterSize;
                    func(sched::BlockPhase::Linear2, scheduler.current_local_expert_idx,
                         L2_SHAPE_K / BLOCK_K, scheduler.m_block_idx, scheduler.n_block_idx);
                } else if (scheduler.current_local_expert_idx >= kNumExpertsPerRank) {
                    break;
                }
            }
        } else {
            MegaMoEPhase::for_each_selected_block(scheduler, func);
        }
    };

    const auto cleanup_workspace = [&]() {
        if constexpr (kNumDispatchWarps > 0) {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                if constexpr (!kOneWarpCleanupRequested)
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if constexpr (kOneWarpCleanupRequested) {
                    if (warp_idx == 0) {
                        if (lane_idx == 0) {
                            *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                            if (cumulative_local_expert_recv_stats != nullptr)
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        }
                    }
                } else {
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }
                }

                if constexpr (kOneWarpCleanupRequested) {
                    if (warp_idx == 0) {
                        for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32) {
                            *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                            *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                        }
                        __syncwarp();
                    }
                } else {
                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                        *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }
        }
        }
    };

    if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only)
        phase_trace_mark(kTraceK2KernelEntry);

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    uint32_t num_dispatch_warps_runtime = kNumDispatchWarps;
    if (warp_idx < num_dispatch_warps_runtime) {
        if constexpr (kNumDispatchWarps > 0) {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2DispatchRoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        const unsigned long long dispatch_total_start = phase_profile_clock();

        if constexpr (MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2DispatchEnter);
            if (!phase_trace_fetch_expert_recv_count())
                return;
            phase_trace_mark(kTraceK2DispatchAfterFetch);
            if constexpr (MegaMoEPhase::needs_combine_reduce) {
                ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
                phase_trace_mark(kTraceK2DispatchAfterSync1);
            }
            if constexpr (kNeedsCombineBarrier && MegaMoEPhase::needs_combine_reduce) {
                ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
                phase_trace_mark(kTraceK2DispatchAfterSync2);
            }
            if constexpr (!MegaMoEPhase::needs_combine_reduce) {
                phase_trace_mark(kTraceK2DispatchReturnNoReduce);
                return;
            }
            const unsigned long long cleanup_start = phase_profile_clock();
            cleanup_workspace();
            const unsigned long long cleanup_end = phase_profile_clock();
            if (lane_idx == 0)
                phase_profile_record(kProfileWorkspaceCleanup, cleanup_end - cleanup_start);
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);
            return;
        }

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks. The PR357-style
        // single-rank route metadata path is kept for A/B testing, but is not
        // the H200 fast path because local fanout stores serialize the pull warp.
        if constexpr (kNumRanks == 1 and kEnableSingleRankRouteDispatch) {
            constexpr uint32_t kTopkRouteMask = static_cast<uint32_t>((uint64_t(1) << kNumTopk) - 1u);
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                const uint32_t src_token_idx = i + lane_idx / kNumTopk;
                const bool in_range = src_token_idx < num_tokens and lane_idx < kNumActivateLanes;
                int expert_idx = -1;
                if (in_range)
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                const bool routed = in_range and expert_idx >= 0;
                const uint32_t local_expert_idx = routed ? static_cast<uint32_t>(expert_idx) : 0u;

                uint32_t dst_slot_idx = 0;
                if (routed) {
                    dst_slot_idx = atomicAdd_block(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
                    *workspace.get_src_token_topk_idx_ptr(local_expert_idx, sym_buffer.rank_idx, dst_slot_idx) =
                        i * kNumTopk + lane_idx;
                }

                const uint32_t group_base = (lane_idx / kNumTopk) * kNumTopk;
                const uint32_t routed_mask = __ballot_sync(0xffffffffu, routed);
                const uint32_t our_routes = routed_mask & (kTopkRouteMask << group_base);
                const uint32_t route_count = __popc(our_routes);
                if (in_range and lane_idx == group_base)
                    *workspace.get_src_route_count_ptr(sym_buffer.rank_idx, src_token_idx) = route_count;
                if (routed and route_count > 1) {
                    const uint32_t packed = (dst_slot_idx << 16) | (local_expert_idx << 6) | (lane_idx - group_base);
                    const uint32_t route_slot_idx = __popc(our_routes & ((1u << lane_idx) - 1u));
                    *workspace.get_src_route_entry_ptr(sym_buffer.rank_idx, src_token_idx, route_slot_idx) = packed;
                }
                __syncwarp();
            }
        } else if constexpr (kNumRanks == 1) {
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                *workspace.get_src_token_topk_idx_ptr(
                    static_cast<uint32_t>(expert_idx), sym_buffer.rank_idx, dst_slot_idx) = token_topk_idx;
            });
        } else {
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                    expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });
        }

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                if constexpr (kNumRanks == 1) {
                    *workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx) =
                        expert_status & 0xffffffff;
                    ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), expert_status);
                } else {
                    *sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                        dst_rank_idx) = expert_status & 0xffffffff;
                    ptx::atomic_add_sys(
                        sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                        expert_status);
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        if constexpr (kNumRanks == 1) {
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
            );
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                false, true);
        }

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long dispatch_pull_start = phase_profile_clock();

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];
        constexpr bool kDeferDispatchStoreWait =
            kNumRanks == 8 and kHidden == 3072 and
            kIntermediateHidden == 1536 and kNumTopk == 16;
        bool pull_store_pending = false;
        uint32_t pending_arrival_block_idx = 0;

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Single-GPU H200 does not need rank round-robin peeling. Keep
            // the old SM100-style selection for multi-rank dispatch.
            uint32_t current_rank_in_expert_idx = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t token_idx_in_rank = token_idx_in_expert;
            if constexpr (kNumRanks > 1) {
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t slot_idx = token_idx_in_expert;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            if constexpr (kNumRanks == 1 and !kEnableSingleRankRouteDispatch) {
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = input_sf_buffer.get_data_buffer(src_token_idx).template get_base_ptr<float>();
                const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kSFPoolStrideTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                if (cute::elect_one_sync()) {
                    const auto weight = *(input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).template get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
                __syncwarp();
                continue;
            }

            if constexpr (kNumRanks == 1 and kEnableSingleRankRouteDispatch) {
                uint32_t route_count = 1;
                if (lane_idx == 0) {
                    const auto route_count_ptr = workspace.get_src_route_count_ptr(sym_buffer.rank_idx, src_token_idx);
                    route_count = *route_count_ptr;
                    if (route_count > 1)
                        route_count = atomicExch(route_count_ptr, 0u);
                }
                route_count = __shfl_sync(0xffffffffu, route_count, 0);
                if (route_count == 0)
                    continue;

                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                constexpr uint32_t kNumSFFloats = kHidden / 128;
                constexpr uint32_t kNumSFFloatsPerLane = math::constexpr_ceil_div(kNumSFFloats, 32u);
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = input_sf_buffer.get_data_buffer(src_token_idx).template get_base_ptr<float>();
                float stored_sf[kNumSFFloatsPerLane] = {};
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFFloatsPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        stored_sf[i] = remote_sf_ptr[j];
                }
                float stored_weight = 0.0f;
                if (lane_idx < kNumTopk)
                    stored_weight = *(input_topk_weights_buffer.get_base_ptr<float>() + src_token_idx * kNumTopk + lane_idx);

                if (cute::elect_one_sync()) {
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                }
                __syncwarp();

                const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();
                if (route_count == 1) {
                    const uint32_t route_token_idx_in_expert = token_idx - expert_start_idx;
                    const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                    const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                    const uint32_t pool_block_idx = expert_pool_block_offset + route_token_idx_in_expert / BLOCK_M;
                    const float weight = __shfl_sync(0xffffffffu, stored_weight, src_topk_idx);
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        ptx::tma_store_1d(
                            l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                            pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                        *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).template get_base_ptr<float>() = weight;
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                        cute::tma_store_arrive();
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFFloatsPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        if (j < kNumSFFloats)
                            local_sf_ptr[j * kSFPoolStrideTokens + sf_pool_token_idx] = stored_sf[i];
                    }
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        ptx::tma_store_wait<0>();
                        ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(pool_block_idx), 1);
                    }
                    __syncwarp();
                    continue;
                }

                uint32_t stored_pool_blocks[kNumTopk] = {};
                for (uint32_t route_idx = 0; route_idx < route_count; ++ route_idx) {
                    const uint32_t packed = *workspace.get_src_route_entry_ptr(
                        current_rank_in_expert_idx, src_token_idx, route_idx);
                    const uint32_t route_topk_slot_idx = packed & 0x3fu;
                    const uint32_t route_local_expert_idx = (packed >> 6) & 0x3ffu;
                    const uint32_t route_token_idx_in_expert = packed >> 16;
                    const uint32_t route_pool_block_offset = scheduler.get_pool_block_offset(route_local_expert_idx);
                    const uint32_t pool_token_idx = route_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                    const uint32_t sf_pool_token_idx = route_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                    const uint32_t pool_block_idx = route_pool_block_offset + route_token_idx_in_expert / BLOCK_M;
                    stored_pool_blocks[route_idx] = pool_block_idx;
                    const float weight = __shfl_sync(0xffffffffu, stored_weight, route_topk_slot_idx);
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        ptx::tma_store_1d(
                            l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                            pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                        *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).template get_base_ptr<float>() = weight;
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, route_topk_slot_idx};
                        cute::tma_store_arrive();
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFFloatsPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        if (j < kNumSFFloats)
                            local_sf_ptr[j * kSFPoolStrideTokens + sf_pool_token_idx] = stored_sf[i];
                    }
                }
                __syncwarp();
                if (cute::elect_one_sync()) {
                    ptx::tma_store_wait<0>();
                    for (uint32_t route_idx = 0; route_idx < route_count; ++ route_idx)
                        ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(stored_pool_blocks[route_idx]), 1);
                }
                __syncwarp();
                continue;
            }

            // Exact-shape dispatch experiment: bypass the per-token
            // TMA-load/SMEM/TMA-store round trip with coalesced 128-bit remote
            // loads and local stores.  The existing one-warp-cleanup selector
            // is otherwise unused by split Green K1/K2 and serves as the JIT
            // experiment bit here.
            constexpr bool kDirectDispatchCopy =
                kOneWarpCleanupRequested and kNumRanks > 1 and kHidden % sizeof(uint4) == 0;
            if constexpr (kDirectDispatchCopy) {
                constexpr uint32_t kNumTokenVec = kHidden / sizeof(uint4);
                const auto src_ptr = sym_buffer.map(
                    input_token_buffer.get_data_buffer(src_token_idx).template get_base_ptr<const uint4>(),
                    current_rank_in_expert_idx);
                const auto dst_ptr =
                    l1_token_buffer.get_data_buffer(
                        expert_pool_block_offset * BLOCK_M + token_idx_in_expert)
                        .template get_base_ptr<uint4>();
                #pragma unroll
                for (uint32_t j = lane_idx; j < kNumTokenVec; j += 32)
                    ptx::st_global_v4_u32(dst_ptr + j, ptx::ld_global_l2_256b(src_ptr + j));
                __threadfence();
            } else {
                if constexpr (kDeferDispatchStoreWait) {
                    if (pull_store_pending and cute::elect_one_sync()) {
                        ptx::tma_store_wait<0>();
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(pending_arrival_block_idx), 1);
                    }
                    __syncwarp();
                    pull_store_pending = false;
                }
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
            }
            __syncwarp();

            // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
            constexpr uint32_t kNumSFFloats = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).template get_base_ptr<float>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
            const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kSFPoolStrideTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).template get_base_ptr<float>() = weight;

                if constexpr (!kDirectDispatchCopy) {
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                }

                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                if constexpr (!kDirectDispatchCopy) {
                    cute::tma_store_arrive();
                    if constexpr (!kDeferDispatchStoreWait)
                        ptx::tma_store_wait<0>();
                }
                if constexpr (kDirectDispatchCopy or !kDeferDispatchStoreWait) {
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(
                            expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
            }
            if constexpr (!kDirectDispatchCopy and kDeferDispatchStoreWait) {
                pull_store_pending = true;
                pending_arrival_block_idx =
                    expert_pool_block_offset + token_idx_in_expert / BLOCK_M;
            }
            __syncwarp();
        }

        if constexpr (kDeferDispatchStoreWait) {
            if (pull_store_pending and cute::elect_one_sync()) {
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(pending_arrival_block_idx), 1);
            }
            __syncwarp();
        }



        // Cleanup workspace, overlapping with combine
        const unsigned long long dispatch_pull_end = phase_profile_clock();
        if (lane_idx == 0) {
            phase_profile_record(kProfileDispatchPull, dispatch_pull_end - dispatch_pull_start);
            phase_profile_record(kProfileDispatchTotal, dispatch_pull_end - dispatch_total_start);
        }
        if constexpr (MegaMoEPhase::is_linear1_only)
            return;

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long cleanup_start = phase_profile_clock();
        cleanup_workspace();
        const unsigned long long cleanup_end = phase_profile_clock();
        if (lane_idx == 0)
            phase_profile_record(kProfileWorkspaceCleanup, cleanup_end - cleanup_start);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);
        }

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Default: 4 non-epilogue warps, two active and two idle.
    //   Compact frontend mode: 2 dispatch warps + 2 TMA warps share the first
    //   warpgroup, reducing total CTA threads for the M128/2WG path.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2TmaARoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_a_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            if (has_valid_m) {
                if (is_linear1_phase) {
                    const unsigned long long arrival_wait_start = phase_profile_clock();
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                    const unsigned long long arrival_wait_end = phase_profile_clock();
                    if (lane_idx == 0)
                        phase_profile_record(kProfileL1ArrivalWait, arrival_wait_end - arrival_wait_start);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaABeforeEmptyWait);
                if (!phase_trace_wait_barrier(empty_barriers[stage_idx], phase ^ 1,
                                              kTraceK2TmaAEmptyTimeout))
                    return;
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaAAfterEmptyWait);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    if (!is_linear1_phase) {
                        constexpr uint32_t kNumReadyBits = BLOCK_K / 64;
                        const uint64_t expected_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                       << (k_block_idx * kNumReadyBits);
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        phase_trace_mark(kTraceK2TmaABeforeArrivalWait);
                        const unsigned long long arrival_wait_start = phase_profile_clock();
                        if (!phase_trace_wait_l2_arrival(ptr, expected_mask, kTraceK2TmaAArrivalTimeout))
                            return;
                        const unsigned long long arrival_wait_end = phase_profile_clock();
                        phase_trace_mark(kTraceK2TmaAAfterArrivalWait);
                        if (lane_idx == 0)
                            phase_profile_record(kProfileL2ArrivalWait, arrival_wait_end - arrival_wait_start);
                    }

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    if constexpr (kDropL2SFA) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                    } else if constexpr (kSplitSFATMA) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                    } else {
                        // TMA load SFA
                        if (is_linear1_phase) {
                            // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                        } else {
                            if constexpr (kL2SFA2RowTMA) {
                                tma::copy<BLOCK_M, 2, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                    m_idx, k_block_idx * 2, 1);
                            } else {
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                    m_idx, k_block_idx * 2, 1);
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx],
                                    smem_sfa[stage_idx] + kL2SFAHalfStride,
                                    m_idx, k_block_idx * 2 + 1, 1);
                            }
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaAAfterFullArrive);
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2TmaBRoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_b_ptr =
                !is_linear1_phase ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = !is_linear1_phase ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaBBeforeEmptyWait);
                if (!phase_trace_wait_barrier(empty_barriers[stage_idx], phase ^ 1,
                                              kTraceK2TmaBEmptyTimeout))
                    return;
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaBAfterEmptyWait);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B (weight SF is now loaded directly by math warps from global)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                        k_idx, n_idx, kClusterSize);

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2TmaBAfterFullArrive);
            }
        });

    } else if (kSplitSFATMA && !kDropL2SFA && warp_idx == kNumDispatchWarps + 2) {
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2TmaARoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            (void)local_expert_idx;
            (void)n_block_idx;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            if (has_valid_m) {
                if (is_linear1_phase) {
                    const unsigned long long arrival_wait_start = phase_profile_clock();
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                    const unsigned long long arrival_wait_end = phase_profile_clock();
                    if (lane_idx == 0)
                        phase_profile_record(kProfileL1ArrivalWait, arrival_wait_end - arrival_wait_start);
                }
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    if (!is_linear1_phase) {
                        constexpr uint32_t kNumReadyBits = BLOCK_K / 64;
                        const uint64_t expected_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                       << (k_block_idx * kNumReadyBits);
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        const unsigned long long arrival_wait_start = phase_profile_clock();
                        while ((ptx::ld_acq_gpu(ptr) & expected_mask) != expected_mask);
                        const unsigned long long arrival_wait_end = phase_profile_clock();
                        if (lane_idx == 0)
                            phase_profile_record(kProfileL2ArrivalWait, arrival_wait_end - arrival_wait_start);
                    }

                    if (is_linear1_phase) {
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
                    } else {
                        if constexpr (kL2SFA2RowTMA) {
                            tma::copy<BLOCK_M, 2, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx * 2, 1);
                        } else {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, k_block_idx * 2 + 1, 1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(2 * BLOCK_M * sizeof(float));
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        });

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2IdleRoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        if constexpr (kPhaseProfileRequested and MegaMoEPhase::is_linear2_only) {
            phase_trace_mark(kTraceK2MathRoleEntry);
            if constexpr (kDirectL2ScatterRequested) {
                phase_trace_mark(kTraceK2EntryOnlyReturn);
                return;
            }
        }
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        uint32_t async_l1_store_stage = 0;
        bool async_l1_store_pending0 = false;
        bool async_l1_store_pending1 = false;
        uint32_t async_l1_store_pool_block_idx0 = 0;
        uint32_t async_l1_store_pool_block_idx1 = 0;
        uint64_t async_l1_store_ready_mask0 = 0;
        uint64_t async_l1_store_ready_mask1 = 0;

        const auto publish_l1_ready_mask = [&](const uint32_t& pool_block_idx,
                                                const uint64_t& ready_mask) {
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                const unsigned long long ready_mask_start = phase_profile_clock();
                ptx::red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx), ready_mask);
                const unsigned long long ready_mask_end = phase_profile_clock();
                phase_profile_record(kProfileL1ReadyMask, ready_mask_end - ready_mask_start);
            }
        };

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if constexpr (kClusterSize == 1) {
                if (lane_idx == 0)
                    empty_barriers[s]->arrive();
            } else {
                if (lane_idx < kClusterSize)
                    empty_barriers[s]->arrive(lane_idx);
            }
        };

        const auto drain_async_l1_store_stage = [&](const uint32_t& store_stage) {
            if constexpr (kAsyncL1TMAStore) {
                const bool pending = store_stage == 0 ? async_l1_store_pending0 : async_l1_store_pending1;
                if (pending) {
                    // Two SMEM L1 store buffers are used in FIFO order; waiting
                    // for <=1 outstanding store makes the older buffer reusable.
                    const unsigned long long tma_wait_start = phase_profile_clock();
                    ptx::tma_store_wait<1>();
                    const unsigned long long tma_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1TMAStoreWait, tma_wait_end - tma_wait_start);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    publish_l1_ready_mask(
                        store_stage == 0 ? async_l1_store_pool_block_idx0 : async_l1_store_pool_block_idx1,
                        store_stage == 0 ? async_l1_store_ready_mask0 : async_l1_store_ready_mask1);
                    if (store_stage == 0)
                        async_l1_store_pending0 = false;
                    else
                        async_l1_store_pending1 = false;
                }
            }
        };

        const auto drain_all_async_l1_stores = [&]() {
            if constexpr (kAsyncL1TMAStore) {
                const bool pending0 = async_l1_store_pending0;
                const bool pending1 = async_l1_store_pending1;
                if (pending0 or pending1) {
                    const unsigned long long tma_wait_start = phase_profile_clock();
                    ptx::tma_store_wait<0>();
                    const unsigned long long tma_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1TMAStoreWait, tma_wait_end - tma_wait_start);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (pending0)
                        publish_l1_ready_mask(async_l1_store_pool_block_idx0, async_l1_store_ready_mask0);
                    if (pending1)
                        publish_l1_ready_mask(async_l1_store_pool_block_idx1, async_l1_store_ready_mask1);
                    async_l1_store_pending0 = false;
                    async_l1_store_pending1 = false;
                }
            }
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSplitNWarpgroups || (BLOCK_M % kNumEpilogueWarpgroups == 0), "Invalid block M");
        if constexpr (kSplitNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Split-N WGs must each run one M64N128 WGMMA per K-block");
        } else if constexpr (kSerialNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Serial-N path runs two M64N128 WGMMAs per K-block");
        } else {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        }

        if constexpr (kNumDispatchWarps > 0) {
            if constexpr (MegaMoEPhase::needs_dispatch_pull or MegaMoEPhase::needs_combine_reduce) {
                ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            }
        }
        if constexpr (MegaMoEPhase::is_linear2_only)
            phase_trace_mark(kTraceK2MathAfterDispatchSync);
        const unsigned long long math_loop_start = phase_profile_clock();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t wg_n_idx = kSplitNWarpgroups ? epilogue_wg_idx * WG_BLOCK_N : 0;
            const uint32_t wg_l1_out_n_idx = kSplitNWarpgroups ? epilogue_wg_idx * WG_L1_OUT_BLOCK_N : 0;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset = kSplitNWarpgroups ? 0 : epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            if constexpr (MegaMoEPhase::is_linear2_only)
                phase_trace_mark(kTraceK2MathBlockStart);
            if constexpr (kAsyncL1TMAStore) {
                if (!is_linear1_phase)
                    drain_all_async_l1_stores();
            }

            if constexpr (kSerialNWarpgroups) {
                using WGMMA = L1WGMMA;
                constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
                constexpr uint32_t kNumSerialN = 2;
                float final_accum[kNumSerialN][kAccumPerThread] = {};
                float accum[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;
                    if (is_linear1_phase) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                    }

                    constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                    constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_wg_n_idx = serial_n_idx * WG_BLOCK_N;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (is_linear1_phase) {
                            const uint32_t gate_n = (n_block_idx * BLOCK_N + serial_wg_n_idx) / 256u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                            gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                            up_sf   = __ldg(base + up_n   * kL1SFKBlocks);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                            }
                        } else {
                            l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert
                                                            + ((n_block_idx * BLOCK_N + serial_wg_n_idx) / 128u) * kL2SFKBlocks + k_block_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                            }

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                            }
                        }
                    }

                    arrive_empty_barrier(stage_idx);
                    __syncwarp();
                }

                if (row_block_offset >= valid_m) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }

                if (is_linear1_phase) {
                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_l1_out_n_idx = serial_n_idx * WG_L1_OUT_BLOCK_N;
                        float swiglu_r0[kNumPairs][2];
                        float swiglu_r1[kNumPairs][2];
                        float amax_r0 = 0.0f, amax_r1 = 0.0f;

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const uint32_t gate = 2 * p, up = 2 * p + 1;
                            auto clamp_gate = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(x, kActivationClamp);
                            };
                            auto clamp_up = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                            };
                            float g_r0_c0 = final_accum[serial_n_idx][gate*4 + 0]; clamp_gate(g_r0_c0);
                            float g_r0_c1 = final_accum[serial_n_idx][gate*4 + 1]; clamp_gate(g_r0_c1);
                            float g_r1_c0 = final_accum[serial_n_idx][gate*4 + 2]; clamp_gate(g_r1_c0);
                            float g_r1_c1 = final_accum[serial_n_idx][gate*4 + 3]; clamp_gate(g_r1_c1);
                            float u_r0_c0 = final_accum[serial_n_idx][up*4   + 0]; clamp_up(u_r0_c0);
                            float u_r0_c1 = final_accum[serial_n_idx][up*4   + 1]; clamp_up(u_r0_c1);
                            float u_r1_c0 = final_accum[serial_n_idx][up*4   + 2]; clamp_up(u_r1_c0);
                            float u_r1_c1 = final_accum[serial_n_idx][up*4   + 3]; clamp_up(u_r1_c1);
                            auto silu = [](float x) -> float {
                                const float sig = kFastMath ?
                                    0.5f * (__tanhf(0.5f * x) + 1.0f) :
                                    1.0f / (1.0f + expf(-x));
                                return x * sig;
                            };
                            if (valid_r0) {
                                swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                                swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                                amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                            } else {
                                swiglu_r0[p][0] = 0.0f;
                                swiglu_r0[p][1] = 0.0f;
                            }
                            if (valid_r1) {
                                swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                                swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                                amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                            } else {
                                swiglu_r1[p][0] = 0.0f;
                                swiglu_r1[p][1] = 0.0f;
                            }
                        }

                        const float weight_r0 = [&]() {
                            if constexpr (kNumMaxTokensPerRank <= 1024) {
                                float weight = 0.0f;
                                if (col_idx == 0)
                                    weight = valid_r0 ? *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + row_offset_r0)
                                        .template get_base_ptr<float>() : 0.0f;
                                return __shfl_sync(0xffffffff, weight, static_cast<int>(lane_idx - col_idx));
                            } else {
                                return valid_r0 ? *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + row_offset_r0)
                                    .template get_base_ptr<float>() : 0.0f;
                            }
                        }();
                        const float weight_r1 = [&]() {
                            if constexpr (kNumMaxTokensPerRank <= 1024) {
                                float weight = 0.0f;
                                if (col_idx == 0)
                                    weight = valid_r1 ? *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + row_offset_r1)
                                        .template get_base_ptr<float>() : 0.0f;
                                return __shfl_sync(0xffffffff, weight, static_cast<int>(lane_idx - col_idx));
                            } else {
                                return valid_r1 ? *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + row_offset_r1)
                                    .template get_base_ptr<float>() : 0.0f;
                            }
                        }();
                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            swiglu_r0[p][0] *= weight_r0;
                            swiglu_r0[p][1] *= weight_r0;
                            swiglu_r1[p][0] *= weight_r1;
                            swiglu_r1[p][1] *= weight_r1;
                        }
                        amax_r0 *= cute::abs(weight_r0);
                        amax_r1 *= cute::abs(weight_r1);
                        amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                        amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                        float sf_r0, sf_inv_r0, sf_r1, sf_inv_r1;
                        {
                            float2 amax_pair = {amax_r0, amax_r1};
                            float2 sf_pair, sf_inv_pair;
                            math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                            sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                        }

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                            const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                            const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                            const float v11 = swiglu_r1[p][1] * sf_inv_r1;
                            const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                            const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                            const uint32_t col = p * 8 + col_idx * 2;
                            auto* p0 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_0 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            auto* p1 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_1 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            if (valid_r0)
                                *p0 = r0_pair.__x;
                            if (valid_r1)
                                *p1 = r1_pair.__x;
                        }

                        if (col_idx == 0) {
                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                            const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                            const uint32_t k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + serial_l1_out_n_idx) / 64u;
                            if (valid_r0)
                                sf_base_ptr[k_sf_idx * kSFPoolStrideTokens + token_r0] = sf_r0;
                            if (valid_r1)
                                sf_base_ptr[k_sf_idx * kSFPoolStrideTokens + token_r1] = sf_r1;
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        constexpr uint32_t kNumReadyBits = L1_OUT_BLOCK_N / 64;
                        const uint64_t ready_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                   << ((n_block_idx * L1_OUT_BLOCK_N) / 64u);
                        ptx::red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx), ready_mask);
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_n_idx_base = n_block_idx * BLOCK_N + serial_n_idx * WG_BLOCK_N;

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                                auto smem_ptr = smem_cd_l2 + row * WG_BLOCK_N + col;
                                *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                            };
                            if (valid_r0) {
                                const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_lo*4 + 0], final_accum[serial_n_idx][chunk_lo*4 + 1]);
                                const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_hi*4 + 0], final_accum[serial_n_idx][chunk_hi*4 + 1]);
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                            }
                            if (valid_r1) {
                                const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_lo*4 + 2], final_accum[serial_n_idx][chunk_lo*4 + 3]);
                                const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_hi*4 + 2], final_accum[serial_n_idx][chunk_hi*4 + 3]);
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                            }
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        const uint32_t row_in_warp_block = lane_idx / 16;
                        const uint32_t lane_in_row = lane_idx % 16;
                        constexpr uint32_t cols_per_lane = WG_BLOCK_N / 16;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                            if (m_idx_in_block >= valid_m) break;

                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx();
                            const uint32_t dst_token_idx = src_metadata.token_idx();
                            const uint32_t dst_topk_idx = src_metadata.topk_idx();
                            auto smem_ptr = smem_cd_l2 + row_in_wg * WG_BLOCK_N + lane_in_row * cols_per_lane;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                serial_n_idx_base * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                }
                return;
            }

            // ---------------- GEMM ----------------
            using WGMMA = std::conditional_t<is_linear1_phase, L1WGMMA, L2WGMMA>;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread];
            if constexpr (!kL2FirstAccumAssign) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                    final_accum[i] = 0.0f;
            }
            float accum[kAccumPerThread];

            const unsigned long long block_gemm_start = phase_profile_clock();
            const auto run_default_gemm_loop = [&]() {
                // ----- Block (128, 128) weight SF (loaded directly from global) -----
                // L1 weight SF shape: (E, 2*IH/128, H/128) MN-major. The N axis is
                // [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave on
                // the FP8 weight, each BLOCK_N=128 tile covers 64 rows of gate plus
                // 64 rows of up taken from the same original 128-row block, so:
                //     gate_sf_n = n_block_idx / 2
                //     up_sf_n   = (IH/128) + n_block_idx / 2
                //
                // L2 weight SF shape: (E, H/128, IH/128) MN-major. One scalar per
                // (BLOCK_N, BLOCK_K) tile, broadcast across all WGMMA accumulators.
                //
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                auto record_l2_scale_factor_load = [&](const unsigned long long& start_clock,
                                                       const unsigned long long& end_clock) {
                    if (!is_linear1_phase and epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ScaleFactorLoad, end_clock - start_clock);
                };
                auto record_l2_activation_scale_load = [&](const unsigned long long& start_clock,
                                                           const unsigned long long& end_clock) {
                    if (!is_linear1_phase and epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ActivationScaleLoad, end_clock - start_clock);
                };
                auto record_l2_weight_scale_load = [&](const unsigned long long& start_clock,
                                                       const unsigned long long& end_clock) {
                    if (!is_linear1_phase and epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2WeightScaleLoad, end_clock - start_clock);
                };
                const float* l2_weight_sf_hoisted_base = nullptr;
                if constexpr (kL2HoistWeightSFBase) {
                    const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                    l2_weight_sf_hoisted_base = l2_weights_sf + local_expert_idx * kL2SFPerExpert +
                        sf_n * kL2SFKBlocks;
                }
                auto load_l2_weight_sf_scalar = [&](const float* ptr) {
                    if constexpr (kL2WeightSFNC)
                        return ptx::ld_global_nc_f32(ptr);
                    else if constexpr (kL2WeightSFLDU)
                        return ptx::ldu_global(ptr);
                    else
                        return __ldg(ptr);
                };
                auto load_l2_weight_scale_factors_for = [&](const uint32_t& sf_k_block_idx,
                                                            float& out_lo, float& out_hi) {
                    const unsigned long long l2_weight_scale_load_start = phase_profile_clock();
                    if constexpr (kL2HoistWeightSFBase) {
                        const auto* sf_ptr = l2_weight_sf_hoisted_base + sf_k_block_idx;
                        out_lo = load_l2_weight_sf_scalar(sf_ptr);
                        out_hi = out_lo;
                    } else if constexpr (kL2FastWeightSFN) {
                        const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_k_block_idx;
                        const uint32_t sf_n = n_block_idx * 2u + epilogue_wg_idx;
                        const auto* sf_ptr = base + sf_n * kL2SFKBlocks;
                        out_lo = load_l2_weight_sf_scalar(sf_ptr);
                        out_hi = out_lo;
                    } else {
                        const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_k_block_idx;
                        const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                        const auto* sf_lo_ptr = base + sf_n * kL2SFKBlocks;
                        out_lo = load_l2_weight_sf_scalar(sf_lo_ptr);
                        if constexpr (WG_BLOCK_N > 128) {
                            const auto* sf_hi_ptr = base + (sf_n + 1u) * kL2SFKBlocks;
                            out_hi = load_l2_weight_sf_scalar(sf_hi_ptr);
                        } else {
                            out_hi = out_lo;
                        }
                    }
                    const unsigned long long l2_weight_scale_load_end = phase_profile_clock();
                    record_l2_weight_scale_load(l2_weight_scale_load_start, l2_weight_scale_load_end);
                };
                float prefetched_l2_sf_lo = 0.0f, prefetched_l2_sf_hi = 0.0f;
                if constexpr (kL2PrefetchWeightSF) {
                    if (!is_linear1_phase && num_k_blocks > 0)
                        load_l2_weight_scale_factors_for(0, prefetched_l2_sf_lo, prefetched_l2_sf_hi);
                }

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeFullWait);
                    const unsigned long long full_barrier_start = phase_profile_clock();
                    if (!phase_trace_wait_barrier(full_barriers[stage_idx], phase,
                                                  kTraceK2MathFullTimeout))
                        return;
                    const unsigned long long full_barrier_end = phase_profile_clock();
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterFullWait);
                    if constexpr (MegaMoEPhase::is_linear2_only && kSplitNWarpgroups && WG_BLOCK_N == 64)
                        phase_trace_mark(kTraceK2MathWGAfterFullWaitBase + epilogue_wg_idx);
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileFullBarrierWait, full_barrier_end - full_barrier_start);

                    float scale_a_0_lo = 0.0f, scale_a_1_lo = 0.0f;
                    float scale_a_0_hi = 0.0f, scale_a_1_hi = 0.0f;  // Only used in L2 (per-64 K)

                float gate_sf = 0.0f, up_sf = 0.0f;
                float l2_sf_lo = 0.0f, l2_sf_hi = 0.0f;
                // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M.
                auto load_l2_activation_scale_factors_lo = [&]() {
                    const unsigned long long l2_activation_scale_load_start = phase_profile_clock();
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    const unsigned long long l2_activation_scale_load_end = phase_profile_clock();
                    record_l2_activation_scale_load(l2_activation_scale_load_start, l2_activation_scale_load_end);
                };
                auto load_l2_activation_scale_factors_hi = [&]() {
                    const unsigned long long l2_activation_scale_load_start = phase_profile_clock();
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                    const unsigned long long l2_activation_scale_load_end = phase_profile_clock();
                    record_l2_activation_scale_load(l2_activation_scale_load_start, l2_activation_scale_load_end);
                };
                auto load_l2_activation_scale_factors = [&]() {
                    load_l2_activation_scale_factors_lo();
                    load_l2_activation_scale_factors_hi();
                };
                auto load_l2_weight_scale_factors = [&]() {
                    load_l2_weight_scale_factors_for(k_block_idx, l2_sf_lo, l2_sf_hi);
                };
                auto load_l2_scale_factors = [&]() {
                    load_l2_activation_scale_factors();
                    load_l2_weight_scale_factors();
                };
                if (is_linear1_phase) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                } else if constexpr (kScaleFreeL1K2) {
                    // Preserve the existing SFA TMA/barrier protocol for this
                    // first attributable experiment, but make K2 math depend
                    // only on the per-K128 weight scale.
                    const unsigned long long l2_scale_factor_load_start = phase_profile_clock();
                    load_l2_weight_scale_factors();
                    const unsigned long long l2_scale_factor_load_end = phase_profile_clock();
                    record_l2_scale_factor_load(l2_scale_factor_load_start, l2_scale_factor_load_end);
                } else if constexpr (!kL2DelaySFALoad) {
                    const unsigned long long l2_scale_factor_load_start = phase_profile_clock();
                    if constexpr (kL2PrefetchWeightSF) {
                        if constexpr (kL2SplitSFALoad) {
                            load_l2_activation_scale_factors_lo();
                        } else {
                            load_l2_activation_scale_factors();
                        }
                        l2_sf_lo = prefetched_l2_sf_lo;
                        l2_sf_hi = prefetched_l2_sf_hi;
                    } else if constexpr (kL2SplitSFALoad) {
                        load_l2_activation_scale_factors_lo();
                        load_l2_weight_scale_factors();
                    } else {
                        load_l2_scale_factors();
                    }
                    const unsigned long long l2_scale_factor_load_end = phase_profile_clock();
                    record_l2_scale_factor_load(l2_scale_factor_load_start, l2_scale_factor_load_end);
                }

                if (is_linear1_phase) {
                    // Single per-128 K-block WGMMA group
                    const unsigned long long l1_wgmma_start = phase_profile_clock();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();
                    const unsigned long long l1_wgmma_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1WgmmaGroup, l1_wgmma_end - l1_wgmma_start);

                    arrive_empty_barrier(stage_idx);

                    // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                    // cols belongs entirely to one of {gate, up}, so .x and .y
                    // share the same scalar.
                    const unsigned long long l1_scale_start = phase_profile_clock();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float sb = (i & 1u) ? up_sf : gate_sf;
                        final_accum[i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                    }
                    const unsigned long long l1_scale_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1AccumScale, l1_scale_end - l1_scale_start);
                } else {
                    const unsigned long long l2_iter_core_start = phase_profile_clock();
                    if constexpr (kScaleFreeL1K2) {
                        DG_STATIC_ASSERT(WG_BLOCK_N == 128 && BLOCK_K == 128,
                                         "Scale-free K2 expects one M64N128K128 tile");

                        const unsigned long long l2_wgmma_start = phase_profile_clock();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k != 0);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        const unsigned long long l2_wgmma_wait_start = phase_profile_clock();
                        ptx::warpgroup_wait<0>();
                        const unsigned long long l2_wgmma_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            phase_profile_record(kProfileL2WgmmaLo,
                                                 l2_wgmma_end - l2_wgmma_start);
                            phase_profile_record(kProfileL2WgmmaLoIssue,
                                                 l2_wgmma_wait_start - l2_wgmma_start);
                            phase_profile_record(kProfileL2WgmmaLoWait,
                                                 l2_wgmma_end - l2_wgmma_wait_start);
                        }

                        const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathBeforeEmptyArrive);
                        arrive_empty_barrier(stage_idx);
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathAfterEmptyArrive);
                        const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2EmptyBarrierArrive,
                                                 l2_empty_barrier_end - l2_empty_barrier_start);

                        const unsigned long long l2_scale_start = phase_profile_clock();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            final_accum[i] += l2_sf_lo * accum[i];
                        const unsigned long long l2_scale_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2ScaleLo, l2_scale_end - l2_scale_start);
                    } else if constexpr (kL2DualAccum) {
                        float accum_hi[kAccumPerThread];

                        const auto desc_a_lo0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K, 1);
                        const auto desc_b_lo0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K, 1);
                        const auto desc_a_lo1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + WGMMA::K, 1);
                        const auto desc_b_lo1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + WGMMA::K, 1);
                        const auto desc_a_hi0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_b_hi0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_a_hi1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);
                        const auto desc_b_hi1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_arrive();
                        WGMMA::wgmma(desc_a_lo0, desc_b_lo0, accum, false);
                        WGMMA::wgmma(desc_a_lo1, desc_b_lo1, accum, true);
                        WGMMA::wgmma(desc_a_hi0, desc_b_hi0, accum_hi, false);
                        WGMMA::wgmma(desc_a_hi1, desc_b_hi1, accum_hi, true);
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_wait<0>();

                        const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                        arrive_empty_barrier(stage_idx);
                        const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2EmptyBarrierArrive,
                                                 l2_empty_barrier_end - l2_empty_barrier_start);

                        if constexpr (WG_BLOCK_N == 128) {
                            const float scale_0_lo = scale_a_0_lo * l2_sf_lo;
                            const float scale_1_lo = scale_a_1_lo * l2_sf_lo;
                            const float scale_0_hi = scale_a_0_hi * l2_sf_lo;
                            const float scale_1_hi = scale_a_1_hi * l2_sf_lo;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_0_lo * accum[i*4+0];
                                final_accum[i*4+1] += scale_0_lo * accum[i*4+1];
                                final_accum[i*4+2] += scale_1_lo * accum[i*4+2];
                                final_accum[i*4+3] += scale_1_lo * accum[i*4+3];
                                final_accum[i*4+0] += scale_0_hi * accum_hi[i*4+0];
                                final_accum[i*4+1] += scale_0_hi * accum_hi[i*4+1];
                                final_accum[i*4+2] += scale_1_hi * accum_hi[i*4+2];
                                final_accum[i*4+3] += scale_1_hi * accum_hi[i*4+3];
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                                final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                                final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum_hi[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum_hi[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum_hi[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum_hi[i*4+3];
                            }
                        }
                    } else {
                    if constexpr (kL2N64Interleave) {
                        DG_STATIC_ASSERT(L2N64WGMMA::N == 64 and L2N64WGMMA::K == WGMMA::K,
                                         "Unexpected N64 WGMMA shape");
                        constexpr uint32_t kN64AccumPerThread = L2N64WGMMA::kNumAccum;
                        DG_STATIC_ASSERT(kN64AccumPerThread * 2 == kAccumPerThread,
                                         "N64 serial path expects half of the N128 accumulator footprint");
                        // Reuse one N64 WGMMA fragment for the two output-column
                        // halves.  The per-output K accumulation order is unchanged,
                        // while the live temporary accumulator footprint drops from
                        // N128 to N64 and avoids the BM192 L2 spill cliff.
                        float* accum_n = accum;
                        const auto scale_l2_n64 = [&](const uint32_t& dst_base,
                                                       const float& scale_0, const float& scale_1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kN64AccumPerThread / 4; ++ i) {
                                final_accum[dst_base + i*4+0] += scale_0 * accum_n[i*4+0];
                                final_accum[dst_base + i*4+1] += scale_0 * accum_n[i*4+1];
                                final_accum[dst_base + i*4+2] += scale_1 * accum_n[i*4+2];
                                final_accum[dst_base + i*4+3] += scale_1 * accum_n[i*4+3];
                            }
                        };
                        const auto run_l2_n64_wgmma = [&](const uint32_t& n_offset,
                                                           const uint32_t& k_offset) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kN64AccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum_n[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / L2N64WGMMA::K; ++ k) {
                                const uint32_t k_off = k_offset + k * L2N64WGMMA::K;
                                const auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                const auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + n_offset * BLOCK_K + k_off, 1);
                                L2N64WGMMA::wgmma(desc_a, desc_b, accum_n, k);
                            }
                            ptx::warpgroup_commit_batch();
                            if constexpr (!kL2SkipPostWGMMAFence) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kN64AccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum_n[i]);
                            }
                            ptx::warpgroup_wait<0>();
                        };

                        const float scale_0_lo = scale_a_0_lo * l2_sf_lo;
                        const float scale_1_lo = scale_a_1_lo * l2_sf_lo;
                        const float scale_0_hi = scale_a_0_hi * l2_sf_lo;
                        const float scale_1_hi = scale_a_1_hi * l2_sf_lo;

                        const unsigned long long l2_wgmma_lo_start = phase_profile_clock();
                        run_l2_n64_wgmma(0, 0);
                        const unsigned long long l2_scale_lo_start = phase_profile_clock();
                        scale_l2_n64(0, scale_0_lo, scale_1_lo);
                        run_l2_n64_wgmma(64, 0);
                        scale_l2_n64(kN64AccumPerThread, scale_0_lo, scale_1_lo);
                        const unsigned long long l2_scale_lo_end = phase_profile_clock();
                        const unsigned long long l2_wgmma_lo_end = l2_scale_lo_end;
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            phase_profile_record(kProfileL2WgmmaLo, l2_wgmma_lo_end - l2_wgmma_lo_start);
                            phase_profile_record(kProfileL2ScaleLo, l2_scale_lo_end - l2_scale_lo_start);
                        }

                        const unsigned long long l2_wgmma_hi_start = phase_profile_clock();
                        run_l2_n64_wgmma(0, BLOCK_K / 2);
                        const unsigned long long l2_scale_hi_start = phase_profile_clock();
                        scale_l2_n64(0, scale_0_hi, scale_1_hi);
                        run_l2_n64_wgmma(64, BLOCK_K / 2);

                        const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathBeforeEmptyArrive);
                        arrive_empty_barrier(stage_idx);
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathAfterEmptyArrive);
                        const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2EmptyBarrierArrive,
                                                 l2_empty_barrier_end - l2_empty_barrier_start);

                        scale_l2_n64(kN64AccumPerThread, scale_0_hi, scale_1_hi);
                        const unsigned long long l2_scale_hi_end = phase_profile_clock();
                        const unsigned long long l2_wgmma_hi_end = l2_scale_hi_end;
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            phase_profile_record(kProfileL2WgmmaHi, l2_wgmma_hi_end - l2_wgmma_hi_start);
                            phase_profile_record(kProfileL2ScaleHi, l2_scale_hi_end - l2_scale_hi_start);
                        }
                    } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    float scale_0_lo_pre = 0.0f, scale_1_lo_pre = 0.0f;
                    const unsigned long long l2_wgmma_lo_start = phase_profile_clock();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    if constexpr (!kL2SkipPostWGMMAFence) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                    }
                    if constexpr (kL2DelaySFALoad) {
                        const unsigned long long l2_scale_factor_load_start = phase_profile_clock();
                        load_l2_scale_factors();
                        const unsigned long long l2_scale_factor_load_end = phase_profile_clock();
                        record_l2_scale_factor_load(l2_scale_factor_load_start, l2_scale_factor_load_end);
                    }
                    if constexpr (kL2PrecomputeScale) {
                        asm volatile("mul.rn.f32 %0, %1, %2;"
                                     : "=f"(scale_0_lo_pre)
                                     : "f"(scale_a_0_lo), "f"(l2_sf_lo));
                        asm volatile("mul.rn.f32 %0, %1, %2;"
                                     : "=f"(scale_1_lo_pre)
                                     : "f"(scale_a_1_lo), "f"(l2_sf_lo));
                    }
                    const unsigned long long l2_wgmma_lo_wait_start = phase_profile_clock();
                    ptx::warpgroup_wait<0>();
                    const unsigned long long l2_wgmma_lo_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2WgmmaLo, l2_wgmma_lo_end - l2_wgmma_lo_start);
                    if (epilogue_warp_idx == 0 and lane_idx == 0) {
                        phase_profile_record(kProfileL2WgmmaLoIssue,
                                             l2_wgmma_lo_wait_start - l2_wgmma_lo_start);
                        phase_profile_record(kProfileL2WgmmaLoWait,
                                             l2_wgmma_lo_end - l2_wgmma_lo_wait_start);
                    }

                    // L2 weight SF is per 128 output columns; M64N256 split WGs see one SF group per WG.
                    const unsigned long long l2_scale_lo_start = phase_profile_clock();
                    if constexpr (WG_BLOCK_N == 128) {
                        const float scale_0_lo = kL2PrecomputeScale ? scale_0_lo_pre : scale_a_0_lo * l2_sf_lo;
                        const float scale_1_lo = kL2PrecomputeScale ? scale_1_lo_pre : scale_a_1_lo * l2_sf_lo;
                        if constexpr (kL2RatioAccum) {
                            const float scale_0_hi = scale_a_0_hi * l2_sf_lo;
                            const float scale_1_hi = scale_a_1_hi * l2_sf_lo;
                            const bool scale_0_hi_nonzero = scale_a_0_hi != 0.0f;
                            const bool scale_1_hi_nonzero = scale_a_1_hi != 0.0f;
                            float inv_scale_a_0_hi = 0.0f, inv_scale_a_1_hi = 0.0f;
                            if (scale_0_hi_nonzero)
                                asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                             : "=f"(inv_scale_a_0_hi) : "f"(scale_a_0_hi));
                            if (scale_1_hi_nonzero)
                                asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                             : "=f"(inv_scale_a_1_hi) : "f"(scale_a_1_hi));
                            const float ratio_0 = scale_a_0_lo * inv_scale_a_0_hi;
                            const float ratio_1 = scale_a_1_lo * inv_scale_a_1_hi;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                if (scale_0_hi_nonzero) {
                                    accum[i*4+0] *= ratio_0;
                                    accum[i*4+1] *= ratio_0;
                                } else {
                                    final_accum[i*4+0] += scale_0_lo * accum[i*4+0];
                                    final_accum[i*4+1] += scale_0_lo * accum[i*4+1];
                                    accum[i*4+0] = 0.0f;
                                    accum[i*4+1] = 0.0f;
                                }
                                if (scale_1_hi_nonzero) {
                                    accum[i*4+2] *= ratio_1;
                                    accum[i*4+3] *= ratio_1;
                                } else {
                                    final_accum[i*4+2] += scale_1_lo * accum[i*4+2];
                                    final_accum[i*4+3] += scale_1_lo * accum[i*4+3];
                                    accum[i*4+2] = 0.0f;
                                    accum[i*4+3] = 0.0f;
                                }
                            }
                        } else if constexpr (kL2FirstAccumAssign) {
                            if (k_block_idx == 0) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    final_accum[i*4+0] = __fmaf_rn(scale_0_lo, accum[i*4+0], 0.0f);
                                    final_accum[i*4+1] = __fmaf_rn(scale_0_lo, accum[i*4+1], 0.0f);
                                    final_accum[i*4+2] = __fmaf_rn(scale_1_lo, accum[i*4+2], 0.0f);
                                    final_accum[i*4+3] = __fmaf_rn(scale_1_lo, accum[i*4+3], 0.0f);
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    final_accum[i*4+0] += scale_0_lo * accum[i*4+0];
                                    final_accum[i*4+1] += scale_0_lo * accum[i*4+1];
                                    final_accum[i*4+2] += scale_1_lo * accum[i*4+2];
                                    final_accum[i*4+3] += scale_1_lo * accum[i*4+3];
                                }
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_0_lo * accum[i*4+0];
                                final_accum[i*4+1] += scale_0_lo * accum[i*4+1];
                                final_accum[i*4+2] += scale_1_lo * accum[i*4+2];
                                final_accum[i*4+3] += scale_1_lo * accum[i*4+3];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                            final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                        }
                    }
                    const unsigned long long l2_scale_lo_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ScaleLo, l2_scale_lo_end - l2_scale_lo_start);

                    // Second half: K=64..127, SFA = scale_a_*_hi
                    float scale_0_hi_pre = 0.0f, scale_1_hi_pre = 0.0f;
                    const unsigned long long l2_wgmma_hi_start = phase_profile_clock();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, kL2RatioAccum ? true : static_cast<bool>(k));
                    }
                    ptx::warpgroup_commit_batch();
                    if constexpr (!kL2SkipPostWGMMAFence) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                    }
                    if constexpr (kL2SplitSFALoad) {
                        load_l2_activation_scale_factors_hi();
                    }
                    if constexpr (kL2PrefetchWeightSF) {
                        if (k_block_idx + 1 < num_k_blocks)
                            load_l2_weight_scale_factors_for(k_block_idx + 1, prefetched_l2_sf_lo, prefetched_l2_sf_hi);
                    }
                    if constexpr (kL2PrecomputeScale) {
                        asm volatile("mul.rn.f32 %0, %1, %2;"
                                     : "=f"(scale_0_hi_pre)
                                     : "f"(scale_a_0_hi), "f"(l2_sf_lo));
                        asm volatile("mul.rn.f32 %0, %1, %2;"
                                     : "=f"(scale_1_hi_pre)
                                     : "f"(scale_a_1_hi), "f"(l2_sf_lo));
                    }
                    const unsigned long long l2_wgmma_hi_wait_start = phase_profile_clock();
                    ptx::warpgroup_wait<0>();
                    const unsigned long long l2_wgmma_hi_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2WgmmaHi, l2_wgmma_hi_end - l2_wgmma_hi_start);
                    if (epilogue_warp_idx == 0 and lane_idx == 0) {
                        phase_profile_record(kProfileL2WgmmaHiIssue,
                                             l2_wgmma_hi_wait_start - l2_wgmma_hi_start);
                        phase_profile_record(kProfileL2WgmmaHiWait,
                                             l2_wgmma_hi_end - l2_wgmma_hi_wait_start);
                    }

                    const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeEmptyArrive);
                    arrive_empty_barrier(stage_idx);
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterEmptyArrive);
                    const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2EmptyBarrierArrive,
                                             l2_empty_barrier_end - l2_empty_barrier_start);

                    // L2 second half: same SFA half, still choose weight SF by N chunk.
                    const unsigned long long l2_scale_hi_start = phase_profile_clock();
                    if constexpr (WG_BLOCK_N == 128) {
                        const float scale_0_hi = kL2PrecomputeScale ? scale_0_hi_pre : scale_a_0_hi * l2_sf_lo;
                        const float scale_1_hi = kL2PrecomputeScale ? scale_1_hi_pre : scale_a_1_hi * l2_sf_lo;
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_0_hi * accum[i*4+0];
                            final_accum[i*4+1] += scale_0_hi * accum[i*4+1];
                            final_accum[i*4+2] += scale_1_hi * accum[i*4+2];
                            final_accum[i*4+3] += scale_1_hi * accum[i*4+3];
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                            final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                        }
                    }
                    const unsigned long long l2_scale_hi_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ScaleHi, l2_scale_hi_end - l2_scale_hi_start);
                    }
                    }
                    const unsigned long long l2_iter_core_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2IterCore, l2_iter_core_end - l2_iter_core_start);
                }
            }
            };


            const auto run_l2_half_k_pipeline_loop = [&]() {
                if constexpr (kL2HalfKPipeline) {
                    struct L2HalfKScales {
                        float a0_lo, a1_lo, a0_hi, a1_hi, sf_lo, sf_hi;
                    };

                    float accum_next[kAccumPerThread];

                    auto wait_l2_full_stage = [&](const uint32_t& s, const uint32_t& p) -> bool {
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathBeforeFullWait);
                        const unsigned long long full_barrier_start = phase_profile_clock();
                        if (!phase_trace_wait_barrier(full_barriers[s], p, kTraceK2MathFullTimeout))
                            return false;
                        const unsigned long long full_barrier_end = phase_profile_clock();
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathAfterFullWait);
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileFullBarrierWait, full_barrier_end - full_barrier_start);
                        return true;
                    };

                    auto load_l2_scales_for_stage = [&](const uint32_t& s, const uint32_t& k_idx,
                                                        L2HalfKScales& scales) {
                        const unsigned long long l2_scale_factor_load_start = phase_profile_clock();
                        const unsigned long long l2_activation_scale_load_start = phase_profile_clock();
                        scales.a0_lo = ptx::ld_shared(smem_sfa[s] + row_offset_r0);
                        scales.a1_lo = ptx::ld_shared(smem_sfa[s] + row_offset_r1);
                        scales.a0_hi = ptx::ld_shared(smem_sfa[s] + kL2SFAHalfStride + row_offset_r0);
                        scales.a1_hi = ptx::ld_shared(smem_sfa[s] + kL2SFAHalfStride + row_offset_r1);
                        const unsigned long long l2_activation_scale_load_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2ActivationScaleLoad,
                                                 l2_activation_scale_load_end - l2_activation_scale_load_start);
                        constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                        constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                        const unsigned long long l2_weight_scale_load_start = phase_profile_clock();
                        const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + k_idx;
                        const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                        const auto* sf_ptr = base + sf_n * kL2SFKBlocks;
                        scales.sf_lo = kL2WeightSFNC ? ptx::ld_global_nc_f32(sf_ptr) : (kL2WeightSFLDU ? ptx::ldu_global(sf_ptr) : __ldg(sf_ptr));
                        scales.sf_hi = scales.sf_lo;
                        const unsigned long long l2_weight_scale_load_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2WeightScaleLoad,
                                                 l2_weight_scale_load_end - l2_weight_scale_load_start);
                        const unsigned long long l2_scale_factor_load_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2ScaleFactorLoad,
                                                 l2_scale_factor_load_end - l2_scale_factor_load_start);
                    };

                    auto fence_accum_fragment = [&](float* fragment) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(fragment[i]);
                    };

                    auto issue_l2_half_wgmma = [&](const uint32_t& s, const uint32_t& k_half_offset,
                                                   float* fragment) {
                        #pragma unroll
                        for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            const uint32_t k_off = k_half_offset + k * WGMMA::K;
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[s] + row_block_offset * BLOCK_K + k_off, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[s] + wg_n_idx * BLOCK_K + k_off, 1);
                            WGMMA::wgmma(desc_a, desc_b, fragment, static_cast<bool>(k));
                        }
                    };

                    auto scale_l2_fragment = [&](float* fragment, const float& scale_0, const float& scale_1) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_0 * fragment[i*4+0];
                            final_accum[i*4+1] += scale_0 * fragment[i*4+1];
                            final_accum[i*4+2] += scale_1 * fragment[i*4+2];
                            final_accum[i*4+3] += scale_1 * fragment[i*4+3];
                        }
                    };

                    auto scale_l2_lo = [&](float* fragment, const L2HalfKScales& scales) {
                        scale_l2_fragment(fragment, scales.a0_lo * scales.sf_lo, scales.a1_lo * scales.sf_lo);
                    };
                    auto scale_l2_hi = [&](float* fragment, const L2HalfKScales& scales) {
                        scale_l2_fragment(fragment, scales.a0_hi * scales.sf_lo, scales.a1_hi * scales.sf_lo);
                    };

                    auto issue_wait_one_half = [&](const uint32_t& s, const uint32_t& k_half_offset,
                                                   float* fragment,
                                                   const uint32_t& profile_total,
                                                   const uint32_t& profile_issue,
                                                   const uint32_t& profile_wait) {
                        const unsigned long long wgmma_start = phase_profile_clock();
                        fence_accum_fragment(fragment);
                        ptx::warpgroup_arrive();
                        issue_l2_half_wgmma(s, k_half_offset, fragment);
                        ptx::warpgroup_commit_batch();
                        fence_accum_fragment(fragment);
                        const unsigned long long wgmma_wait_start = phase_profile_clock();
                        ptx::warpgroup_wait<0>();
                        const unsigned long long wgmma_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            phase_profile_record(profile_total, wgmma_end - wgmma_start);
                            phase_profile_record(profile_issue, wgmma_wait_start - wgmma_start);
                            phase_profile_record(profile_wait, wgmma_end - wgmma_wait_start);
                        }
                    };

                    auto issue_wait_hi_and_next_lo = [&](const uint32_t& hi_stage, const uint32_t& lo_stage) {
                        const unsigned long long wgmma_start = phase_profile_clock();
                        fence_accum_fragment(accum);
                        fence_accum_fragment(accum_next);
                        ptx::warpgroup_arrive();
                        issue_l2_half_wgmma(hi_stage, BLOCK_K / 2, accum);
                        issue_l2_half_wgmma(lo_stage, 0, accum_next);
                        ptx::warpgroup_commit_batch();
                        fence_accum_fragment(accum);
                        fence_accum_fragment(accum_next);
                        const unsigned long long wgmma_wait_start = phase_profile_clock();
                        ptx::warpgroup_wait<0>();
                        const unsigned long long wgmma_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            phase_profile_record(kProfileL2WgmmaHi, wgmma_end - wgmma_start);
                            phase_profile_record(kProfileL2WgmmaHiIssue, wgmma_wait_start - wgmma_start);
                            phase_profile_record(kProfileL2WgmmaHiWait, wgmma_end - wgmma_wait_start);
                        }
                    };

                    uint32_t k_block_idx = 0;
                    if (!wait_l2_full_stage(stage_idx, phase))
                        return;

                    L2HalfKScales current_scales;
                    load_l2_scales_for_stage(stage_idx, k_block_idx, current_scales);

                    const unsigned long long l2_iter_core_start = phase_profile_clock();
                    issue_wait_one_half(stage_idx, 0, accum,
                                        kProfileL2WgmmaLo, kProfileL2WgmmaLoIssue, kProfileL2WgmmaLoWait);
                    const unsigned long long l2_scale_lo_start = phase_profile_clock();
                    scale_l2_lo(accum, current_scales);
                    const unsigned long long l2_scale_lo_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ScaleLo, l2_scale_lo_end - l2_scale_lo_start);

                    while (k_block_idx + 1 < num_k_blocks) {
                        const uint32_t current_stage = stage_idx;
                        const L2HalfKScales prev_scales = current_scales;

                        advance_pipeline(k_block_idx);
                        if (!wait_l2_full_stage(stage_idx, phase))
                            return;
                        load_l2_scales_for_stage(stage_idx, k_block_idx, current_scales);

                        issue_wait_hi_and_next_lo(current_stage, stage_idx);

                        const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathBeforeEmptyArrive);
                        arrive_empty_barrier(current_stage);
                        if constexpr (MegaMoEPhase::is_linear2_only)
                            phase_trace_mark(kTraceK2MathAfterEmptyArrive);
                        const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2EmptyBarrierArrive,
                                                 l2_empty_barrier_end - l2_empty_barrier_start);

                        const unsigned long long l2_scale_hi_start = phase_profile_clock();
                        scale_l2_hi(accum, prev_scales);
                        const unsigned long long l2_scale_hi_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2ScaleHi, l2_scale_hi_end - l2_scale_hi_start);

                        const unsigned long long next_l2_scale_lo_start = phase_profile_clock();
                        scale_l2_lo(accum_next, current_scales);
                        const unsigned long long next_l2_scale_lo_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL2ScaleLo,
                                                 next_l2_scale_lo_end - next_l2_scale_lo_start);
                    }

                    issue_wait_one_half(stage_idx, BLOCK_K / 2, accum,
                                        kProfileL2WgmmaHi, kProfileL2WgmmaHiIssue, kProfileL2WgmmaHiWait);

                    const unsigned long long l2_empty_barrier_start = phase_profile_clock();
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeEmptyArrive);
                    arrive_empty_barrier(stage_idx);
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterEmptyArrive);
                    const unsigned long long l2_empty_barrier_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2EmptyBarrierArrive,
                                             l2_empty_barrier_end - l2_empty_barrier_start);

                    const unsigned long long l2_scale_hi_start = phase_profile_clock();
                    scale_l2_hi(accum, current_scales);
                    const unsigned long long l2_scale_hi_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2ScaleHi, l2_scale_hi_end - l2_scale_hi_start);

                    advance_pipeline(k_block_idx);
                    const unsigned long long l2_iter_core_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2IterCore, l2_iter_core_end - l2_iter_core_start);
                }
            };

            const auto run_l1_dual_k_gemm_loop = [&]() {
                DG_STATIC_ASSERT((kHidden / BLOCK_K) % 2 == 0, "L1 dual-K expects an even number of K blocks");
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                const uint32_t up_n   = kL1SFGateBlks + gate_n;
                const float* expert_sf_base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;
                const float* gate_sf_base = expert_sf_base + gate_n * kL1SFKBlocks;
                const float* up_sf_base = expert_sf_base + up_n * kL1SFKBlocks;
                float accum_b[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                    const uint32_t stage0 = stage_idx;
                    const uint32_t phase0 = phase;
                    const uint32_t k0 = k_block_idx;
                    full_barriers[stage0]->wait(phase0);

                    const float scale_a0_r0 = ptx::ld_shared(smem_sfa[stage0] + row_offset_r0);
                    const float scale_a0_r1 = ptx::ld_shared(smem_sfa[stage0] + row_offset_r1);
                    const float2 gate_sf_pair = __ldg(reinterpret_cast<const float2*>(gate_sf_base + k0));
                    const float2 up_sf_pair = __ldg(reinterpret_cast<const float2*>(up_sf_base + k0));
                    const float gate_sf0 = gate_sf_pair.x;
                    const float up_sf0   = up_sf_pair.x;

                    advance_pipeline(k_block_idx);
                    const uint32_t stage1 = stage_idx;
                    const uint32_t phase1 = phase;
                    full_barriers[stage1]->wait(phase1);

                    const float scale_a1_r0 = ptx::ld_shared(smem_sfa[stage1] + row_offset_r0);
                    const float scale_a1_r1 = ptx::ld_shared(smem_sfa[stage1] + row_offset_r1);
                    const float gate_sf1 = gate_sf_pair.y;
                    const float up_sf1   = up_sf_pair.y;

                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                        ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_fence_operand(accum_b[i]);
                    }
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage0] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage0] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage1] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage1] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum_b, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                        ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_fence_operand(accum_b[i]);
                    }
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage0);
                    arrive_empty_barrier(stage1);

                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float sb0 = (i & 1u) ? up_sf0 : gate_sf0;
                        const float sb1 = (i & 1u) ? up_sf1 : gate_sf1;
                        final_accum[i*4+0] += scale_a0_r0 * sb0 * accum[i*4+0];
                        final_accum[i*4+1] += scale_a0_r0 * sb0 * accum[i*4+1];
                        final_accum[i*4+2] += scale_a0_r1 * sb0 * accum[i*4+2];
                        final_accum[i*4+3] += scale_a0_r1 * sb0 * accum[i*4+3];
                        final_accum[i*4+0] += scale_a1_r0 * sb1 * accum_b[i*4+0];
                        final_accum[i*4+1] += scale_a1_r0 * sb1 * accum_b[i*4+1];
                        final_accum[i*4+2] += scale_a1_r1 * sb1 * accum_b[i*4+2];
                        final_accum[i*4+3] += scale_a1_r1 * sb1 * accum_b[i*4+3];
                    }

                    advance_pipeline(k_block_idx);
                }
            };

            if constexpr (kL2HalfKPipeline) {
                run_l2_half_k_pipeline_loop();
            } else if constexpr (kL1DualKAccum) {
                if (is_linear1_phase)
                    run_l1_dual_k_gemm_loop();
                else
                    run_default_gemm_loop();
            } else {
                run_default_gemm_loop();
            }
            if constexpr (MegaMoEPhase::is_linear2_only)
                phase_trace_mark(kTraceK2MathAfterDefaultGemmLoop);
            if constexpr (MegaMoEPhase::is_linear2_only && kSplitNWarpgroups && WG_BLOCK_N == 64)
                phase_trace_mark(kTraceK2MathWGAfterDefaultGemmLoopBase + epilogue_wg_idx);

            const unsigned long long block_gemm_end = phase_profile_clock();
            if constexpr (MegaMoEPhase::is_linear2_only)
                phase_trace_mark(kTraceK2MathAfterGemmLoop);
            if (epilogue_warp_idx == 0 and lane_idx == 0)
                phase_profile_record(kProfileGemmCore, block_gemm_end - block_gemm_start);

            // Skip epilogue when block is past valid M (still must release via empty).
            // A dummy cluster peer may still carry an async L1 store from the
            // previous valid block, so drain it before leaving the L1 wave.
            if (row_block_offset >= valid_m) {
                if constexpr (kAsyncL1TMAStore) {
                    if (is_linear1_phase)
                        drain_all_async_l1_stores();
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const unsigned long long block_epilogue_start = phase_profile_clock();
            if constexpr (is_linear1_phase) {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                constexpr uint32_t kNumSFGroups = WG_L1_OUT_BLOCK_N / 64;
                DG_STATIC_ASSERT(WG_L1_OUT_BLOCK_N % 64 == 0, "L1 output SF is per 64 columns");
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

                // Per-row amax, one scale for each 64-col L1 output group.
                float amax_r0[kNumSFGroups] = {};
                float amax_r1[kNumSFGroups] = {};

                // Compute SwiGLU + per-group amax.
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;
                    const uint32_t sf_group = p / 8;

                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                    float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                    float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                    float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                    float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                    float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                    float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                    float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                    auto silu = [](float x) {
                        const float sig = kFastMath ?
                            0.5f * (__tanhf(0.5f * x) + 1.0f) :
                            1.0f / (1.0f + expf(-x));
                        return x * sig;
                    };

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        if constexpr (!kScaleFreeL1K2)
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        if constexpr (!kScaleFreeL1K2)
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }


                float weight_r0 = 0.0f, weight_r1 = 0.0f;
                if constexpr (kNumMaxTokensPerRank <= 1024) {
                    const int topk_weight_src_lane = static_cast<int>(lane_idx - col_idx);
                    if (col_idx == 0) {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .template get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .template get_base_ptr<float>() : 0.0f;
                    }
                    weight_r0 = __shfl_sync(0xffffffff, weight_r0, topk_weight_src_lane);
                    weight_r1 = __shfl_sync(0xffffffff, weight_r1, topk_weight_src_lane);
                } else {
                    weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .template get_base_ptr<float>() : 0.0f;
                    weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
                        .template get_base_ptr<float>() : 0.0f;
                }
                if constexpr (!kScaleFreeL1K2) {
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] *= cute::abs(weight_r0);
                        amax_r1[g] *= cute::abs(weight_r1);
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }
                }

                float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                    if constexpr (kScaleFreeL1K2) {
                        // Keep the existing SFA publication protocol alive in
                        // stage 1, but make its numerical value neutral.
                        sf_r0[g] = 1.0f; sf_inv_r0[g] = weight_r0;
                        sf_r1[g] = 1.0f; sf_inv_r1[g] = weight_r1;
                    } else {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        // The stored scale already contains abs(topk_weight).
                        // Fold the signed weight into its reciprocal once per SF
                        // group instead of multiplying every SwiGLU element first:
                        //   (x * w) / (amax * abs(w)) == x * (w / sf).
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x * weight_r0;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y * weight_r1;
                    }
                }

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                const uint32_t l1_store_stage = kAsyncL1TMAStore ? async_l1_store_stage : 0u;
                if constexpr (kAsyncL1TMAStore)
                    drain_async_l1_store_stage(l1_store_stage);
                auto* smem_cd_l1_wg = smem_cd_l1
                    + l1_store_stage * SMEM_CD_L1_ASYNC_ELEMS
                    + (kSplitNWarpgroups ? 0 : epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N);
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t sf_group = p / 8;
                    const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                    const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                    const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                    const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                    uint16_t r0_bits, r1_bits;
                    if constexpr (kScaleFreeL1K2) {
                        const __nv_fp8x2_e5m2 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e5m2 r1_pair(make_float2(v10, v11));
                        r0_bits = r0_pair.__x;
                        r1_bits = r1_pair.__x;
                    } else {
                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                        r0_bits = r0_pair.__x;
                        r1_bits = r1_pair.__x;
                    }

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_0 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    if (valid_r0)
                        *p0 = r0_bits;
                    if (valid_r1)
                        *p1 = r1_bits;
                }

                // Write L2-activation SF as float, one value per 64 output columns.
                if constexpr (!kDropScaleFreeSFA) {
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                        const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                        const uint32_t base_k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / 64u;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            if (valid_r0)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r0] = sf_r0[g];
                            if (valid_r1)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r1] = sf_r1[g];
                        }
                    }
                }

                // Issue TMA store of the entire tile. Padding rows beyond
                // `valid_m` are written with stale/garbage FP8 to the L1-output
                // pool buffer, but they are never consumed downstream: the L2
                // GEMM tile loads them, but its NVLink-scatter epilogue is
                // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                // padding rows can produce NaN accumulators that simply stay
                // in registers (only valid rows are converted to BF16 and
                // STSM'd into smem). Using TMA for partial tiles is a large
                // win for low-batch / decode where every tile is partial.
                if constexpr (kSplitNWarpgroups) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1_wg,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    if constexpr (kAsyncL1TMAStore) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        constexpr uint32_t kNumReadyBits = L1_OUT_BLOCK_N / 64;
                        const uint64_t ready_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                    << ((n_block_idx * L1_OUT_BLOCK_N) / 64u);
                        if (l1_store_stage == 0) {
                            async_l1_store_pool_block_idx0 = pool_block_idx;
                            async_l1_store_ready_mask0 = ready_mask;
                            async_l1_store_pending0 = true;
                        } else {
                            async_l1_store_pool_block_idx1 = pool_block_idx;
                            async_l1_store_ready_mask1 = ready_mask;
                            async_l1_store_pending1 = true;
                        }
                        async_l1_store_stage ^= 1u;
                    } else {
                        const unsigned long long tma_wait_start = phase_profile_clock();
                        ptx::tma_store_wait<0>();
                        const unsigned long long tma_wait_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL1TMAStoreWait, tma_wait_end - tma_wait_start);
                        constexpr uint32_t kNumReadyBits = L1_OUT_BLOCK_N / 64;
                        const uint64_t ready_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                   << ((n_block_idx * L1_OUT_BLOCK_N) / 64u);
                        publish_l1_ready_mask(pool_block_idx, ready_mask);
                    }
                } else {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1_wg,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    if constexpr (kAsyncL1TMAStore) {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        constexpr uint32_t kNumReadyBits = L1_OUT_BLOCK_N / 64;
                        const uint64_t ready_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                    << ((n_block_idx * L1_OUT_BLOCK_N) / 64u);
                        if (l1_store_stage == 0) {
                            async_l1_store_pool_block_idx0 = pool_block_idx;
                            async_l1_store_ready_mask0 = ready_mask;
                            async_l1_store_pending0 = true;
                        } else {
                            async_l1_store_pool_block_idx1 = pool_block_idx;
                            async_l1_store_ready_mask1 = ready_mask;
                            async_l1_store_pending1 = true;
                        }
                        async_l1_store_stage ^= 1u;
                    } else {
                        const unsigned long long tma_wait_start = phase_profile_clock();
                        ptx::tma_store_wait<0>();
                        const unsigned long long tma_wait_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL1TMAStoreWait, tma_wait_end - tma_wait_start);
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        constexpr uint32_t kNumReadyBits = L1_OUT_BLOCK_N / 64;
                        const uint64_t ready_mask = ((uint64_t(1) << kNumReadyBits) - 1)
                                                   << ((n_block_idx * L1_OUT_BLOCK_N) / 64u);
                        publish_l1_ready_mask(pool_block_idx, ready_mask);
                    }
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL1Epilogue, block_epilogue_end - block_epilogue_start);
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kFP8CompressedCombine) {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 128 && WG_BLOCK_M == 64,
                                     "Compressed combine expects M64N128 per warpgroup");
                    auto* compressed_smem = reinterpret_cast<uint8_t*>(smem_cd_l2);
                    auto* compressed_wg_smem = compressed_smem +
                        epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N;

                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const __nv_fp8x2_e5m2 r0_pair(make_float2(
                            final_accum[i*4+0], final_accum[i*4+1]));
                        const __nv_fp8x2_e5m2 r1_pair(make_float2(
                            final_accum[i*4+2], final_accum[i*4+3]));
                        const uint32_t col = i * 8 + col_idx * 2;
                        if (valid_r0)
                            *reinterpret_cast<uint16_t*>(
                                compressed_wg_smem + r_0 * WG_BLOCK_N + col) = r0_pair.__x;
                        if (valid_r1)
                            *reinterpret_cast<uint16_t*>(
                                compressed_wg_smem + r_1 * WG_BLOCK_N + col) = r1_pair.__x;
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    const uint32_t row_in_warp_block = lane_idx / 16;
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kCompressedColsPerLane = WG_BLOCK_N / 16;
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg =
                            warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata =
                            *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx();
                        const uint32_t dst_token_idx = src_metadata.token_idx();
                        const uint32_t dst_topk_idx = src_metadata.topk_idx();
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                        const auto packed = *reinterpret_cast<uint2*>(
                            compressed_wg_smem + row_in_wg * WG_BLOCK_N +
                            lane_in_row * kCompressedColsPerLane);
                        auto dst_data = math::advance_ptr<uint2>(
                            dst_token.get_base_ptr(),
                            n_idx + lane_in_row * sizeof(uint2));
                        *sym_buffer.map(dst_data, dst_rank_idx) = packed;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else if constexpr (kDirectL2Scatter) {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128,
                                     "Direct L2 scatter prototype only supports N64/N128");

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row,
                                                  const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            uint32_t dst_rank_idx = 0, dst_token_idx = 0, dst_topk_idx = 0;
                            const uint32_t row_group_base = lane_idx - col_idx;
                            if (col_idx == 0) {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
                                dst_rank_idx = src_metadata.rank_idx();
                                dst_token_idx = src_metadata.token_idx();
                                dst_topk_idx = src_metadata.topk_idx();
                            }
                            const uint32_t row_group_mask = 0xfu << row_group_base;
                            const int src_lane = static_cast<int>(row_group_base);
                            dst_rank_idx = __shfl_sync(row_group_mask, dst_rank_idx, src_lane);
                            dst_token_idx = __shfl_sync(row_group_mask, dst_token_idx, src_lane);
                            dst_topk_idx = __shfl_sync(row_group_mask, dst_topk_idx, src_lane);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                const uint32_t packed_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                } else {
                    // STSM into smem_cd_l2 (BF16). Reuse SM100 column-swizzle layout.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                        // Each i consumes 8 floats (one 16x256b chunk in SM100 terms).
                        // For SM90 WGMMA layout, 8 floats per i correspond to 2 chunks of 4 floats:
                        //   final_accum[i*8 + (0..3)] = chunk 2i: (r0c0, r0c1, r1c0, r1c1)
                        //   final_accum[i*8 + (4..7)] = chunk 2i+1: same shape
                        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                        // Write to SMEM at appropriate position
                        // Row r_0 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r0_lo
                        // Row r_0 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r0_hi
                        // Row r_1 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r1_lo
                        // Row r_1 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r1_hi
                        auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                            auto smem_ptr = smem_cd_l2
                                + epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N
                                + row * WG_BLOCK_N
                                + col;
                            // BF16 STS: 2 bf16 elements
                            *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                        };
                        if (valid_r0) {
                            const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 0], final_accum[chunk_lo*4 + 1]);
                            const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 0], final_accum[chunk_hi*4 + 1]);
                            write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                            write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                        }
                        if (valid_r1) {
                            const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 2], final_accum[chunk_lo*4 + 3]);
                            const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 2], final_accum[chunk_hi*4 + 3]);
                            write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                            write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                        }
                    }

                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeScatterSync);
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterScatterSync);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    const uint32_t cols_per_lane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N == 64 or WG_BLOCK_N == 128 or WG_BLOCK_N == 256,
                                  "L2 scatter supports per-WG N64/N128/N256");

                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeRemoteScatter);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx();
                        const uint32_t dst_token_idx = src_metadata.token_idx();
                        const uint32_t dst_topk_idx = src_metadata.topk_idx();

                        auto smem_ptr = smem_cd_l2
                            + epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N
                            + row_in_wg * WG_BLOCK_N
                            + lane_in_row * cols_per_lane;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);

                        if constexpr (WG_BLOCK_N == 256) {
                            const auto packed0 = *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed1 = *(reinterpret_cast<uint4*>(smem_ptr) + 1);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * 2u * sizeof(uint4));
                            auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);
                            mapped_dst_ptr[0] = packed0;
                            mapped_dst_ptr[1] = packed1;
                        } else if constexpr (WG_BLOCK_N == 128) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterRemoteScatter);

                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathBeforeFinalEpiSync);
                    if constexpr (MegaMoEPhase::is_linear2_only && !MegaMoEPhase::needs_combine_reduce &&
                                  kSplitNWarpgroups && WG_BLOCK_N == 64) {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                    if constexpr (MegaMoEPhase::is_linear2_only)
                        phase_trace_mark(kTraceK2MathAfterFinalEpiSync);
                    if constexpr (MegaMoEPhase::is_linear2_only && kSplitNWarpgroups && WG_BLOCK_N == 64)
                        phase_trace_mark(kTraceK2MathWGAfterFinalEpiSyncBase + epilogue_wg_idx);
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if constexpr (MegaMoEPhase::is_linear2_only)
                    phase_trace_mark(kTraceK2MathAfterBlock);
                if constexpr (MegaMoEPhase::is_linear2_only && kSplitNWarpgroups && WG_BLOCK_N == 64)
                    phase_trace_mark(kTraceK2MathWGAfterBlockBase + epilogue_wg_idx);
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL2Epilogue, block_epilogue_end - block_epilogue_start);
            }
        });
        if constexpr (MegaMoEPhase::is_linear2_only)
            phase_trace_mark(kTraceK2MathLoopEnd);
        if constexpr (MegaMoEPhase::is_linear2_only && kSplitNWarpgroups && WG_BLOCK_N == 64)
            phase_trace_mark(kTraceK2MathWGLoopEndBase + epilogue_wg_idx);
        const unsigned long long math_loop_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0) {
            phase_profile_record(kProfileMathLoop, math_loop_end - math_loop_start);
            if constexpr (MegaMoEPhase::runs_linear1) {
                phase_profile_record(kProfileL1MathLoop, math_loop_end - math_loop_start);
            } else if constexpr (MegaMoEPhase::runs_linear2) {
                phase_profile_record(kProfileL2MathLoop, math_loop_end - math_loop_start);
            }
        }

        if constexpr (!kNeedsCombineBarrier) {
            if constexpr (kAsyncL1TMAStore)
                drain_all_async_l1_stores();
            return;
        }

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        const unsigned long long combine_barrier_start = phase_profile_clock();
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); },
            true, MegaMoEPhase::needs_combine_reduce ? true : false
        );
        const unsigned long long combine_barrier_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineBarrier, combine_barrier_end - combine_barrier_start);

        if constexpr (kNumDispatchWarps > 0 && MegaMoEPhase::needs_combine_reduce) {
            // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
            // dispatch may now safely clean workspace state.
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }
        if constexpr (!MegaMoEPhase::needs_combine_reduce)
            return;
        const unsigned long long combine_reduce_start = phase_profile_clock();

        constexpr uint32_t kNumHiddenBytes = kCombineHiddenBytes;
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = kCombineChunkSlots;
        constexpr uint32_t kNumChunks = kCombineNumChunks;
        constexpr uint32_t kNumChunkBytes = kCombineChunkBytes;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        ptx::tma_store_wait<0>();
        __syncwarp();
        const unsigned long long combine_reduce_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineReduce, combine_reduce_end - combine_reduce_start);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}


template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kPhaseProfileRequested = false,
    bool kDirectGlobalReduceRequested = false,
    uint32_t kNumThreads = 256,
    uint32_t kNumWarps = kNumThreads / 32,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_combine_reduce_impl(
    void* y,
    int* cumulative_local_expert_recv_stats,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());
    const auto l1_token_buffer           = layout::Buffer(fp8_token_layout, 1, workspace.num_max_pool_tokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer              = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer    = layout::Buffer(l1_topk_weights_layout, 1, workspace.num_max_pool_tokens, l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer           = layout::Buffer(fp8_intermediate_token_layout, 1, workspace.num_max_pool_tokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer              = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer      = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    constexpr uint32_t kSharedMemoryAlignment = 1024;
    constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kNumHiddenVec = kNumHiddenBytes / sizeof(uint4);
    constexpr uint32_t kNumElemsPerVec = sizeof(uint4) / sizeof(nv_bfloat162);
    constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
    constexpr uint32_t kNumChunkSlots = 3;
    constexpr uint32_t kSm90DynamicSmemBudget = 228u * 1024u;
    constexpr uint32_t kNumChunks =
        (kNumChunkSlots * kNumWarps * kNumHiddenBytes <= kSm90DynamicSmemBudget and kHidden <= 4096) ? 1 : 2;
    constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
    constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
    constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
#if defined(DG_SM90_MOE_FP8_COMPRESSED_COMBINE)
    constexpr bool kFP8CompressedCombineRequested = true;
#else
    constexpr bool kFP8CompressedCombineRequested = false;
#endif
    constexpr bool kFP8CompressedCombine =
        kFP8CompressedCombineRequested &&
        (kNumThreads == 256 || kNumThreads == 384) && kNumRanks == 8 &&
        kNumMaxTokensPerRank >= 7000 && kHidden == 3072 &&
        kIntermediateHidden == 1536 && kNumExperts == 128 && kNumTopk == 16;
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Combine-reduce thread count must be warp-aligned");
    DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by combine chunks");
    DG_STATIC_ASSERT(kNumHiddenBytes % sizeof(uint4) == 0,
                     "Hidden must be divisible by one 128-bit vector of BF16");
    DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned");
    DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of a warp");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

    constexpr uint32_t kCombineReduceGridSyncIndex = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;
    constexpr uint32_t kProfileCombineReduce = 4;
    constexpr uint32_t kProfileWorkspaceCleanup = 10;
    constexpr uint32_t kNumProfileMetrics = 31;
    const auto phase_profile_clock = [&]() -> unsigned long long {
        if constexpr (kPhaseProfileRequested) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            return t;
        } else {
            return 0ull;
        }
    };
    const auto phase_profile_record = [&](const uint32_t& metric, const unsigned long long& cycles) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and cycles > 0) {
                constexpr uint32_t kProfileIntOffset = (kNumExpertsPerRank + 1u) & ~1u;
                auto profile = reinterpret_cast<unsigned long long*>(
                    cumulative_local_expert_recv_stats + kProfileIntOffset);
                atomicAdd(profile + metric, cycles);
                atomicMax(profile + kNumProfileMetrics + metric, cycles);
                atomicAdd(profile + 2 * kNumProfileMetrics + metric, 1ull);
            }
        }
    };

    const unsigned long long combine_reduce_start = phase_profile_clock();
    if constexpr (kFP8CompressedCombine) {
        using Barrier = cutlass::arch::ClusterTransactionBarrier;
        constexpr uint32_t kCompressedInputBytes = kHidden;
        constexpr uint32_t kCompressedInputVec = kHidden / sizeof(uint4);
        constexpr uint32_t kCompressedInputVecPerLane = kCompressedInputVec / 32;
        constexpr uint32_t kCompressedValuesPerVec = sizeof(uint4) / sizeof(__nv_fp8_e5m2);
        constexpr uint32_t kLoadBytes = 2 * kNumWarps * kCompressedInputBytes;
        constexpr uint32_t kStoreOffset =
            math::constexpr_align<uint32_t>(kLoadBytes, kSharedMemoryAlignment);
        constexpr uint32_t kStoreBytes = kNumWarps * kNumHiddenBytes;
        constexpr uint32_t kBarrierOffset =
            math::constexpr_align<uint32_t>(kStoreOffset + kStoreBytes,
                                            kSharedMemoryAlignment);
        extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
        auto* barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kBarrierOffset);

        if (warp_idx == 0) {
            for (uint32_t i = thread_idx; i < kNumWarps * 2; i += kNumThreads)
                barrier_start_ptr[i].init(1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        const auto load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return smem_buffer + (warp_idx + i * kNumWarps) * kCompressedInputBytes;
        });
        auto* store_buffer = reinterpret_cast<uint4*>(
            smem_buffer + kStoreOffset + warp_idx * kNumHiddenBytes);
        auto load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return barrier_start_ptr + i + warp_idx * 2;
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumWarps + warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                       token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask =
                __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = combine_token_buffer.get_rank_buffer(slot_idx)
                                                 .get_data_buffer(token_idx)
                                                 .template get_base_ptr<uint8_t>();
                        ptx::tma_load_1d(
                            load_buffer[i], src_ptr, load_barriers[i],
                            kCompressedInputBytes);
                        ptx::mbarrier_arrive_and_set_tx(
                            load_barriers[i], kCompressedInputBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            bool do_reduce = move_mask_and_load(load_stage_idx);
            float reduced[kCompressedInputVecPerLane * kCompressedValuesPerVec] = {};
            while (do_reduce) {
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                load_barriers[load_stage_idx]->wait(combine_phase);
                auto* stage_vec = reinterpret_cast<uint4*>(load_buffer[load_stage_idx]);
                #pragma unroll
                for (uint32_t j = 0; j < kCompressedInputVecPerLane; ++ j) {
                    const uint32_t vec_idx = j * 32 + lane_idx;
                    const auto packed = stage_vec[vec_idx];
                    const auto* fp8_values =
                        reinterpret_cast<const __nv_fp8_e5m2*>(&packed);
                    #pragma unroll
                    for (uint32_t l = 0; l < kCompressedValuesPerVec; ++ l)
                        reduced[j * kCompressedValuesPerVec + l] +=
                            static_cast<float>(fp8_values[l]);
                }
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            #pragma unroll
            for (uint32_t j = 0; j < kCompressedInputVecPerLane; ++ j) {
                uint4 casted_lo, casted_hi;
                auto* bf16_lo = reinterpret_cast<nv_bfloat162*>(&casted_lo);
                auto* bf16_hi = reinterpret_cast<nv_bfloat162*>(&casted_hi);
                #pragma unroll
                for (uint32_t l = 0; l < 4; ++ l) {
                    bf16_lo[l] = __float22bfloat162_rn(make_float2(
                        reduced[j * kCompressedValuesPerVec + l * 2],
                        reduced[j * kCompressedValuesPerVec + l * 2 + 1]));
                    bf16_hi[l] = __float22bfloat162_rn(make_float2(
                        reduced[j * kCompressedValuesPerVec + 8 + l * 2],
                        reduced[j * kCompressedValuesPerVec + 8 + l * 2 + 1]));
                }
                const uint32_t out_vec_idx = (j * 32 + lane_idx) * 2;
                ptx::st_shared(store_buffer + out_vec_idx,
                               casted_lo.x, casted_lo.y, casted_lo.z, casted_lo.w);
                ptx::st_shared(store_buffer + out_vec_idx + 1,
                               casted_hi.x, casted_hi.y, casted_hi.z, casted_hi.w);
            }
            __syncwarp();
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes),
                    store_buffer, kNumHiddenBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
        ptx::tma_store_wait<0>();
        __syncwarp();
    } else if constexpr (kNumRanks == 1) {
        __shared__ int32_t topk_idx_smem[kNumTopk];
        for (uint32_t token_idx = sm_idx; token_idx < num_tokens; token_idx += kNumSMs) {
            const auto topk_idx_ptr = input_topk_idx_buffer.get_data_buffer(token_idx)
                                          .get_base_ptr<const int64_t>();
            if (thread_idx < kNumTopk)
                topk_idx_smem[thread_idx] = static_cast<int32_t>(topk_idx_ptr[thread_idx]);
            __syncthreads();

            for (uint32_t vec_idx = thread_idx; vec_idx < kNumHiddenVec; vec_idx += kNumThreads) {
                float2 reduced[kNumElemsPerVec];
                #pragma unroll
                for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
                    reduced[elem_idx] = make_float2(0.0f, 0.0f);

                #pragma unroll
                for (uint32_t topk_slot_idx = 0; topk_slot_idx < kNumTopk; ++topk_slot_idx) {
                    const auto valid_topk = static_cast<int>(topk_idx_smem[topk_slot_idx]);
                    const auto src_ptr = combine_token_buffer.get_rank_buffer(topk_slot_idx)
                                             .get_data_buffer(token_idx)
                                             .get_base_ptr<const uint4>();
                    const auto vec_values = ptx::ld_gez_pred(src_ptr + vec_idx, valid_topk);
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&vec_values);
                    #pragma unroll
                    for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
                        ptx::accumulate(reduced[elem_idx], bf16_values[elem_idx]);
                }

                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
                    casted_bf16[elem_idx] = __float22bfloat162_rn(reduced[elem_idx]);

                const auto dst_ptr = math::advance_ptr<uint4>(
                    y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes);
                ptx::st_global_v4_u32(dst_ptr + vec_idx, casted);
            }
            __syncthreads();
        }
    } else if constexpr (kDirectGlobalReduceRequested) {
        for (uint32_t token_idx = sm_idx * kNumWarps + warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
            const auto dst_ptr = math::advance_ptr<uint4>(
                y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;
                const uint32_t chunk_vec_offset = chunk * kNumChunkUint4;
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};

                uint32_t mask = total_mask;
                while (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    const auto src_ptr = math::advance_ptr<uint4>(
                        combine_token_buffer.get_rank_buffer(slot_idx)
                                            .get_data_buffer(token_idx).get_base_ptr(),
                        chunk_byte_offset);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = ptx::ld_global_l2_256b(src_ptr + j * 32 + lane_idx);
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                    ptx::st_global_v4_u32(dst_ptr + chunk_vec_offset + j * 32 + lane_idx, casted);
                }
            }
        }
    } else {
        using Barrier = cutlass::arch::ClusterTransactionBarrier;
        extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
        constexpr uint32_t kSmemBufferBytes = kNumChunkSlots * kNumWarps * kNumChunkBytes;
        constexpr uint32_t kSmemBarrierOffset = math::constexpr_align<uint32_t>(kSmemBufferBytes, kSharedMemoryAlignment);
        auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kSmemBarrierOffset);

        if (warp_idx == 0) {
            for (uint32_t i = thread_idx; i < kNumWarps * 2; i += kNumThreads)
                barrier_start_ptr[i].init(1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (warp_idx + i * kNumWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (warp_idx + kNumWarps * 2) * kNumChunkBytes);
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return barrier_start_ptr + i + warp_idx * 2;
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumWarps + warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        ptx::tma_store_wait<0>();
        __syncwarp();
    }
    const unsigned long long combine_reduce_end = phase_profile_clock();
    if (warp_idx == 0 and lane_idx == 0)
        phase_profile_record(kProfileCombineReduce, combine_reduce_end - combine_reduce_start);

    const unsigned long long cleanup_start = phase_profile_clock();
    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < kNumExperts; i += kNumSMs * kNumThreads)
        *workspace.get_expert_send_count_ptr(i) = 0;
    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < kNumExpertsPerRank; i += kNumSMs * kNumThreads) {
        const auto recv_status = *workspace.get_expert_recv_count_sum_ptr(i);
        const auto num_recv_tokens = static_cast<uint32_t>(recv_status);
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
        if (cumulative_local_expert_recv_stats != nullptr and num_recv_tokens > 0)
            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
    }
    const uint32_t num_l1_arrival_slots = math::align(workspace.num_max_pool_blocks, 2u);
    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < num_l1_arrival_slots; i += kNumSMs * kNumThreads)
        *workspace.get_l1_arrival_count_ptr(i) = 0;
    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < workspace.num_max_pool_blocks; i += kNumSMs * kNumThreads)
        *workspace.get_l2_arrival_mask_ptr(i) = 0;
    const unsigned long long cleanup_end = phase_profile_clock();
    if (thread_idx == 0)
        phase_profile_record(kProfileWorkspaceCleanup, cleanup_end - cleanup_start);

    if constexpr (kNumRanks > 1) {
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads,
                             kCombineReduceGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            []() { __syncthreads(); },
            true, false);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_l1_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    sm90_fp8_mega_moe_core<DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS(MegaMoELinear1Phase)>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_l2_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    sm90_fp8_mega_moe_core<DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS(MegaMoELinear2Phase)>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_l2_no_reduce_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    sm90_fp8_mega_moe_core<DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS(MegaMoELinear2NoReducePhase)>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

#undef DG_SM90_FP8_MOE_TEMPLATE_PARAMS
#undef DG_SM90_FP8_MOE_KERNEL_ARGS_DECL
#undef DG_SM90_FP8_MOE_CORE_ARGS_DECL
#undef DG_SM90_FP8_MOE_KERNEL_ARGS
#undef DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS

} // namespace deep_gemm

#pragma clang diagnostic pop
