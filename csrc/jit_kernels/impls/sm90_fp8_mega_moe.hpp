#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh` and uses the same
// dispatch/combine contract with an SM90 FP8 TMA/WGMMA implementation.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are per-128-channel float (not UE8M0
//     int + per-32 UTCCP layout).
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90FP8MegaMoERuntime final : public LaunchRuntime<SM90FP8MegaMoERuntime> {
public:
    enum class KernelPhase {
        Linear1,
        Linear2,
        Linear2NoReduce
    };

    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool direct_l2_scatter;
        bool phase_profile;
        bool l2_nmajor_schedule;
        bool one_warp_cleanup;
        int dispatch_num_sms;
        KernelPhase kernel_phase;
        bool scale_free_l1_k2;
        bool drop_scale_free_sfa;
        bool fp8_compressed_combine;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and weights. Weight scale factors use
        // block (128, 128) quantization and are loaded by the math warpgroup
        // directly from global memory (no TMA descriptor required).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const float* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args = LaunchArgs(1, 1);
    };

    static std::string generate_impl(const Args& args) {
        const char* kernel_symbol = args.kernel_phase == KernelPhase::Linear1 ? "sm90_fp8_mega_moe_l1_impl" :
            (args.kernel_phase == KernelPhase::Linear2 ? "sm90_fp8_mega_moe_l2_impl" :
             "sm90_fp8_mega_moe_l2_no_reduce_impl");
        const bool scale_free_l1_k2 =
            args.scale_free_l1_k2 &&
            (args.kernel_phase == KernelPhase::Linear1 ||
             args.kernel_phase == KernelPhase::Linear2NoReduce) &&
            args.num_ranks == 8 && args.num_max_tokens_per_rank >= 7000 &&
            args.num_tokens == 7000 && args.hidden == 3072 &&
            args.intermediate_hidden == 1536 && args.num_experts == 128 &&
            args.num_topk == 16 && args.config.block_m == 192 &&
            args.config.block_n == 128 && args.config.block_k == 128;
        const bool drop_scale_free_sfa = scale_free_l1_k2 &&
            args.drop_scale_free_sfa;
        std::string feature_defines;
        if (scale_free_l1_k2)
            feature_defines += "#define DG_SM90_MOE_SCALE_FREE_L1_K2 1\n";
        if (drop_scale_free_sfa)
            feature_defines += "#define DG_SM90_MOE_SCALE_FREE_DROP_SFA 1\n";
        if (args.fp8_compressed_combine)
            feature_defines += "#define DG_SM90_MOE_FP8_COMPRESSED_COMBINE 1\n";
        return fmt::format(R"(
{}
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

// JIT cache salt: current SM90 MegaMoE three-stage green-context core.
static constexpr int kSm90MegaMoECoreJitVersion = 51;

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
    feature_defines,
    kernel_symbol,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.sf_pool_stride_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.cluster_size,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.direct_l2_scatter ? "true" : "false",
    args.phase_profile ? "true" : "false",
    args.l2_nmajor_schedule ? "true" : "false",
    args.one_warp_cleanup ? "true" : "false",
    args.dispatch_num_sms,
    args.config.split_sfa_tma ? "true" : "false",
    args.config.l2_dual_accum ? "true" : "false",
    args.config.l1_dual_k_accum ? "true" : "false",
    args.config.l2_sfa_2row_tma ? "true" : "false",
    args.config.l2_precompute_scale ? "true" : "false",
    args.config.l2_ratio_accum ? "true" : "false",
    args.config.l2_weight_sf_ldu ? "true" : "false",
    args.config.l2_prefetch_weight_sf ? "true" : "false",
    args.config.l2_fast_weight_sf_n ? "true" : "false",
    args.config.l2_hoist_weight_sf_base ? "true" : "false",
    args.config.l2_delay_sfa_load ? "true" : "false",
    args.config.l2_split_sfa_load ? "true" : "false",
    args.config.l2_skip_post_wgmma_fence ? "true" : "false",
    args.config.l2_first_accum_assign ? "true" : "false",
    args.config.l2_n64_interleave ? "true" : "false",
    args.config.l2_half_k_pipeline ? "true" : "false",
    args.config.async_l1_tma_store ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_weights_sf
        ));
    }
};


class SM90FP8MegaMoECombineReduceRuntime final : public LaunchRuntime<SM90FP8MegaMoECombineReduceRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int num_padded_sf_pool_tokens;
        bool phase_profile;
        bool direct_global_reduce;
        bool fp8_compressed_combine;

        void* y;
        int* cumulative_local_expert_recv_stats;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        LaunchArgs launch_args = LaunchArgs(1, 1);
    };

    static std::string generate_impl(const Args& args) {
        const char* feature_define = args.fp8_compressed_combine ?
            "#define DG_SM90_MOE_FP8_COMPRESSED_COMBINE 1" : "";
        return fmt::format(R"(
{}
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

// JIT cache salt: single-rank and opt-in multi-rank K3 can use token/vector global reduce.
// Keep this in the generated code so header-only K3 changes invalidate the cache.
static constexpr int kSm90MegaMoECombineReduceJitVersion = 8;

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm90_fp8_mega_moe_combine_reduce_impl<
            {},
            {}, {},
            {}, {},
            {},
            {}, {},
            {},
            {},
            {}
        >);
}};
)",
    feature_define,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.num_padded_sf_pool_tokens,
    args.launch_args.grid_dim.first, args.num_ranks,
    args.phase_profile ? "true" : "false",
    args.direct_global_reduce ? "true" : "false",
    args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs
        ));
    }
};

static int get_sm90_moe_combine_reduce_smem_size(const int& hidden, const int& num_threads) {
    constexpr int kSharedMemoryAlignment = 1024;
    constexpr int kNumChunkSlots = 3;
    constexpr int kSm90DynamicSmemBudget = 228 * 1024;
    const int num_warps = num_threads / 32;
    const int hidden_bytes = hidden * static_cast<int>(sizeof(nv_bfloat16));
    const int num_chunks = (kNumChunkSlots * num_warps * hidden_bytes <= kSm90DynamicSmemBudget and hidden <= 4096) ? 1 : 2;
    DG_HOST_ASSERT(hidden % num_chunks == 0);
    const int chunk_bytes = hidden_bytes / num_chunks;
    DG_HOST_ASSERT(chunk_bytes % 16 == 0);
    DG_HOST_ASSERT((chunk_bytes / static_cast<int>(sizeof(uint4))) % 32 == 0);
    const int buffer_bytes = kNumChunkSlots * num_warps * chunk_bytes;
    const int barrier_bytes = num_warps * 2 * static_cast<int>(sizeof(uint64_t));
    return align(buffer_bytes, kSharedMemoryAlignment) + barrier_bytes;
}


#if CUDART_VERSION >= 13020

class SM90FP8MegaMoEGreenCtxGraph final {
private:
    struct BufferViews {
        torch::Tensor l1_acts;
        torch::Tensor l1_acts_sf;
        torch::Tensor l2_acts;
        torch::Tensor l2_acts_sf;
    };

    torch::Tensor y_;
    torch::Tensor sym_buffer_;
    torch::Tensor l1_weights_;
    torch::Tensor l1_weights_sf_;
    torch::Tensor l2_weights_;
    torch::Tensor l2_weights_sf_;
    torch::Tensor stats_;
    std::vector<int64_t> sym_buffer_ptrs_;

    uint32_t rank_idx_ = 0;
    uint32_t num_max_tokens_per_rank_ = 0;
    uint32_t num_experts_ = 0;
    uint32_t num_experts_per_rank_ = 0;
    uint32_t num_topk_ = 0;
    uint32_t num_tokens_ = 0;
    uint32_t hidden_ = 0;
    uint32_t intermediate_hidden_ = 0;
    uint32_t num_padded_sf_pool_tokens_ = 0;
    uint32_t kernel1_sms_ = 0;
    uint32_t kernel2_sms_ = 0;
    uint32_t reduce_sms_ = 0;
    float activation_clamp_ = 0.0f;
    bool fast_math_ = true;
    bool fp8_compressed_combine_requested_ = false;
    MegaMoESM90Config config_;

    int device_idx_ = 0;
    cudaExecutionContext_t primary_context_ = nullptr;
    std::array<cudaExecutionContext_t, 2> green_contexts_ = {nullptr, nullptr};
    std::array<unsigned long long, 2> green_context_ids_ = {0, 0};
    cudaStream_t last_launch_stream_ = nullptr;
    bool graph_launched_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraphNode_t kernel1_node_ = nullptr;
    cudaGraphNode_t kernel2_node_ = nullptr;
    cudaGraphNode_t reduce_node_ = nullptr;
    std::shared_ptr<KernelRuntime> kernel1_runtime_;
    std::shared_ptr<KernelRuntime> kernel2_runtime_;
    std::shared_ptr<KernelRuntime> reduce_runtime_;
    KernelHandle kernel1_graph_kernel_{};
    KernelHandle kernel2_graph_kernel_{};
    KernelHandle reduce_graph_kernel_{};

    SM90FP8MegaMoERuntime::Args kernel1_args_;
    SM90FP8MegaMoERuntime::Args kernel2_args_;
    SM90FP8MegaMoECombineReduceRuntime::Args reduce_args_;
    std::array<void*, 13> kernel1_params_{};
    std::array<void*, 13> kernel2_params_{};
    std::array<void*, 4> reduce_params_{};

    static uint32_t checked_positive_u32(const int& value) {
        DG_HOST_ASSERT(value > 0);
        return static_cast<uint32_t>(value);
    }

    static uint32_t checked_nonnegative_u32(const int& value) {
        DG_HOST_ASSERT(value >= 0);
        return static_cast<uint32_t>(value);
    }

    uint32_t num_ranks() const {
        return static_cast<uint32_t>(sym_buffer_ptrs_.size());
    }

    size_t get_required_sym_buffer_bytes() const {
        const auto workspace = layout::Workspace(
            nullptr, num_ranks(), num_experts_, num_max_tokens_per_rank_, num_topk_);
        const auto fp8_token_layout = layout::Data(hidden_);
        const auto bf16_token_layout = layout::Data(hidden_ * sizeof(nv_bfloat16));
        const auto fp8_sf_layout = layout::Data(hidden_ / 32);
        const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden_);
        const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden_ / 16);
        const auto input_topk_idx_layout = layout::Data(num_topk_ * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk_ * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        const auto input_token_buffer = layout::Buffer(fp8_token_layout, 1, num_max_tokens_per_rank_, workspace.get_end_ptr());
        const auto input_sf_buffer = layout::Buffer(fp8_sf_layout, 1, num_max_tokens_per_rank_, input_token_buffer.get_end_ptr());
        const auto input_topk_idx_buffer = layout::Buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank_, input_sf_buffer.get_end_ptr());
        const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank_, input_topk_idx_buffer.get_end_ptr());
        const auto l1_token_buffer = layout::Buffer(fp8_token_layout, 1, workspace.num_max_pool_tokens, input_topk_weights_buffer.get_end_ptr());
        const auto l1_sf_buffer = layout::Buffer(fp8_sf_layout, 1, num_padded_sf_pool_tokens_, l1_token_buffer.get_end_ptr());
        const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, workspace.num_max_pool_tokens, l1_sf_buffer.get_end_ptr());
        const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, workspace.num_max_pool_tokens, l1_topk_weights_buffer.get_end_ptr());
        const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, num_padded_sf_pool_tokens_, l2_token_buffer.get_end_ptr());
        const auto combine_token_buffer = layout::Buffer(
            bf16_token_layout, num_topk_, num_max_tokens_per_rank_, l2_sf_buffer.get_end_ptr());
        return static_cast<size_t>(reinterpret_cast<uintptr_t>(combine_token_buffer.get_end_ptr()));
    }

    BufferViews slice_buffer(const torch::Tensor& buffer) const {
        const auto workspace = layout::Workspace(
            nullptr, num_ranks(), num_experts_, num_max_tokens_per_rank_, num_topk_);
        constexpr int kGranK = 128;
        const auto fp8_token_layout = layout::Data(hidden_);
        const auto fp8_sf_layout = layout::Data(hidden_ / 32);
        const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden_);
        const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden_ / 16);
        const auto input_topk_idx_layout = layout::Data(num_topk_ * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk_ * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        const auto input_token_buffer = layout::Buffer(fp8_token_layout, 1, num_max_tokens_per_rank_, workspace.get_end_ptr());
        const auto input_sf_buffer = layout::Buffer(fp8_sf_layout, 1, num_max_tokens_per_rank_, input_token_buffer.get_end_ptr());
        const auto input_topk_idx_buffer = layout::Buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank_, input_sf_buffer.get_end_ptr());
        const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank_, input_topk_idx_buffer.get_end_ptr());
        const auto l1_token_buffer = layout::Buffer(fp8_token_layout, 1, workspace.num_max_pool_tokens, input_topk_weights_buffer.get_end_ptr());
        const auto l1_sf_buffer = layout::Buffer(fp8_sf_layout, 1, num_padded_sf_pool_tokens_, l1_token_buffer.get_end_ptr());
        const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, workspace.num_max_pool_tokens, l1_sf_buffer.get_end_ptr());
        const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, workspace.num_max_pool_tokens, l1_topk_weights_buffer.get_end_ptr());
        const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, num_padded_sf_pool_tokens_, l2_token_buffer.get_end_ptr());

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {static_cast<int64_t>(workspace.num_max_pool_tokens), static_cast<int64_t>(hidden_)},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {static_cast<int64_t>(config_.sf_pool_stride_tokens), static_cast<int64_t>(hidden_ / kGranK)},
            {1, static_cast<int64_t>(config_.sf_pool_stride_tokens)},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {static_cast<int64_t>(workspace.num_max_pool_tokens), static_cast<int64_t>(intermediate_hidden_)},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {static_cast<int64_t>(config_.sf_pool_stride_tokens), static_cast<int64_t>(intermediate_hidden_ / 64)},
            {1, static_cast<int64_t>(config_.sf_pool_stride_tokens)},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return {l1_acts, l1_acts_sf, l2_acts, l2_acts_sf};
    }

    void check_inputs(const std::tuple<int, int, int>& recipe, const std::string& activation) const {
        DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
        const auto [rm, rn, rk] = recipe;
        DG_HOST_ASSERT(rm == 128 and rn == 128 and rk == 128);
        DG_HOST_ASSERT(activation == "swiglu");
        DG_HOST_ASSERT(not sym_buffer_ptrs_.empty());
        DG_HOST_ASSERT(num_ranks() <= layout::kNumMaxRanks);
        DG_HOST_ASSERT(rank_idx_ < num_ranks());
        DG_HOST_ASSERT(num_experts_ % num_ranks() == 0);
        DG_HOST_ASSERT(num_topk_ <= 32);
        DG_HOST_ASSERT(activation_clamp_ >= 0);
        DG_HOST_ASSERT(num_tokens_ <= num_max_tokens_per_rank_);
        DG_HOST_ASSERT(l1_weights_.dim() == 3 and l2_weights_.dim() == 3);
        DG_HOST_ASSERT(l1_weights_.size(0) == num_experts_per_rank_ and
                       l2_weights_.size(0) == num_experts_per_rank_);
        DG_HOST_ASSERT(l1_weights_.size(1) == 2 * intermediate_hidden_ and l1_weights_.size(2) == hidden_);
        DG_HOST_ASSERT(l2_weights_.size(1) == hidden_ and l2_weights_.size(2) == intermediate_hidden_);
        DG_HOST_ASSERT(y_.dim() == 2 and y_.size(1) == hidden_);
        DG_HOST_ASSERT(hidden_ % 128 == 0 and intermediate_hidden_ % 128 == 0);
        DG_HOST_ASSERT(intermediate_hidden_ / 64 <= 64);
        DG_HOST_ASSERT(kernel1_sms_ > 1 and kernel2_sms_ > 1);
        const auto device_sms = static_cast<uint32_t>(device_runtime->get_num_sms());
        DG_HOST_ASSERT(kernel1_sms_ + kernel2_sms_ <= device_sms);
        DG_HOST_ASSERT(reduce_sms_ <= static_cast<uint32_t>(device_runtime->get_num_sms()));
        DG_HOST_ASSERT(config_.cluster_size > 0);
        const auto split_coscheduled_sm_count = std::max<uint32_t>(2, static_cast<uint32_t>(config_.cluster_size));
        DG_HOST_ASSERT(kernel1_sms_ % split_coscheduled_sm_count == 0);
        DG_HOST_ASSERT(kernel2_sms_ % split_coscheduled_sm_count == 0);

        DG_HOST_ASSERT(y_.is_cuda() and y_.is_contiguous() and y_.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(sym_buffer_.is_cuda() and sym_buffer_.is_contiguous() and sym_buffer_.scalar_type() == torch::kInt8);
        DG_HOST_ASSERT(sym_buffer_.nbytes() >= get_required_sym_buffer_bytes());
        DG_HOST_ASSERT(stats_.is_cuda() and stats_.is_contiguous() and stats_.scalar_type() == torch::kInt);
        const auto stats_numel = stats_.numel();
        const bool phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0;
        const auto phase_profile_int_offset = math::align<uint32_t>(num_experts_per_rank_, 2u);
        DG_HOST_ASSERT(phase_profile ?
                       stats_numel >= phase_profile_int_offset + 256 :
                       stats_numel >= num_experts_per_rank_);
        DG_HOST_ASSERT(get_major_type_ab(l1_weights_) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(l2_weights_) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(l1_weights_.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(l2_weights_.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(l1_weights_.is_contiguous() and l2_weights_.is_contiguous());

        check_sf_layout(l1_weights_sf_, intermediate_hidden_ * 2, hidden_, 128, 128,
                        num_experts_per_rank_, false, true, torch::kFloat);
        check_sf_layout(l2_weights_sf_, hidden_, intermediate_hidden_, 128, 128,
                        num_experts_per_rank_, false, true, torch::kFloat);
    }

    MegaMoESM90Config make_phase_config(
        const SM90FP8MegaMoERuntime::KernelPhase& phase) const {
        auto phase_config = config_;

        // Exact-shape experiment: allow the two Green Context phases to use
        // different row tilings.  The graph already builds separate kernels
        // and TMA descriptors for L1 and L2, so keeping this override here
        // isolates phase-local tiling without perturbing the regular selector.
        const bool exact_h200_target =
            num_ranks() == 8 and num_experts_ == 128 and num_experts_per_rank_ == 16 and
            num_max_tokens_per_rank_ >= 7000 and num_tokens_ == 7000 and num_topk_ == 16 and
            hidden_ == 3072 and intermediate_hidden_ == 1536;
        const char* phase_block_m_env =
            phase == SM90FP8MegaMoERuntime::KernelPhase::Linear1 ?
            "DG_SM90_MOE_GREEN_L1_BLOCK_M" : "DG_SM90_MOE_GREEN_L2_BLOCK_M";
        const int phase_block_m = get_env<int>(phase_block_m_env, 0);
        DG_HOST_ASSERT(phase_block_m == 0 or phase_block_m == 64 or phase_block_m == 128);
        if (exact_h200_target) {
            // Keep experimental switches scoped to the requested H200 shape.
            // Each switch is independently JIT-specialized through the config.
            phase_config.l2_dual_accum = get_env<int>("DG_SM90_MOE_L2_DUAL_ACCUM", 0) != 0;
            phase_config.l1_dual_k_accum = get_env<int>("DG_SM90_MOE_L1_DUAL_K", 0) != 0;
            phase_config.l2_sfa_2row_tma = get_env<int>("DG_SM90_MOE_L2_SFA_2ROW_TMA", 0) != 0;
            phase_config.l2_precompute_scale = get_env<int>("DG_SM90_MOE_L2_PRECOMPUTE_SCALE", 0) != 0;
            phase_config.l2_ratio_accum = get_env<int>("DG_SM90_MOE_L2_RATIO_ACCUM", 0) != 0;
            phase_config.l2_weight_sf_ldu = get_env<int>("DG_SM90_MOE_L2_WEIGHT_SF_LDU", 0) != 0;
            phase_config.l2_prefetch_weight_sf = get_env<int>("DG_SM90_MOE_L2_PREFETCH_WEIGHT_SF", 0) != 0;
            phase_config.l2_fast_weight_sf_n = get_env<int>("DG_SM90_MOE_L2_FAST_WEIGHT_SF_N", 0) != 0;
            phase_config.l2_hoist_weight_sf_base = get_env<int>("DG_SM90_MOE_L2_HOIST_WEIGHT_SF_BASE", 0) != 0;
            phase_config.l2_delay_sfa_load = get_env<int>("DG_SM90_MOE_L2_DELAY_SFA_LOAD", 0) != 0;
            phase_config.l2_split_sfa_load = get_env<int>("DG_SM90_MOE_L2_SPLIT_SFA_LOAD", 0) != 0;
            phase_config.l2_skip_post_wgmma_fence = get_env<int>("DG_SM90_MOE_L2_SKIP_POST_WGMMA_FENCE", 0) != 0;
            phase_config.l2_first_accum_assign = get_env<int>("DG_SM90_MOE_L2_FIRST_ACCUM_ASSIGN", 0) != 0;
            phase_config.l2_n64_interleave = get_env<int>("DG_SM90_MOE_L2_N64_INTERLEAVE", 0) != 0;
            phase_config.l2_half_k_pipeline = get_env<int>("DG_SM90_MOE_L2_HALF_K_PIPELINE", 0) != 0;
            phase_config.async_l1_tma_store = get_env<int>("DG_SM90_MOE_ASYNC_L1_STORE", 0) != 0;
            if (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce and
                get_env<int>("DG_SM90_MOE_L2_SPLIT_SFA_TMA", 0) != 0) {
                // K2NoReduce does not need dispatch/cleanup workers.  Reuse the
                // same four-warp frontend as three independent TMA producers
                // plus one idle warp, keeping the math warpgroups aligned.
                phase_config.num_dispatch_threads = 0;
                phase_config.num_non_epilogue_threads = 128;
                phase_config.split_sfa_tma = true;
            }
            if (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce) {
                const int forced_l2_stages =
                    get_env<int>("DG_SM90_MOE_GREEN_L2_NUM_STAGES", 0);
                DG_HOST_ASSERT(forced_l2_stages == 0 or
                               (forced_l2_stages >= 2 and forced_l2_stages <= 5));
                if (forced_l2_stages > 0)
                    phase_config.num_stages = forced_l2_stages;
            }
        }
        if (exact_h200_target and phase_block_m != 0) {
            phase_config.block_m = phase_block_m;
            phase_config.block_n = phase_block_m == 64 ? 256 : 128;
            phase_config.block_k = 128;
            phase_config.cluster_size = 1;
            phase_config.num_dispatch_threads = phase_block_m == 64 ? 64 : 128;
            phase_config.num_non_epilogue_threads = phase_block_m == 64 ? 64 : 128;
            phase_config.num_epilogue_threads = 256;
            phase_config.num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
                static_cast<int>(num_experts_per_rank_), static_cast<int>(num_tokens_),
                static_cast<int>(num_topk_), static_cast<int>(intermediate_hidden_),
                phase_config.block_m, phase_config.block_n, device_runtime->get_num_sms());
            phase_config.direct_l2_scatter = false;
            phase_config.l2_nmajor_schedule = false;
            phase_config.one_warp_cleanup = false;
            const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
                SM90ArchSpec::smem_capacity,
                static_cast<int>(num_experts_), static_cast<int>(hidden_),
                phase_config.block_m, phase_config.block_n, phase_config.block_k,
                phase_config.num_dispatch_threads / 32,
                phase_config.num_epilogue_threads / 32,
                false, false, 4);
            phase_config.num_stages = num_stages;
            phase_config.smem_size = smem_size;
        }
        // Use the tuned EPW only for the fully compressed target configuration.
        // Preserve the global EPW environment variable as a debugging override.
        if (exact_h200_target and
            kernel1_sms_ == 78 and kernel2_sms_ == 54 and reduce_sms_ == 124 and
            fp8_compressed_combine_requested_ and
            get_env<int>("DG_SM90_MOE_SCALE_FREE_L1_K2", 0) != 0 and
            get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE", 0) == 0) {
            phase_config.num_experts_per_wave = 4;
        }
        if (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear1)
            phase_config.direct_l2_scatter = false;
        if (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear1 and
            phase_config.async_l1_tma_store) {
            const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
                SM90ArchSpec::smem_capacity,
                static_cast<int>(num_experts_), static_cast<int>(hidden_),
                phase_config.block_m, phase_config.block_n, phase_config.block_k,
                phase_config.num_dispatch_threads / 32,
                phase_config.num_epilogue_threads / 32,
                false, true, phase_config.num_stages);
            phase_config.num_stages = num_stages;
            phase_config.smem_size = smem_size;
        }
        if (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce) {
            const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
                SM90ArchSpec::smem_capacity,
                static_cast<int>(num_experts_), static_cast<int>(hidden_),
                phase_config.block_m, phase_config.block_n, phase_config.block_k,
                phase_config.num_dispatch_threads / 32,
                phase_config.num_epilogue_threads / 32,
                phase_config.direct_l2_scatter,
                phase_config.async_l1_tma_store,
                phase_config.num_stages);
            phase_config.num_stages = num_stages;
            phase_config.smem_size = smem_size;
        }
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
            const auto phase_name = phase == SM90FP8MegaMoERuntime::KernelPhase::Linear1 ? "Linear1" :
                (phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2 ? "Linear2" : "Linear2NoReduce");
            const auto key = fmt::format(
                "MegaMoESM90PhaseConfig(phase={}, num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, block_m={}, block_n={}, block_k={}, dispatch_warps={}, non_epilogue_warps={}, epilogue_warps={}, stages={}, smem_size={}, direct_l2={}, nmajor={}, cleanup={}, l2dual={}, l1dual={}, l2sfa2={}, l2prescale={}, l2ratio={}, l2sfldu={}, l2prefetchsf={}, l2fastsfn={}, l2hoistsfb={}, delay_sfa={}, l2split_sfa={}, l2skip_post_fence={}, l2first_assign={}, l2n64_interleave={}, l2halfpipe={}, async_l1_tma={})",
                phase_name, num_ranks(), num_experts_, hidden_, intermediate_hidden_,
                num_max_tokens_per_rank_, num_tokens_, num_topk_,
                phase_config.block_m, phase_config.block_n, phase_config.block_k,
                phase_config.num_dispatch_threads / 32,
                phase_config.num_non_epilogue_threads / 32,
                phase_config.num_epilogue_threads / 32,
                phase_config.num_stages, phase_config.smem_size,
                phase_config.direct_l2_scatter, phase_config.l2_nmajor_schedule,
                phase_config.one_warp_cleanup,
                phase_config.l2_dual_accum,
                phase_config.l1_dual_k_accum,
                phase_config.l2_sfa_2row_tma,
                phase_config.l2_precompute_scale,
                phase_config.l2_ratio_accum,
                phase_config.l2_weight_sf_ldu,
                phase_config.l2_prefetch_weight_sf,
                phase_config.l2_fast_weight_sf_n,
                phase_config.l2_hoist_weight_sf_base,
                phase_config.l2_delay_sfa_load,
                phase_config.l2_split_sfa_load,
                phase_config.l2_skip_post_wgmma_fence,
                phase_config.l2_first_accum_assign,
                phase_config.l2_n64_interleave,
                phase_config.l2_half_k_pipeline,
                phase_config.async_l1_tma_store);
            static std::unordered_set<std::string> printed;
            if (printed.count(key) == 0) {
                std::cout << key << std::endl;
                printed.insert(key);
            }
        }
        return phase_config;
    }

    SM90FP8MegaMoERuntime::Args make_phase_args(
        const SM90FP8MegaMoERuntime::KernelPhase& phase,
        const uint32_t& num_sms,
        const uint32_t& dispatch_num_sms) const {
        auto phase_config = make_phase_config(phase);

        return {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_), .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_), .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .activation_clamp = activation_clamp_,
            .fast_math = fast_math_,
            .direct_l2_scatter = phase_config.direct_l2_scatter,
            .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
            .l2_nmajor_schedule = phase_config.l2_nmajor_schedule,
            .one_warp_cleanup = phase_config.one_warp_cleanup,
            .dispatch_num_sms = static_cast<int>(dispatch_num_sms),
            .kernel_phase = phase,
            .scale_free_l1_k2 = get_env<int>("DG_SM90_MOE_SCALE_FREE_L1_K2", 0) != 0,
            .drop_scale_free_sfa = get_env<int>("DG_SM90_MOE_SCALE_FREE_DROP_SFA", 1) != 0,
            .fp8_compressed_combine = false,
            .config = phase_config,
            .y = nullptr,
            .cumulative_local_expert_recv_stats = nullptr,
            .num_tokens = static_cast<int>(num_tokens_),
            .sym_buffer_ptrs = layout::SymBuffer<>(),
            .tensor_map_l1_acts = {},
            .tensor_map_l1_acts_sf = {},
            .tensor_map_l1_weights = {},
            .l1_weights_sf = nullptr,
            .tensor_map_l1_output = {},
            .tensor_map_l2_acts = {},
            .tensor_map_l2_acts_sf = {},
            .tensor_map_l2_weights = {},
            .l2_weights_sf = nullptr,
            .launch_args = LaunchArgs(static_cast<int>(num_sms),
                                      phase_config.num_dispatch_threads + phase_config.num_non_epilogue_threads + phase_config.num_epilogue_threads,
                                      phase_config.smem_size, phase_config.cluster_size, false)
        };
    }

    uint32_t get_reduce_grid() const {
        uint32_t reduce_grid = (num_ranks() == 1 and num_tokens_ <= 2048) ?
            std::max(num_tokens_, reduce_sms_) : reduce_sms_;
        DG_HOST_ASSERT(reduce_grid > 0);
        if (num_ranks() > 1)
            DG_HOST_ASSERT(reduce_grid <= reduce_sms_);
        return reduce_grid;
    }

    int get_reduce_threads() const {
        const int threads = get_env<int>("DG_SM90_MOE_K3_THREADS", 256);
        DG_HOST_ASSERT(threads == 128 or threads == 256 or threads == 384 or threads == 512);
        return threads;
    }

    bool use_fused_reduce() const {
        return get_env<int>("DG_SM90_MOE_GREEN_FUSED_REDUCE", 0) != 0;
    }

    void build_kernel_runtimes() {
        const auto kernel2_phase = use_fused_reduce() ?
            SM90FP8MegaMoERuntime::KernelPhase::Linear2 :
            SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce;
        const auto kernel1_static_args = make_phase_args(
            SM90FP8MegaMoERuntime::KernelPhase::Linear1, kernel1_sms_, kernel1_sms_);
        auto kernel2_static_args = make_phase_args(
            kernel2_phase, kernel2_sms_, kernel1_sms_);
        const int reduce_threads = get_reduce_threads();
        const uint32_t reduce_grid = get_reduce_grid();
        const bool direct_global_reduce =
            num_ranks() > 1 and
            get_env<int>("DG_SM90_MOE_K3_DIRECT_GLOBAL_REDUCE", 0) != 0;
        const bool fp8_compressed_combine =
            fp8_compressed_combine_requested_ &&
            kernel2_phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce &&
            not direct_global_reduce && (reduce_threads == 256 || reduce_threads == 384) &&
            num_ranks() == 8 && num_max_tokens_per_rank_ >= 7000 &&
            hidden_ == 3072 && intermediate_hidden_ == 1536 &&
            num_experts_ == 128 && num_topk_ == 16 &&
            kernel2_static_args.config.block_m == 192 &&
            kernel2_static_args.config.block_n == 128 &&
            kernel2_static_args.config.block_k == 128 &&
            kernel2_static_args.config.num_epilogue_threads == 384;
        kernel2_static_args.fp8_compressed_combine = fp8_compressed_combine;
        const int reduce_smem_size = (num_ranks() == 1 or direct_global_reduce) ? 0 :
            get_sm90_moe_combine_reduce_smem_size(static_cast<int>(hidden_), reduce_threads);
        const SM90FP8MegaMoECombineReduceRuntime::Args reduce_static_args = {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_),
            .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_),
            .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .num_padded_sf_pool_tokens = static_cast<int>(num_padded_sf_pool_tokens_),
            .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
            .direct_global_reduce = direct_global_reduce,
            .fp8_compressed_combine = fp8_compressed_combine,
            .y = nullptr,
            .cumulative_local_expert_recv_stats = nullptr,
            .num_tokens = num_tokens_,
            .sym_buffer_ptrs = layout::SymBuffer<>(),
            .launch_args = LaunchArgs(static_cast<int>(reduce_grid), reduce_threads,
                                      reduce_smem_size, 1, false)
        };

        const auto kernel1_code = SM90FP8MegaMoERuntime::generate(kernel1_static_args);
        const auto kernel2_code = SM90FP8MegaMoERuntime::generate(kernel2_static_args);
        const auto reduce_code = SM90FP8MegaMoECombineReduceRuntime::generate(reduce_static_args);
        kernel1_runtime_ = compiler->build("sm90_fp8_mega_moe_l1_impl", kernel1_code);
        kernel2_runtime_ = compiler->build("sm90_fp8_mega_moe_l2_no_reduce_impl", kernel2_code);
        reduce_runtime_ = compiler->build("sm90_fp8_mega_moe_combine_reduce_impl", reduce_code);
        kernel1_graph_kernel_ = kernel1_runtime_->kernel;
        kernel2_graph_kernel_ = kernel2_runtime_->kernel;
        reduce_graph_kernel_ = reduce_runtime_->kernel;
    }

    void prepare_kernel_params() {
        const auto views = slice_buffer(sym_buffer_);
        constexpr int kGranK = 128;
        constexpr int kL2ActsSFGranK = 64;

        const auto kernel2_phase = use_fused_reduce() ?
            SM90FP8MegaMoERuntime::KernelPhase::Linear2 :
            SM90FP8MegaMoERuntime::KernelPhase::Linear2NoReduce;
        kernel1_args_ = make_phase_args(
            SM90FP8MegaMoERuntime::KernelPhase::Linear1, kernel1_sms_, kernel1_sms_);
        kernel2_args_ = make_phase_args(
            kernel2_phase, kernel2_sms_, kernel1_sms_);
        const auto& l1_config = kernel1_args_.config;
        const auto& l2_config = kernel2_args_.config;

        const auto tensor_map_l1_acts = make_tma_2d_desc(
            views.l1_acts, hidden_, l1_config.num_max_pool_tokens,
            l1_config.block_k, l1_config.block_m,
            static_cast<int>(views.l1_acts.stride(-2)),
            l1_config.swizzle_acts_mode);
        const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
            cute::UMMA::Major::MN, views.l1_acts_sf,
            l1_config.sf_pool_stride_tokens, hidden_,
            l1_config.block_m, kGranK,
            1, 0);
        const auto tensor_map_l1_weights = make_tma_2d_desc(
            l1_weights_, hidden_, num_experts_per_rank_ * intermediate_hidden_ * 2,
            l1_config.block_k, l1_config.block_n,
            static_cast<int>(l1_weights_.stride(-2)),
            l1_config.swizzle_weights_mode);
        const int num_epilogue_warpgroups_h = l1_config.num_epilogue_threads / 128;
        const bool split_n_warpgroups_h =
            l1_config.block_m == 64 and l1_config.block_n == 256 and num_epilogue_warpgroups_h == 2;
        const int wg_block_m = split_n_warpgroups_h ? l1_config.block_m : l1_config.block_m / num_epilogue_warpgroups_h;
        const auto tensor_map_l1_output = make_tma_2d_desc(
            views.l2_acts, intermediate_hidden_, l1_config.num_max_pool_tokens,
            l1_config.block_n / 2, wg_block_m,
            static_cast<int>(views.l2_acts.stride(-2)),
            0);
        const auto tensor_map_l2_acts = make_tma_2d_desc(
            views.l2_acts, intermediate_hidden_, l2_config.num_max_pool_tokens,
            l2_config.block_k, l2_config.block_m,
            static_cast<int>(views.l2_acts.stride(-2)),
            l2_config.swizzle_acts_mode);
        const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
            cute::UMMA::Major::MN, views.l2_acts_sf,
            l2_config.sf_pool_stride_tokens, intermediate_hidden_,
            l2_config.block_m, kL2ActsSFGranK,
            1, 0, 0, l2_config.l2_sfa_2row_tma ? 2 : 1);
        const auto tensor_map_l2_weights = make_tma_2d_desc(
            l2_weights_, intermediate_hidden_, num_experts_per_rank_ * hidden_,
            l2_config.block_k, l2_config.block_n,
            static_cast<int>(l2_weights_.stride(-2)),
            l2_config.swizzle_weights_mode);

        for (auto* args: {&kernel1_args_, &kernel2_args_}) {
            args->y = y_.data_ptr();
            args->cumulative_local_expert_recv_stats = stats_.data_ptr<int>();
            args->num_tokens = static_cast<int>(num_tokens_);
            args->sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs_, rank_idx_);
            args->tensor_map_l1_acts = tensor_map_l1_acts;
            args->tensor_map_l1_acts_sf = tensor_map_l1_acts_sf;
            args->tensor_map_l1_weights = tensor_map_l1_weights;
            args->l1_weights_sf = l1_weights_sf_.data_ptr<float>();
            args->tensor_map_l1_output = tensor_map_l1_output;
            args->tensor_map_l2_acts = tensor_map_l2_acts;
            args->tensor_map_l2_acts_sf = tensor_map_l2_acts_sf;
            args->tensor_map_l2_weights = tensor_map_l2_weights;
            args->l2_weights_sf = l2_weights_sf_.data_ptr<float>();
        }
        const uint32_t reduce_grid = get_reduce_grid();
        const bool direct_global_reduce =
            num_ranks() > 1 and
            get_env<int>("DG_SM90_MOE_K3_DIRECT_GLOBAL_REDUCE", 0) != 0;
        const int reduce_threads = get_reduce_threads();
        const int reduce_smem_size = (num_ranks() == 1 or direct_global_reduce) ? 0 :
            get_sm90_moe_combine_reduce_smem_size(static_cast<int>(hidden_), reduce_threads);
        reduce_args_ = {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_),
            .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_),
            .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .num_padded_sf_pool_tokens = static_cast<int>(num_padded_sf_pool_tokens_),
            .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
            .direct_global_reduce = direct_global_reduce,
            .y = y_.data_ptr(),
            .cumulative_local_expert_recv_stats = stats_.data_ptr<int>(),
            .num_tokens = num_tokens_,
            .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs_, rank_idx_),
            .launch_args = LaunchArgs(static_cast<int>(reduce_grid), reduce_threads,
                                      reduce_smem_size, 1, false)
        };

        kernel1_params_ = {
            &kernel1_args_.y,
            &kernel1_args_.cumulative_local_expert_recv_stats,
            &kernel1_args_.num_tokens,
            &kernel1_args_.sym_buffer_ptrs,
            &kernel1_args_.tensor_map_l1_acts,
            &kernel1_args_.tensor_map_l1_acts_sf,
            &kernel1_args_.tensor_map_l1_weights,
            &kernel1_args_.l1_weights_sf,
            &kernel1_args_.tensor_map_l1_output,
            &kernel1_args_.tensor_map_l2_acts,
            &kernel1_args_.tensor_map_l2_acts_sf,
            &kernel1_args_.tensor_map_l2_weights,
            &kernel1_args_.l2_weights_sf
        };
        kernel2_params_ = {
            &kernel2_args_.y,
            &kernel2_args_.cumulative_local_expert_recv_stats,
            &kernel2_args_.num_tokens,
            &kernel2_args_.sym_buffer_ptrs,
            &kernel2_args_.tensor_map_l1_acts,
            &kernel2_args_.tensor_map_l1_acts_sf,
            &kernel2_args_.tensor_map_l1_weights,
            &kernel2_args_.l1_weights_sf,
            &kernel2_args_.tensor_map_l1_output,
            &kernel2_args_.tensor_map_l2_acts,
            &kernel2_args_.tensor_map_l2_acts_sf,
            &kernel2_args_.tensor_map_l2_weights,
            &kernel2_args_.l2_weights_sf
        };
        reduce_params_ = {
            &reduce_args_.y,
            &reduce_args_.cumulative_local_expert_recv_stats,
            &reduce_args_.num_tokens,
            &reduce_args_.sym_buffer_ptrs
        };
    }

    void create_green_contexts() {
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGetDevice(&device_idx_));
        DG_CUDA_RUNTIME_CHECK(lazy_cudaSetDevice(device_idx_));
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDeviceGetExecutionCtx(&primary_context_, device_idx_));
        DG_HOST_ASSERT(primary_context_ != nullptr);

        cudaDevResource sm_resource = {};
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDeviceGetDevResource(device_idx_, &sm_resource, cudaDevResourceTypeSm));
        DG_HOST_ASSERT(sm_resource.type == cudaDevResourceTypeSm);
        DG_HOST_ASSERT(kernel1_sms_ + kernel2_sms_ <= sm_resource.sm.smCount);

        cudaDevResource workqueue_resource = {};
        bool has_workqueue_resource = false;
        const auto workqueue_status = lazy_cudaDeviceGetDevResource(
            device_idx_, &workqueue_resource, cudaDevResourceTypeWorkqueueConfig);
        if (workqueue_status == cudaSuccess) {
            DG_HOST_ASSERT(workqueue_resource.type == cudaDevResourceTypeWorkqueueConfig);
            workqueue_resource.wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
            has_workqueue_resource = true;
        } else if (workqueue_status != cudaErrorInvalidResourceType) {
            DG_CUDA_RUNTIME_CHECK(workqueue_status);
        }

        std::array<cudaDevResource, 2> split_resources = {};
        cudaDevResource remainder = {};
        const auto split_coscheduled_sm_count = std::max<uint32_t>(2, static_cast<uint32_t>(config_.cluster_size));
        DG_HOST_ASSERT(kernel1_sms_ % split_coscheduled_sm_count == 0);
        DG_HOST_ASSERT(kernel2_sms_ % split_coscheduled_sm_count == 0);
        std::array<cudaDevSmResourceGroupParams, 2> group_params = {};
        group_params[0].smCount = kernel1_sms_;
        group_params[0].coscheduledSmCount = split_coscheduled_sm_count;
        group_params[1].smCount = kernel2_sms_;
        group_params[1].coscheduledSmCount = split_coscheduled_sm_count;
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDevSmResourceSplit(
            split_resources.data(), static_cast<unsigned int>(split_resources.size()),
            &sm_resource, &remainder, 0, group_params.data()));
        DG_HOST_ASSERT(split_resources[0].sm.smCount == kernel1_sms_);
        DG_HOST_ASSERT(split_resources[1].sm.smCount == kernel2_sms_);

        for (uint32_t context_idx = 0; context_idx < split_resources.size(); ++context_idx) {
            DG_HOST_ASSERT(split_resources[context_idx].type == cudaDevResourceTypeSm);
            std::array<cudaDevResource, 2> context_resources = {
                split_resources[context_idx],
                workqueue_resource
            };
            const auto num_context_resources = has_workqueue_resource ? context_resources.size() : 1;
            cudaDevResourceDesc_t resource_desc = nullptr;
            DG_CUDA_RUNTIME_CHECK(lazy_cudaDevResourceGenerateDesc(
                &resource_desc, context_resources.data(), num_context_resources));
            DG_CUDA_RUNTIME_CHECK(lazy_cudaGreenCtxCreate(
                &green_contexts_[context_idx], resource_desc, device_idx_, 0));
            DG_CUDA_RUNTIME_CHECK(lazy_cudaExecutionCtxGetId(
                green_contexts_[context_idx], &green_context_ids_[context_idx]));
        }
    }

    cudaGraphNode_t add_kernel_node(
        const cudaExecutionContext_t& context,
        const KernelHandle& kernel,
        const LaunchArgs& launch_args,
        void** kernel_params,
        const std::vector<cudaGraphNode_t>& dependencies) {
        if (launch_args.smem_size > 0) {
        #if defined(DG_JIT_USE_RUNTIME_API)
            DG_CUDA_RUNTIME_CHECK(lazy_cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_args.smem_size));
        #else
            DG_CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(
                kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, launch_args.smem_size));
        #endif
        }

        cudaGraphNode_t node = nullptr;
        cudaGraphNodeParams node_params = {};
        node_params.type = cudaGraphNodeTypeKernel;
    #if defined(DG_JIT_USE_RUNTIME_API)
        node_params.kernel.kern = kernel;
        node_params.kernel.functionType = cudaKernelFunctionTypeKernel;
    #else
        node_params.kernel.cuFunc = kernel;
        node_params.kernel.functionType = cudaKernelFunctionTypeFunction;
    #endif
        node_params.kernel.gridDim = dim3(
            static_cast<unsigned>(launch_args.grid_dim.first),
            static_cast<unsigned>(launch_args.grid_dim.second), 1);
        node_params.kernel.blockDim = dim3(static_cast<unsigned>(launch_args.num_threads), 1, 1);
        node_params.kernel.sharedMemBytes = launch_args.smem_size;
        node_params.kernel.kernelParams = kernel_params;
        node_params.kernel.extra = nullptr;
        node_params.kernel.ctx = context;

        const auto dependency_ptr = dependencies.empty() ? nullptr : dependencies.data();
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphAddNode(
            &node, graph_, dependency_ptr, nullptr, dependencies.size(), &node_params));

        if (launch_args.cluster_dim > 1) {
            cudaKernelNodeAttrValue attr = {};
            attr.clusterDim = {static_cast<unsigned>(launch_args.cluster_dim), 1, 1};
            DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphKernelNodeSetAttribute(
                node, cudaKernelNodeAttributeClusterDimension, &attr));
        }
        return node;
    }

    void build_graph() {
        prepare_kernel_params();
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphCreate(&graph_, 0));

        kernel1_node_ = add_kernel_node(
            green_contexts_[0], kernel1_graph_kernel_, kernel1_args_.launch_args,
            kernel1_params_.data(), {});
        kernel2_node_ = add_kernel_node(
            green_contexts_[1], kernel2_graph_kernel_, kernel2_args_.launch_args,
            kernel2_params_.data(), {});
        if (!use_fused_reduce()) {
            const std::vector<cudaGraphNode_t> reduce_dependencies = {
                kernel1_node_, kernel2_node_
            };
            reduce_node_ = add_kernel_node(
                primary_context_, reduce_graph_kernel_, reduce_args_.launch_args,
                reduce_params_.data(), reduce_dependencies);
        }

        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphInstantiate(&graph_exec_, graph_, 0));
    }

    void destroy_noexcept() noexcept {
        if (graph_launched_ and last_launch_stream_ != nullptr) {
            (void) lazy_cudaStreamSynchronize(last_launch_stream_);
            graph_launched_ = false;
            last_launch_stream_ = nullptr;
        }
        if (graph_exec_ != nullptr) {
            (void) lazy_cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void) lazy_cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        const auto context0 = green_contexts_[0];
        const auto context1 = green_contexts_[1];
        if (context0 != nullptr) {
            (void) lazy_cudaExecutionCtxDestroy(context0);
        }
        if (context1 != nullptr and context1 != context0) {
            (void) lazy_cudaExecutionCtxDestroy(context1);
        }
        green_contexts_ = {nullptr, nullptr};
        green_context_ids_ = {0, 0};
    }

public:
    SM90FP8MegaMoEGreenCtxGraph(
        const torch::Tensor& y,
        const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
        const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
        const torch::Tensor& cumulative_local_expert_recv_stats,
        const torch::Tensor& sym_buffer,
        const std::vector<int64_t>& sym_buffer_ptrs,
        const int& rank_idx,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_topk,
        const std::tuple<int, int, int>& recipe,
        const std::string& activation,
        const std::optional<float>& activation_clamp_opt,
        const bool& fast_math,
        const int& kernel1_sms,
        const int& kernel2_sms,
        const int& reduce_sms
    ) : y_(y),
        sym_buffer_(sym_buffer),
        l1_weights_(std::get<0>(l1_weights_tuple)),
        l1_weights_sf_(std::get<1>(l1_weights_tuple)),
        l2_weights_(std::get<0>(l2_weights_tuple)),
        l2_weights_sf_(std::get<1>(l2_weights_tuple)),
        stats_(cumulative_local_expert_recv_stats),
        sym_buffer_ptrs_(sym_buffer_ptrs),
        rank_idx_(checked_nonnegative_u32(rank_idx)),
        num_max_tokens_per_rank_(checked_positive_u32(num_max_tokens_per_rank)),
        num_experts_(checked_positive_u32(num_experts)),
        num_topk_(checked_positive_u32(num_topk)),
        num_tokens_(checked_nonnegative_u32(static_cast<int>(y.size(0)))),
        hidden_(checked_positive_u32(static_cast<int>(std::get<0>(l1_weights_tuple).size(2)))),
        intermediate_hidden_(checked_positive_u32(static_cast<int>(std::get<0>(l2_weights_tuple).size(2)))),
        kernel1_sms_(checked_positive_u32(kernel1_sms)),
        kernel2_sms_(checked_positive_u32(kernel2_sms)),
        reduce_sms_(checked_positive_u32(reduce_sms)),
        activation_clamp_(activation_clamp_opt.value_or(std::numeric_limits<float>::infinity())),
        fast_math_(fast_math),
        fp8_compressed_combine_requested_(
            get_env<int>("DG_SM90_MOE_FP8_COMPRESSED_COMBINE", 0) != 0) {
        DG_HOST_ASSERT(not sym_buffer_ptrs_.empty());
        DG_HOST_ASSERT(num_experts_ % num_ranks() == 0);
        num_experts_per_rank_ = num_experts_ / num_ranks();
        const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
            static_cast<int>(num_ranks()), static_cast<int>(num_max_tokens_per_rank_),
            static_cast<int>(num_topk_), static_cast<int>(num_experts_per_rank_));
        int num_padded_sf_pool_tokens = 0;
        for (int block_m: layout::kCandidateBlockM) {
            num_padded_sf_pool_tokens = std::max(
                num_padded_sf_pool_tokens,
                layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m));
        }
        num_padded_sf_pool_tokens_ = static_cast<uint32_t>(num_padded_sf_pool_tokens);
        config_ = get_mega_moe_config_sm90(
            static_cast<int>(num_ranks()), static_cast<int>(num_experts_), static_cast<int>(num_experts_per_rank_),
            static_cast<int>(num_max_tokens_per_rank_), static_cast<int>(num_tokens_), static_cast<int>(num_topk_),
            static_cast<int>(hidden_), static_cast<int>(intermediate_hidden_),
            static_cast<int>(num_padded_sf_pool_tokens_));
        check_inputs(recipe, activation);
        try {
            build_kernel_runtimes();
            create_green_contexts();
            build_graph();
        } catch (...) {
            destroy_noexcept();
            throw;
        }
    }

    SM90FP8MegaMoEGreenCtxGraph(const SM90FP8MegaMoEGreenCtxGraph&) = delete;
    SM90FP8MegaMoEGreenCtxGraph& operator=(const SM90FP8MegaMoEGreenCtxGraph&) = delete;

    ~SM90FP8MegaMoEGreenCtxGraph() {
        destroy_noexcept();
    }

    void replay() {
        DG_HOST_ASSERT(graph_exec_ != nullptr);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if (graph_launched_ and last_launch_stream_ != nullptr and last_launch_stream_ != stream) {
            DG_CUDA_RUNTIME_CHECK(lazy_cudaStreamSynchronize(last_launch_stream_));
        }
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphLaunch(graph_exec_, stream));
        last_launch_stream_ = stream;
        graph_launched_ = true;
    }

    std::tuple<unsigned long long, unsigned long long> get_green_context_ids() const {
        return {green_context_ids_[0], green_context_ids_[1]};
    }
};

#else

class SM90FP8MegaMoEGreenCtxGraph final {
public:
    SM90FP8MegaMoEGreenCtxGraph(
        const torch::Tensor&,
        const std::tuple<torch::Tensor, torch::Tensor>&,
        const std::tuple<torch::Tensor, torch::Tensor>&,
        const torch::Tensor&,
        const torch::Tensor&,
        const std::vector<int64_t>&,
        const int&,
        const int&,
        const int&,
        const int&,
        const std::tuple<int, int, int>&,
        const std::string&,
        const std::optional<float>&,
        const bool&,
        const int&,
        const int&,
        const int&) {
        DG_HOST_UNREACHABLE("SM90 MegaMoE Green Context graph requires CUDA Runtime 13.2+");
    }

    void replay() {
        DG_HOST_UNREACHABLE("SM90 MegaMoE Green Context graph requires CUDA Runtime 13.2+");
    }

    std::tuple<unsigned long long, unsigned long long> get_green_context_ids() const {
        DG_HOST_UNREACHABLE("SM90 MegaMoE Green Context graph requires CUDA Runtime 13.2+");
    }
};

#endif

static void sm90_fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.sf_pool_stride_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. In split-N mode, two WGs produce different N halves of the same
    // M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups_h =
        config.block_m == 64 and config.block_n == 256 and num_epilogue_warpgroups_h == 2;
    const int wg_block_m = split_n_warpgroups_h ? config.block_m : config.block_m / num_epilogue_warpgroups_h;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, wg_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.sf_pool_stride_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0, 0, config.l2_sfa_2row_tma ? 2 : 1);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .direct_l2_scatter = config.direct_l2_scatter,
        .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
        .l2_nmajor_schedule = config.l2_nmajor_schedule,
        .one_warp_cleanup = config.one_warp_cleanup,
        .dispatch_num_sms = num_sms,
        .kernel_phase = SM90FP8MegaMoERuntime::KernelPhase::Linear1,
        .scale_free_l1_k2 = false,
        .drop_scale_free_sfa = false,
        .fp8_compressed_combine = false,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = l1_weights_sf.data_ptr<float>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = l2_weights_sf.data_ptr<float>(),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto launch_with_phase = [&](const SM90FP8MegaMoERuntime::KernelPhase kernel_phase,
                                       const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        const auto code = SM90FP8MegaMoERuntime::generate(split_args);
        const auto runtime = compiler->build(kernel_name, code);
        SM90FP8MegaMoERuntime::launch(runtime, split_args);
    };

    launch_with_phase(SM90FP8MegaMoERuntime::KernelPhase::Linear1, "sm90_fp8_mega_moe_l1_impl");
    launch_with_phase(SM90FP8MegaMoERuntime::KernelPhase::Linear2, "sm90_fp8_mega_moe_l2_impl");
}

} // namespace deep_gemm
