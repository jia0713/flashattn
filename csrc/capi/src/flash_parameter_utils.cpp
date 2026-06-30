#include "flash_parameter.h" // Parameter
#include "flash_parameter_utils.h"
#include "utils.h"
#include <iostream>
#include <stdexcept>

using namespace mcFlashAttn;

int mcGetCurrentProcessorCount() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops.multiProcessorCount;
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const size_t d_v,
                      const size_t d_v_rounded,
                      // device pointers
                      Tensor_t q,
                      Tensor_t k,
                      Tensor_t v,
                      Tensor_t out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.is_bf16 = (get_tensor_dtype(q) == MCFLASHATTN_DATATYPE_BF16);

    // Set the pointers and strides.
    params.q_ptr = get_tensor_data(q);
    params.k_ptr = get_tensor_data(k);
    params.v_ptr = get_tensor_data(v);
    // All stride are in elements, not bytes.
    // params.q_row_stride = q.stride(-3);
    // params.k_row_stride = k.stride(-3);
    // params.v_row_stride = v.stride(-3);
    // params.q_head_stride = q.stride(-2);
    // params.k_head_stride = k.stride(-2);
    // params.v_head_stride = v.stride(-2);

    // All stride are in elements, not bytes.
    params.q_row_stride = get_tensor_stride(q,-3);
    params.k_row_stride = get_tensor_stride(k,-3);
    params.v_row_stride = get_tensor_stride(v,-3);
    params.q_head_stride = get_tensor_stride(q,-2);
    params.k_head_stride = get_tensor_stride(k,-2);
    params.v_head_stride = get_tensor_stride(v,-2);


    params.o_ptr = get_tensor_data(out);

    params.o_row_stride = get_tensor_stride(out,-3);
    params.o_head_stride = get_tensor_stride(out,-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = get_tensor_stride(q,0);
        params.k_batch_stride = get_tensor_stride(k,0);
        params.v_batch_stride = get_tensor_stride(v,0);
        params.o_batch_stride = get_tensor_stride(out,0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    // params.q_batch_stride = 0;
    // params.k_batch_stride = 0;
    // params.v_batch_stride = 0;
    // params.o_batch_stride = 0;

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.d_value = d_v;
    params.d_value_rounded = d_v_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif

    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    // TORCH_CHECK(p_dropout < 1.f);
    // #ifdef FLASHATTENTION_DISABLE_DROPOUT
    //     TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    // #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // #ifdef FLASHATTENTION_DISABLE_LOCAL
    //     TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
    //         "This flash attention build does not support local attention.");
    // #endif

    params.is_seqlens_k_cumulative = true;

    // #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
    //     TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    // #endif

    // default value
    params.num_splits = 1;
    params.unpadded_lse = false;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const size_t d_v,
                      const size_t d_v_rounded,
                      // device pointers
                      const Tensor_t q,
                      const Tensor_t k,
                      const Tensor_t v,
                      const Tensor_t out,
                      const Tensor_t dout,
                      Tensor_t dq,
                      Tensor_t dk,
                      Tensor_t dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic,
                      const float softcap) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     h, h_k, d, d_rounded,d_v,d_v_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);

    // Set the pointers and strides.
    params.do_ptr = get_tensor_data(dout);
    params.do_row_stride = get_tensor_stride(dout,-3);
    params.do_head_stride = get_tensor_stride(dout,-2);
    params.dq_ptr = get_tensor_data(dq);
    params.dk_ptr = get_tensor_data(dk);
    params.dv_ptr = get_tensor_data(dv);
    params.dq_row_stride = get_tensor_stride(dq,-3);
    params.dk_row_stride = get_tensor_stride(dk,-3);
    params.dv_row_stride = get_tensor_stride(dv,-3);
    params.dq_head_stride = get_tensor_stride(dq,-2);
    params.dk_head_stride = get_tensor_stride(dk,-2);
    params.dv_head_stride = get_tensor_stride(dv,-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = get_tensor_stride(dout,0);
        params.dq_batch_stride = get_tensor_stride(dq,0);
        params.dk_batch_stride = get_tensor_stride(dk,0);
        params.dv_batch_stride = get_tensor_stride(dv,0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int64_t batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    //if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    //Technically kBlockM = 64 only for the splitKV kernels
    if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
        return 1;
    }
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks) * num_splits / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.96 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int process_count, InternalTensor::DataType data_type) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    //const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int block_n = 32;
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 32 - 1) / 32;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) { 
            params.num_splits = 1;
        }
        if (params.num_splits > 1) { 
            // at::Tensor softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            // at::Tensor out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            // params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            // params.oaccum_ptr = out_accum.data_ptr();
            CHECK_THROW(false, "params.num_splits > 1 is not support in cpai for now");
        }
        // TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}

void set_params_alibi(Flash_fwd_params &params, Tensor_t alibi_slopes, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    // TORCH_CHECK(!alibi_slopes.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes != NULL) {
        auto alibi_slopes_ = alibi_slopes;
        // TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        // CHECK_DEVICE(alibi_slopes);
        // TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        // TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = get_tensor_data(alibi_slopes_);
        params.alibi_slopes_batch_stride = get_tensor_dims(alibi_slopes_) == 2 ? get_tensor_stride(alibi_slopes_, 0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
std::vector<int64_t>
get_attn_mask_stride(std::vector<int64_t> &mask_shape, std::vector<int64_t> &score_shape) {

    CHECK_THROW(score_shape.size() == 4, "score_shape must be 4-dim");
    CHECK_THROW(1 <= mask_shape.size() && mask_shape.size() <= score_shape.size(), "attn_mask should have dim less than score_mask and at least have 1 dim");

    int64_t accum_stride = 1; // dim-0 * ... * dim-n ,should be stride of dim-(n+1)
    std::vector<int64_t> mask_stride = {0, 0, 0, 0};

    bool is_shape_valid = true;
    int seqlen_k_padded = ALIGNUP(score_shape[3], 4);
    for(int i = mask_shape.size() - 1; i >= 0; i--) {
        if (i <= 1) {
            // batch and nheads dim
            if (mask_shape[i] == 1) {
                mask_stride[i] = 0;
            } else if (score_shape[i] % mask_shape[i] == 0) {
                mask_stride[i] = accum_stride;
                accum_stride *= mask_shape[i];
            } else {
                is_shape_valid = false;
                break;
            }
        } else {
            // seqlenq and seqlenk dim
            if (mask_shape[i] == 1) {
                mask_stride[i] = 0;
            } else if (score_shape[i] == mask_shape[i] || (i == 3 && mask_shape[i] == seqlen_k_padded)) {
                mask_stride[i] = accum_stride;
                accum_stride *= mask_shape[i];
            } else {
                is_shape_valid = false;
                break;
            }
        }
    }

    CHECK_THROW(is_shape_valid, "the attn_mask shape is not support, please check it!");

    return mask_stride;
}

void set_params_attn_mask(Flash_fwd_params &params, Tensor_t attn_mask, int batch_size, int num_heads, int seqlen_q, int seqlen_k) {
    if (attn_mask == nullptr) {
        params.has_attn_mask = false;
    } else {
        auto attn_mask_ = attn_mask;
        // TORCH_CHECK(seqlen_q == seqlen_k, "seqlen_q should equal to seqlen_k when attn_mask is true");
        // TORCH_CHECK((attn_mask.dtype() == torch::kBFloat16) || (attn_mask.dtype() == torch::kFloat16), "attn_mask must have dtype fp16/bf16");
        // CHECK_DEVICE(attn_mask);
        // TORCH_CHECK(attn_mask.stride(-1) == 1, "attn_mask tensor must have contiguous last dimension");
        // CHECK_SHAPE(attn_mask, batch_size, seqlen_q);
        int attn_mask_dim = get_tensor_dims(attn_mask);
        int seqlen_k_rounded = (seqlen_k + 3) / 4 * 4;
        if (seqlen_k_rounded == get_tensor_size(attn_mask, attn_mask_dim - 1)) {
            // the attn_mask last dim will be padded to multiples of 4
            seqlen_k = seqlen_k_rounded;
        }
        std::vector<int64_t> score_shape = {batch_size, num_heads, seqlen_q, seqlen_k};
        std::vector<int64_t> mask_shape = {1, 1, 1, 1};
        for (int i = attn_mask_dim - 1, j = 3; i >= 0; i--, j--) {
            mask_shape[j] = get_tensor_size(attn_mask, i);
        }
        auto mask_stride = get_attn_mask_stride(mask_shape, score_shape);


        params.has_attn_mask = true;
        params.attn_mask_ptr = get_tensor_data(attn_mask_);
        params.attn_mask_batch_stride = mask_stride[0];
        params.attn_mask_nheads_stride = mask_stride[1];
        params.attn_mask_row_stride = mask_stride[2];
        params.attn_mask_col_stride = mask_stride[3];
        params.attn_mask_batch_shape = mask_shape[0];
        params.attn_mask_nheads_shape = mask_shape[1];
        params.attn_mask_row_shape = mask_shape[2];
        params.attn_mask_col_shape = mask_shape[3];
    }
}
