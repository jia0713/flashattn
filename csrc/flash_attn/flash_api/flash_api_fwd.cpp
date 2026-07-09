#include "flash_api.h"
#include "flash_parameter.h"
#include "flash_parameter_utils.h"
#include "run_mha.h"
#include "host_utils.h"
#include "flash_splitkv.h"

using namespace mcFlashAttn;

/*---------------------------------------training apis-------------------------------------------------------*/
/*
// score ->[bs,      head_num,      q,      k  ]
// mask -> [bs_mask, head_num_mask, q_mask, k_mask]
// Mask shape should satisfy these rules
// 1. bs % bs_mask == 0
// 2. head_num % head_num_mask == 0
// 3. q_mask == 1 or q_mask == q
// 4. k_mask == 1 or k_mask == k or k_mask == (k + 3) / 4 * 4 (align k to multiples of 4)
*/
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        c10::optional<at::Tensor> &attn_mask_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &s_aux_, // (n_heads)
        bool return_max_logit
        ) {


    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads_ori = sizes[2];

    int num_heads = num_heads_ori;
    const int head_size_og = sizes[3];
    const int head_size_og_v = v.sizes()[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension <= 512");
    TORCH_CHECK(head_size_og_v <= 512, "FlashAttention forward only supports head dimension <= 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(head_size_og >= head_size_og_v, "Head dimension of query/key must greater or equal to head dimension in query");


    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    const uint32_t ngroups = seqlenq_ngroups_swapped ? num_heads / num_heads_k : 1;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og_v);

    at::Tensor q_padded, k_padded, v_padded, attn_mask_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_og_v % 8 != 0) {
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_v % 8}));
    } else {
        v_padded = v;
    }

    if (attn_mask_.has_value()) {
        auto attn_mask_shape = attn_mask_.value().sizes();
        int attn_mask_last_dim = attn_mask_shape[attn_mask_shape.size() - 1];
        if (seqlen_k % 4 != 0 && attn_mask_last_dim == seqlen_k) {
            // we padded attn_mask for merge ldg_u16 -> ldg_b64 when load attn_mask
            // we only pad it when attn_mask_last_dim == seqlen_k to load right data
            attn_mask_padded = torch::nn::functional::pad(attn_mask_.value(), torch::nn::functional::PadFuncOptions({0, 4 - seqlen_k % 4}));
        } else {
            attn_mask_padded = attn_mask_.value();
        }
    }

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og_v);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size_og_v}).transpose(1, 2);
        }
    } else {
        out = torch::empty({ batch_size, seqlen_q, num_heads, head_size_og_v }, opts);
    }
    if (head_size_og_v % 8 != 0) {
        out = torch::nn::functional::pad(out, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_v % 8}));
    }
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size, 32) == 224 ? 256 : round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_og_v, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32) == 224 ? 256 : round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    const bool is_MLA = head_size_rounded == 192 && head_size_v_rounded == 128;
    TORCH_CHECK(is_MLA || !return_max_logit, "we only support return_max_logit in MLA(dim 192,128) now");

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor max_logit;
    if (return_max_logit) {
        max_logit = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    }
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*seqlenq_ngroups_swapped=*/false,
                     /*unpadded_lse=*/false,
                     head_size_v,
                     head_size_v_rounded);
    params.arch = arch;
    params.ngroups = ngroups;
    params.max_logit_ptr = return_max_logit ? max_logit.data_ptr() : nullptr;


    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (p_dropout > 0.0)  {
        get_philox_state(gen_, rng_state, counter_offset);
        set_params_rng_state(params, rng_state);
    }

    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        TORCH_CHECK(s_aux.scalar_type() == at::ScalarType::BFloat16,
            "We only support bf16 dtype for S extra.");
        TORCH_CHECK(head_size_rounded == 64 && head_size_v_rounded == 64,
            "We only support head_dim 64 for S extra.");
        TORCH_CHECK(dprops.major == 10, "We only support S extra in arch xcore1000 series now.");
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads_ori);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    TORCH_CHECK(!(alibi_slopes_.has_value() && attn_mask_.has_value()), "alibi_slopes_ and attn_mask_ will not present together");
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
    if (attn_mask_.has_value()) {
        set_params_attn_mask(params, is_causal, attn_mask_padded,
                            batch_size, num_heads, seqlen_q, seqlen_k);
    } else {
        params.has_attn_mask = false;
    }

    bool force_split_kernel = false;
    int num_splits = 0;
    if (params.d == 80 && params.seqlen_q == 131 && params.seqlen_k == 131 && params.b == 128 && params.p_dropout >= 1.f) {
        force_split_kernel = true;
    }
    if (params.d == 80 && params.seqlen_q == 1024 && params.seqlen_k == 77) {
        force_split_kernel = true;
    }
    if (params.d == 160 && params.seqlen_q == 64 && params.seqlen_k == 77) {
        num_splits = 1;
    }

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (attn_mask_.has_value() || head_size != head_size_v){
            params.num_splits = 1;
            auto splitkv_accum = malloc_accum_by_numsplits(params);
            run_mha_fwd(params, stream);
            (void)splitkv_accum;
        }else {
            compute_params_numsplits(params, num_splits, force_split_kernel);
            auto splitkv_accum = malloc_accum_by_numsplits(params);
            run_mha_fwd(params, stream, force_split_kernel);
            (void)splitkv_accum;
        }
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og_v % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og_v)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og_v});
        out_padded = out_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og_v});
        q_padded = q_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        if (return_max_logit) {
            max_logit = max_logit.reshape({batch_size, num_heads_k * seqlen_q, 1});
        }
    }
    if (return_max_logit) {
        // reduce max in batch and seqlen_q dimension
        max_logit = torch::amax(max_logit, {0, 2});
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state, attn_mask_padded, max_logit};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               c10::optional<const at::Tensor> &leftpad_k_, // batch_size
               c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               c10::optional<at::Generator> gen_,
               c10::optional<at::Tensor> &s_aux_, // (n_heads)
               bool return_max_logit
               ) {

    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads_ori = sizes[1];
    int num_heads = num_heads_ori;
    const int head_size_og = sizes[2];
    const int head_size_og_v = paged_KV ? v.size(3) : v.size(2);
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);


    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 0 : k.size(1);
    TORCH_CHECK(!paged_KV || (page_block_size > 0 && (page_block_size & (page_block_size - 1)) == 0), "Paged KV cache block size must be a power of 2!");

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_.has_value() && head_size_og != head_size_og_v;
    const uint32_t ngroups = seqlenq_ngroups_swapped ? num_heads / num_heads_k : 1;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size_og});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension <= 256");
    TORCH_CHECK(head_size_og_v <= 512, "FlashAttention forward only supports head dimension <= 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size_og_v);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_og_v);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
    }
    if (head_size_og_v % 8 != 0) {
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_v % 8}));
    } else {
        v_padded = v;
    }

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og_v);
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size_og_v);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size_og_v}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size_og_v});
        }
    } else {
        out = torch::empty({ sizes[0], sizes[1], head_size_og_v }, opts);
    }
    if (head_size_og_v % 8 != 0) {
        out = torch::nn::functional::pad(out, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_v % 8}));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    // we do not have headdim=224 kernel, padding head_size_rounded to 256 to support theses headdim
    const int head_size_rounded = round_multiple(head_size, 32) == 224 ? 256 : round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_og_v, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32) == 224 ? 256 : round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    TORCH_CHECK(!(page_block_size == 1 && head_size_rounded != 128), "page_size = 1 only support in head_dim128");
    const bool is_MLA = head_size_rounded == 192 && head_size_v_rounded == 128;
    TORCH_CHECK(is_MLA || !return_max_logit, "we only support return_max_logit in MLA(dim 192,128) now");

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    bool unpadded_lse;
    at::Tensor softmax_lse;

        // varlen uses unpadded lse, shape of softmax_lse is (num_heads x total_q)
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
        unpadded_lse = true;
    at::Tensor max_logit;
    if (return_max_logit) {
        max_logit = torch::full({batch_size, num_heads, max_seqlen_q}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    }

    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/unpadded_lse,
                     head_size_v,
                     head_size_v_rounded);

    params.total_q = total_q;
    params.arch = arch;
    params.ngroups = ngroups;
    params.max_logit_ptr = return_max_logit ? max_logit.data_ptr() : nullptr;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k_padded.stride(0);
        params.v_batch_stride = v_padded.stride(0);
    }
    params.page_block_size = page_block_size;

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        TORCH_CHECK(s_aux.scalar_type() == at::ScalarType::BFloat16,
            "We only support bf16 dtype for S extra.");
        TORCH_CHECK(head_size_rounded == 64 && head_size_v_rounded == 64,
            "We only support head_dim 64 for S extra.");
        TORCH_CHECK(dprops.major == 10, "We only support S extra in arch xcore1000 series now.");
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads_ori);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto rng_state = torch::empty({2}, torch::kInt64);

    if (p_dropout > 0.0)  {
        get_philox_state(gen_, rng_state, counter_offset);
        set_params_rng_state(params, rng_state);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (seqlenq_ngroups_swapped) {
            // Only apply split-k for decoding
            compute_params_numsplits(params, 0, paged_KV);
            auto splitkv_accum = malloc_accum_by_numsplits(params);
            run_mha_fwd(params, stream, paged_KV);
            (void)splitkv_accum;
        }else {
            compute_params_numsplits(params, 1, paged_KV);
            run_mha_fwd(params, stream, paged_KV);
        }
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og_v % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og_v)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size_og};
        int64_t size_before_v[] = {batch_size, max_seqlen_q, num_heads_k, head_size_og_v};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size_og};
        int64_t size_after_v[] = {batch_size, num_heads_k * max_seqlen_q, head_size_og_v};
        out = out.reshape(size_before_v).transpose(1, 2).reshape(size_after_v);
        out_padded = out_padded.reshape(size_before_v).transpose(1, 2).reshape(size_after_v);
        q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
        if (return_max_logit) {
            max_logit = max_logit.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
        }
    }
    if (return_max_logit) {
        // reduce max in batch and seqlen_q dimension
        max_logit = torch::amax(max_logit, {0, 2});
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state, max_logit};
}
