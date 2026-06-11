# Head-Dim YAML Profiling

`profile_hdim512_yaml_standalone.py` profiles YAML-captured paged causal
attention cases without depending on the local flash-attn or vLLM source tree.

Runtime dependencies:

- Required: `torch`, `triton`
- Optional: `PyYAML`
- Optional: installed `flash_attn` Python package for the MACA flash-attn backend

The script embeds the Triton unified-attention implementation it needs. It does
not import vLLM. It may import an installed `flash_attn` package, but it does
not import this repository's flash-attn source code.

## Supported Cases

The expected YAML format is a mapping of problem names to problem fields. The
`api` field is recorded in the CSV but is not used to decide Triton support;
support is based on the paged attention shape fields.

```yaml
problem_1:
  api: _flash_attn_varlen_forward
  paged_kv: true
  batch_size: 1
  cu_seqlens_q: [0, 2044]
  cu_seqlens_kv: [0, 2183]
  seqlens_q: [2044]
  seqlens_kv: [2183]
  num_heads_q: 8
  num_heads_kv: 1
  head_dim: 512
  head_dim_v: 512
  paged_block_size: 32
  paged_num_blocks: 92659
  causal: true
```

Multiple top-level problems are supported. Each problem is emitted as one row
per backend in the output CSV.

The required fields for Triton unified profiling are:

```text
batch_size
seqlens_q
seqlens_kv
num_heads_q
num_heads_kv
head_dim
paged_block_size
```

`cu_seqlens_q` and `cu_seqlens_kv` are preferred. If they are absent, the script
derives them from `seqlens_q` and `seqlens_kv`.

`paged_kv: true` is preferred. If `paged_kv` is absent but `paged_block_size` is
present, the case is treated as paged.

For `api: flash_attn_with_kvcache` decode cases, the script treats
`max_seqlen_q` and `max_seqlen_kv` as the per-sequence query and KV lengths
when the YAML does not provide per-batch `cu_seqlens_*`.

The standalone profiler supports both `head_dim=256` and `head_dim=512`, as
long as the selected backend supports that shape. The current
`flash_attn_with_kvcache` automatic Triton-unified route in the library is still
limited to `head_dim=512`; that is separate from this standalone profiler.

Current practical backend target:

- MACA / MetaX: `flash_attn_varlen`
- CUDA / NVIDIA: `triton_unified_attention`

MACA Triton unified profiling is intentionally not enabled by default.

## Commands

Run on MACA. Auto mode selects installed `flash_attn`:

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:${LD_LIBRARY_PATH}
python profile_hdim512_yaml_standalone.py \
  --yaml headdim512.yaml \
  --csv maca_hdim_yaml_profile.csv
```

Run on CUDA / NVIDIA. Auto mode selects Triton unified:

```bash
python profile_hdim512_yaml_standalone.py \
  --yaml headdim512.yaml \
  --csv a100_hdim_yaml_profile.csv
```

Explicit backend selection:

```bash
python profile_hdim512_yaml_standalone.py --yaml headdim512.yaml --backends flash
python profile_hdim512_yaml_standalone.py --yaml headdim512.yaml --backends triton
python profile_hdim512_yaml_standalone.py --yaml headdim512.yaml --backends both
```

If CUDA has only `torch` and `triton`, keep `--backends triton` or default
`auto`. Flash-attn will not be required.

## Output

The CSV includes the case identity, shape, timing, TFLOPs, and correctness:

```text
case,backend,dtype,device_name,device_capability,api,count,hash_code,
batch_size,batch_size_c,total_seqlens_q,total_seqlens_kv,
max_seqlen_q,max_seqlen_kv,seqlens_q,seqlens_kv,
num_heads_q,num_heads_kv,head_dim,head_dim_v,
paged_block_size,paged_num_blocks,causal,window_left,window_right,softcap,
time_ms,tflops,max_abs,max_rel,allclose,status
```

TFLOPs are computed with:

```text
FLOPs = 4 * num_heads_q * head_dim * sum(seqlen_q[i] * seqlen_kv[i])
TFLOPs = FLOPs / time_seconds / 1e12
```

The reference path is a torch implementation generated from the same YAML case.
Backends are validated against that reference before timing results are reported.
