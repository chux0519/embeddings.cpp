# GGML Overlay Patches

`ggml/` should track llama.cpp upstream as closely as possible. Local changes
belong here only when they are required by embedding correctness or measured
Snowflake performance.

## Current Overlay Areas

These local changes exist on top of the vendored `ggml/` tree and must be
reviewed during every upstream refresh.

### GTE/Snowflake Fused Ops

Files:

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cpu/ops.h`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/include/ggml-rpc.h`

Ops:

- `GGML_OP_GTE_QKV_ROPE`
- `GGML_OP_GTE_CLS_POOL`
- `GGML_OP_GTE_GEGLU`
- `GGML_OP_GTE_NORM_AFFINE`
- `GGML_OP_GTE_LINEAR`

Purpose:

- fuse packed QKV split with NeoX RoPE and flash-attention layout;
- gather CLS embeddings without scalar graph overhead;
- fuse GEGLU;
- fuse layer norm with affine parameters;
- provide a Snowflake-oriented fp32 linear path with four-token SIMD blocking.

Porting rule:

- keep only if the model graph still uses these helpers and benchmarks show a
  material win over equivalent upstream primitives;
- update op count tables and RPC protocol guard together;
- rerun correctness, batch-vs-single alignment, and Snowflake performance.

### Llamafile Fallback Guard

File:

- `ggml/src/ggml-cpu/ggml-cpu.c`

Purpose:

- avoid returning early from `ggml_compute_forward_mul_mat` when one batched
  llamafile sgemm call fails and a later slice would otherwise be skipped.

Porting rule:

- drop if upstream already has equivalent all-slices fallback behavior;
- otherwise preserve the guard.

## Refresh Checklist

1. Inspect the source checkout:

   ```bash
   uv run scripts/ggml_upstream.py inspect-source --source /path/to/llama.cpp
   ```

2. Import from a pinned upstream ref:

   ```bash
   uv run scripts/ggml_upstream.py import \
     --source /path/to/llama.cpp \
     --ref <tag-or-branch> \
     --commit <exact-commit>
   ```

3. Reapply only justified overlay patches.
4. Build and run correctness.
5. Run Snowflake fp32 and quantized benchmarks before updating README claims.
