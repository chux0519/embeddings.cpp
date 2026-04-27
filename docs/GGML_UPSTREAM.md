# GGML Upstream Tracking

`ggml/` is vendored source, not a git submodule. Keep the upstream identity in
`.vendor/ggml-upstream.json` so future updates are repeatable.

## Upstream

- Repository: `https://github.com/ggml-org/llama.cpp`
- Subdirectory: `ggml`
- Local path: `ggml/`

The current vendored tree is recorded in `.vendor/ggml-upstream.json` as
llama.cpp `b8833` at commit
`45cac7ca703fb9085eae62b9121fca01d20177f6`, imported in
`f5bb1543e899bacc11b90330632adfdf6cffd811` on 2026-04-25.

The latest upstream release checked for this tracking pass was llama.cpp
`b8833` (`45cac7c`) on 2026-04-24.

## Update Workflow

The preferred policy is upstream-first: import `ggml/` from a pinned llama.cpp
release tag, then reapply only the local embedding-specific patches that still
matter. Do not maintain a broad fork unless an upstream implementation is
measurably worse for this project.

1. Pick an upstream llama.cpp tag or commit.
2. Resolve and record the exact upstream commit:

   ```bash
   uv run scripts/ggml_upstream.py set --ref b8833 --commit 45cac7c
   ```

3. Import `ggml/` from a clean local llama.cpp checkout:

   ```bash
   uv run scripts/ggml_upstream.py import --source /path/to/llama.cpp --ref b8833 --commit 45cac7c
   ```

4. Reapply or port local embedding-specific changes.
5. Run correctness and Snowflake benchmarks before committing release claims.

Useful checks:

```bash
uv run scripts/ggml_upstream.py status
git diff -- ggml
```

## Local Changes To Preserve

When updating upstream, treat these local areas as overlay patches. Reapply them
only when the upstream tree does not already provide equivalent behavior or
performance:

- GTE/Snowflake fused linear operation and runtime controls.
- CPU fast-path behavior used by `src/gte.cpp`.
- Any embedding-specific ggml op registration or serialization changes.
- Snowflake correctness tests and benchmark scripts.
- Browser WebGPU fallback compatibility for GTE/Snowflake graphs until the
  missing WebGPU kernels are implemented upstream or in our overlay.

The current overlay inventory lives in `docs/GGML_OVERLAY_PATCHES.md`.

Do not silently overwrite local performance patches. If a local patch is
replaced by an upstream implementation, record that in the commit message.

## Switch Criteria

Switch directly to upstream `ggml/` when:

- correctness passes for README-supported models;
- Snowflake fp32 and quantized benchmarks are not materially worse, or the
  regression is understood and accepted;
- local embedding patches are either ported, dropped with evidence, or replaced
  by upstream code;
- the provenance file records the exact upstream tag and commit.
