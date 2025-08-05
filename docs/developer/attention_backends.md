# Contributing Attention Backends

vLLM supports pluggable attention implementations. Custom backends can be
registered without modifying the core source tree by exposing an entry point or
providing a configuration file.

## Implementing a Backend

Backends should subclass
[`vllm.attention.backends.base.AttentionBackend`](../../vllm/attention/backends/base.py)
and implement:

* `forward` – executes the attention kernel and optionally updates the KV
  cache.
* `get_kv_cache_shape` – returns the shape of the backend's KV cache tensor.
* `swap_blocks` and `copy_blocks` – utilities for moving cache blocks.

Existing CUDA and ROCm implementations (`FlashAttentionBackend` and
`ROCmFlashAttentionBackend`) provide reference implementations.

## Registering a Backend

Backends can be discovered in two ways:

1. **Python entry points** – expose the backend class in the
   `vllm.attention_backends` entry point group. The entry point name becomes the
   backend identifier.
2. **Configuration file** – set the environment variable
   `VLLM_ATTENTION_BACKENDS_CONFIG` to the path of a JSON file mapping backend
   names to import paths, for example:

   ```json
   { "my_backend": "my_pkg.module:MyBackend" }
   ```

After registering, select the backend by setting `VLLM_ATTENTION_BACKEND` to the
backend name or by calling `get_attn_backend` with that name.

## Testing

Ensure that new backends conform to the interface and include tests demonstrating
correctness.
