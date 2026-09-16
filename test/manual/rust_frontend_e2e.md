# Rust frontend: whole-stack validation

`test_rust_frontend_e2e.py` is an opt-in, one-GPU test, not a benchmark or a
Python-frontend comparison. It launches a real model through the source-built
Rust frontend. It is manual because the before/after run requires two source
checkouts and an explicitly pinned local model. Missing prerequisites fail;
there is no missing-extension skip and test-method retries are disabled. The
repository's launch helper retains its offline-to-online startup fallback;
inspect startup logs as well as the test result.

The stack builds on Rain Jiang's multi-protocol prototype
([sglang#36923](https://github.com/sgl-project/sglang/pull/36923)). This change adds
validation only: no runtime capabilities, protocol changes, or production fixes.

## Coverage

- Native text/token-ID generation and OpenAI completions/chat, over HTTP and
  runtime.v1 gRPC, each streaming and non-streaming (16 dual-server cases).
- JSON/SSE/protobuf terminal framing; output, finish reasons, and token usage.
- Health, model/server information, model listing, and detokenization.
- Unsupported RPC/feature and invalid supported-request status codes.
- Concurrent clients with different prompts to catch response cross-delivery.
- Native and two-choice OpenAI cancellation, confirmed by scheduler abort logs,
  followed by a successful request. Natural completion is not cancellation.
- Shutdown with live streams, child-process and listener cleanup, restart on the
  same ports, and idle shutdown. This checks bounded cancellation, not draining.

The transport readers in `_rust_frontend_e2e_client.py` reject truncated streams.
Native output is cumulative; it must not be concatenated like OpenAI deltas.
`stream=false` still uses a server-streaming gRPC method.

## Prerequisites

Use one idle GPU, a Python environment matching `python/pyproject.toml`, the
Rust workspace toolchain/build dependencies, and `grpcio-tools`. Use the Python
interpreter named in the installed `sglang` entrypoint's shebang. The test forces
the child process to import the selected source checkout.

Use an immutable local model snapshot with a text chat template, such as the
repository's small-model fixture, Llama-3.2-1B-Instruct. Keep model, GPU, package
versions, and launch settings identical across runs. Do not pass API keys,
SMG/legacy gRPC flags, or skip-tokenizer options.

Set `SGLANG_CACHE_DIR` on generated-storage: the extension loader puts build
outputs there, independently of `CARGO_TARGET_DIR`. Also set
`SGLANG_JIT_CACHE_DIR` and `TORCH_EXTENSIONS_DIR`; these caches do not inherit
`SGLANG_CACHE_DIR`. Keep other framework/compiler caches and temporary files on
the same appropriate storage. Generated client bindings belong to each run's
output directory, never the source tree.

## Run the three configurations

Keep the new test files in the final checkout while switching **imports** with
`PYTHONPATH`; do not copy tests into or modify the baseline checkout.

```bash
# Set these to your checkouts, immutable model snapshot and artifact directory.
export SGLANG_E2E_MODEL_PATH=/path/to/model/snapshots/COMMIT
export SGLANG_CACHE_DIR=/path/to/build-cache/sglang
export SGLANG_JIT_CACHE_DIR=/path/to/build-cache/sglang-jit
export TORCH_EXTENSIONS_DIR=/path/to/build-cache/torch-extensions
test_file=/path/to/final/test/manual/test_rust_frontend_e2e.py
baseline=/path/to/pre-stack-checkout
final=/path/to/final
artifacts=/path/to/validation-results

# A: pre-stack Rust HTTP.
PYTHONPATH="$baseline/python" SGLANG_E2E_HTTP_ONLY=1 \
  SGLANG_E2E_OUTPUT_DIR="$artifacts/baseline" python "$test_file" -v

# B: final Rust HTTP, gRPC disabled.
PYTHONPATH="$final/python" SGLANG_E2E_HTTP_ONLY=1 \
  SGLANG_E2E_REFERENCE="$artifacts/baseline/report.json" \
  SGLANG_E2E_OUTPUT_DIR="$artifacts/http-only" python "$test_file" -v

# C: final Rust HTTP + gRPC, against the same HTTP reference.
PYTHONPATH="$final/python" SGLANG_E2E_HTTP_ONLY=0 \
  SGLANG_E2E_REFERENCE="$artifacts/baseline/report.json" \
  SGLANG_E2E_OUTPUT_DIR="$artifacts/dual" python "$test_file" -v
```

Each output directory must be new. Defaults are localhost HTTP 30000 and gRPC
50051; override with `SGLANG_E2E_HTTP_PORT` / `SGLANG_E2E_GRPC_PORT` if needed.
For read-only container mounts without Git metadata, supply the verified source
SHA through `SGLANG_E2E_SOURCE_REVISION`. Do not use that to relabel different code.

Each run retains `server.log`, generated client bindings (dual mode), and
`report.json`: source/extension identity, model, packages, GPU, launch arguments,
HTTP snapshots, completed tests, shutdown timing, and overall pass status.
Preserve unittest output and its exit status too. `passed` requires all six test
methods and successful cleanup; a targeted partial run is not a full pass.

HTTP snapshots preserve full non-streaming payloads except generated IDs and
timestamps/latency. Streaming comparisons use reconstructed output, finish
reasons, and usage, not timing-dependent chunk boundaries. The suite compares
the selected requests, not every possible HTTP option. Keep the existing HTTP
regression suites and direct Rust lifecycle tests as complementary coverage.

## Validation record

Source baseline: `3eeb7d37f930e1386cc7e53674844887e961ef71` (before step [2]).
Restacked tip: `eaa116f541e0f95fd1dd5c95e7f30cf8da9d48a4` (through step [7],
including its OpenAI error-semantics fix).

Run environment (2026-09-16 UTC):

- Model: Llama-3.2-1B-Instruct, snapshot
  `9213176726f574b556790deb65791e0c5aa438b6`.
- GPU: one RTX 6000 Ada, driver 590.48.01.
- Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.12.1,
  sglang-kernel 0.4.7, grpcio/grpcio-tools 1.83.1.
- Isolated local container derived from `lmsysorg/sglang:v0.5.19-cu130-runtime`,
  with the kernel dependency updated, protoc installed, and Rust 1.93.1 mounted.
  Both frontend extensions were release-built from their recorded source SHAs;
  the baked frontend was not used. Model and source mounts were read-only.

| Configuration | Result |
| --- | --- |
| Pre-stack HTTP | All six test groups passed |
| Stack-tip HTTP only | All six test groups passed |
| Stack-tip HTTP + gRPC | All six test groups passed |

Initial validation at `a478144841` passed generation, rejection checks,
concurrency, cancellation, and shutdown, but caught a shared server-info
regression. The baseline returned HTTP 200; the stack returned HTTP 500 and
gRPC INTERNAL. Those failed reports remain valid records of the pre-fix code.

The post-fix runs use fresh artifact directories: `baseline-server-info`,
`final-http-server-info`, and `final-dual-server-info`. Each retains
`report.json`, `server.log`, and console output. The restacked Rust workspace
passed all 366 tests, formatting, and Clippy with warnings denied.

All three test processes exited successfully and report `passed: true`. The
dual-server run passed all 16 generation cases, metadata/health, rejection,
concurrency, cancellation, and shutdown/restart checks. Both stack runs matched
all nine baseline HTTP snapshots (eight generation cases and one invalid request).
HTTP `/server_info` and gRPC `GetServerInfo` now succeed. Active-stream and idle
shutdown each completed in approximately 0.22 seconds in the dual-server run.

Both baseline and stack runs emit a Python multiprocessing semaphore-cleanup
warning during shutdown. The process-exit, listener-close, and same-port restart
checks still pass; these results are not a claim of warning-free execution.

### Regression fixed in step [3]: server-info decoding

The real scheduler's KV-cache memory measurement is a NumPy scalar. The existing
Python-to-Rust MessagePack bridge stringifies it, but step [3]'s original
`MemoryUsage.kvcache: Option<f64>` rejected that string. Fix `ec516226ea` changes
only this metric to an untagged number-or-string type, preserving the existing
public representation instead of coercing strings into numbers. Other metrics
and the private-field allowlist are unchanged.

The existing HTTP route test now injects both MessagePack forms and checks their
exact JSON values/types, alongside its private-field filtering assertions. The
fix lives in the step [3] branch, not this validation PR; downstream commits
were replayed unchanged. No assertions were skipped or marked expected-failure.
