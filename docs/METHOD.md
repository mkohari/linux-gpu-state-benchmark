# Method and claim boundaries

This document specifies the Transformers prefix-KV experiment (`run`/`smoke`).
The new synthetic numerical experiment and sequential starter suite are defined
in [WORKLOADS.md](WORKLOADS.md). The optional HTTP serving probe has its own
[scope and timing definitions](OPTIONAL-VLLM.md); it does not measure KV disk restore.

## What prefill and saved state mean

Prefill computes a model's intermediate attention key/value tensors for input
tokens. Restoring KV lets the model reuse those prefix calculations when the
identical token sequence appears again. It still processes the new question
suffix and calculates every new output token. We never save an answer and use it
instead of running the model.

Each case contains a distinct synthetic prefix and a fixed question suffix. The
exact token arrays are saved in `cases.json`. Recompute and restore process
identical token sequences, use the same model, dtype, attention implementation
and prefill chunk size, and generate the same fixed number of greedy tokens.
No sampling, neural retrieval, reranking, portfolio pricing or token-server
concurrency is part of this experiment.

## Warmup and population

The chosen local model/tokenizer files are fully hashed outside measurement.
Custom model-repository code is disabled. Model weights remain resident in the
chosen GPU. An untimed case warms prefix, suffix and single-token decode shapes.

For every case, compute the prefix KV, hash all tensors and write that state.
Record prefill and storage-write costs separately. Then process the suffix and
generate reference output token IDs. Save the prefix fingerprint, model identity
and expected output. Reference output is only a correctness comparison, never
the measured answer path. Prefix preparation is a real up-front cost and must
be considered when estimating the benefit of reuse.

## Explicit state release

Before every measured request, release prior prefix-state references, collect
Python garbage, synchronize and release unused PyTorch GPU allocator blocks.
The benchmark retains neither prefix KV nor reusable host payloads between
requests. Input token IDs, model weights and small control metadata remain.

This is a controlled state-release microbenchmark. It does not cause or audit
natural serving-cache LRU evictions. It does not require the on-disk corpus to
exceed the GPU's physical memory. It cannot establish eviction counts or serving
cache hit rates. Do not describe it as the previous 512-context capacity test.

## Timing boundaries

| Metric | Starts | Ends |
|---|---|---|
| Recompute first token | Immediately before transferring/processing prefix token IDs | Same suffix processed and first newly chosen token available to Python, after GPU synchronization |
| Restore first token | Immediately before backend read, including metadata access | State read/purged/validated/deserialized/transferred, same suffix processed and first new token available after synchronization |
| Completion | Same request start | All configured output tokens available after synchronization |
| Validation | After fixed-count generation completes | Output IDs and every full prefix KV tensor hash checked |

Filesystem restore includes payload SHA-256, deserialization and CPU-to-GPU copy.
Its bytes temporarily occupy application RAM. Full KV hash diagnostics separately
copy tensors to CPU after generation; their cost is outside first-token and
completion timings. This boundary is visible in every report. Population,
initial model load and the explicit allocator release are outside measured
requests. Source and runtime initialization are not steady-state inference.

The engine issues prefill in chunks (`--chunk-tokens`, default 2,048), with the
same public model-forward path during population and recomputation. It requests
only the last logit row per forward. This is a synchronous Python reference
implementation, not a claim of parity with vLLM, TensorRT-LLM or another serving
engine. Changing chunking or kernels can materially change the ratio.

## Filesystem cache control

The supplied backend synchronously writes each new payload with `fsync`.
Before and after each payload read/write, targeted `POSIX_FADV_DONTNEED` and
`mincore` verify zero resident file pages. File pages may populate during I/O;
the audit records that population. The purge has bounded retries and its cost is
part of the request. A failure stops the measurement.

No machine-wide cache flush, storage-controller cache flush, drive-cold claim,
cuFile/GDS claim or whole-application RAM-free claim is made. Metadata files are
control data. Third-party filesystem gateways can retain additional caches that
this local-page test cannot see. Characterize them separately. Original data is
never overwritten; state goes to a new owned directory. Results/state remain
available until the owner removes those exact run directories.

## Sample order and acceptance

For each prefix and repetition, run one request through each path. Shuffle the
complete balanced job list with a recorded seed. Default: eight prefixes and
three repetitions, 24 samples per path. p50 is the median; p95 uses linear
interpolation. These are descriptive samples within one execution, not confidence
intervals or an independent-replication study. Repeat complete runs on an otherwise
controlled host and retain both favorable and unfavorable results.

Every request must match the reference token IDs and full KV tensor hashes.
The filesystem measurement additionally requires nonzero actual read/write bytes
and zero file-page residency before/after every operation. Failing runs produce
`failure.json`; they do not produce an accepted summary. CPU smoke reports cannot
assert GPU performance, even if their measured timings happen to look favorable.

Generated token equivalence checks computation reuse, not general answer quality,
financial accuracy or citation faithfulness. The fixed output count deliberately
continues after EOS so both arms do identical declared work.

## Interpreting speedup

`speedup = recompute first-token p50 / restore first-token p50`

A hypothetical 350 ms recompute median and 100 ms restore median would mean
3.5x. **Those are arithmetic illustrations, not measurements.** The source does
not target or tune acceptance around that number. Short prefixes, very fast
GPUs, large KV tensors, CPU staging, slower storage or different model kernels
can make restore slower than recomputation. This is a legitimate result.

First use must pay the compute and save costs. The benefit depends on enough
subsequent reuse to recover those costs. No end-to-end application cost, energy
saving, maximum throughput or benefit for every request follows from this ratio.

## Evidence to retain when publishing

Keep the complete run directory and I/O audit, source commit, installed dependency
versions, exact model snapshot hashes, GPU identity/memory, driver version, kernel,
filesystem and mount configuration, storage hardware, dtype, attention kernel,
chunk size, token counts, repetition order and counts. Record background load and
storage topology yourself. The program captures many of these automatically;
operator context and drive/controller details require additional evidence.
GPU memory is reported generically because its physical technology depends on
the card. Do not label GDDR memory as HBM.

No historical vendor benchmark values or datasets are included. Results from an
added backend are new measurements with their own implementation and evidence.
