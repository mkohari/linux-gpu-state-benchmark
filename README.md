# Linux GPU state benchmark

**Created, authored and maintained by [Moiz Kohari](https://github.com/mkohari).**
Copyright © 2026 Moiz Kohari. Licensed under the [MIT License](LICENSE).

A new, vendor-neutral reference implementation. Compare the
time to start a fresh answer after either recomputing a text prefix on a GPU or
restoring its saved model KV state from a Linux filesystem. A separately authored
synthetic Monte Carlo experiment makes the same comparison for simulated price paths.

The only included storage backend is Linux filesystem I/O. There is **no DDN/FSI
source, copied numerical kernel, vendor SDK, native plugin, cloud client, model weight,
private build dependency, historical benchmark dataset or commercial EULA** in
this repository. New source is MIT licensed. Public Python packages and separately
obtained model weights remain subject to their own licenses.

## New users: start here

**[Beginner walkthrough: setup → demo → model → GPU tests → results](docs/GETTING-STARTED.md)**

After cloning this repository, on Linux:

```bash
bash scripts/setup.sh
source .venv/bin/activate
gpu-state-bench demo              # No trained model needed; correctness only
gpu-state-bench doctor            # Check GPU, dependencies and storage
gpu-state-bench download-model    # Explicit ~3 GiB public model download
gpu-state-bench start             # Both GPU experiments; one results page
```

For a machine without a GPU, use `bash scripts/setup.sh --cpu` then run `demo`.
The setup installs public dependencies; it never changes the driver or stops
another user's job. The model is pinned Qwen2.5-1.5B-Instruct, downloaded
separately with its license. Neither model weights nor vendor code are in Git.

| Included path | Purpose |
|---|---|
| `demo` | Offline tiny model + tiny Monte Carlo, CPU correctness only |
| `start` | Real GPU prefix-state and numerical-state experiments, sequentially, with a combined results page |
| `run` | Configurable Transformers KV recompute/restore comparison |
| `mc` | New synthetic GBM/Asian-option path-state recompute/restore/reprice comparison |
| `vllm-probe` | Optional HTTP generation exercise; **not** a KV filesystem comparison |

Read [workload definitions](docs/WORKLOADS.md),
[optional vLLM setup](docs/OPTIONAL-VLLM.md) and the
[operator runbook](docs/RUNBOOK.md) for customization. These are new reference
experiments, not the original concurrent FSI/RAG test. GPU speedups and optional
vLLM compatibility still require validation on the target GPU; CPU CI is not
evidence of either.

## The question

A model has already processed a document. Its intermediate key/value (KV) state
is no longer retained on the GPU. For the next request, is it quicker to process
that document again, or read the saved state and continue from it?

| Measured path | Work before the first newly generated token |
|---|---|
| Recompute | Transfer input token IDs, process the complete prefix on GPU, process the question suffix and choose the first output token |
| Filesystem restore | Read saved prefix KV, purge its file pages, check the file hash, deserialize, copy state through RAM to GPU, reconstruct the cache, process the same suffix and choose a new output token |

**A 3.5× improvement is a hypothesis, not an included result or acceptance rule.**
The report calculates `median(recompute time) / median(restore time)`. A result
below 1 means restoring was slower. We publish neither invented numbers nor
results from a different implementation. This initial revision has CPU
correctness and Linux-I/O validation; a measured trained-model GPU result must
be produced on the target machine before quoting a speedup.

Use the [operator runbook](docs/RUNBOOK.md) for detailed validation,
measurement, interpretation and troubleshooting steps. Adapter authors should
read the [package/interface walkthrough](docs/BACKENDS.md) and
[S3/native GPU implementation recipes](docs/ADAPTER-RECIPES.md).

## Install on Linux

Use Python 3.10+ and a CUDA-capable NVIDIA GPU/driver compatible with your PyTorch
installation. Consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for your host. No Docker, DDN image, Infinia client, NIXL or FSI application is
required. Provide enough host RAM and GPU memory for the selected model and one
request's KV state and working buffers.

```bash
git clone https://github.com/mkohari/linux-gpu-state-benchmark.git
cd linux-gpu-state-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[run]'
python -m unittest discover -s tests -v
```

The reference integration pins PyTorch 2.9.0, Transformers 4.57.1 and safetensors
0.6.2. It supports full-attention **Llama and Qwen2 model architectures** using
the public DynamicCache interface. It does not support every model or serving
stack. Sliding-window, hybrid, quantized and distributed caches need a separately
validated engine. Models require their own access and license terms. An ordinary
run does not load from the network or execute custom model-repository code.
Only the explicit `download-model` helper fetches the pinned public snapshot.

## Run a real GPU measurement

Place an authorized local safetensors model snapshot, including tokenizer files,
at your chosen path. The prefix plus question and output must fit its supported
context window. Choose a fresh output directory each time.

```bash
gpu-state-bench run \
  --model /models/YOUR-LOCAL-SNAPSHOT \
  --device cuda:0 \
  --state-root /mnt/YOUR-FILESYSTEM/kv-state \
  --output runs/linux-001 \
  --contexts 8 --prefix-tokens 4096 --repeats 3 --new-tokens 8
```

`--state-root` selects the storage actually being measured. Use your local ext4,
XFS or other target only after its purge/residency checks pass. The program records
the detected filesystem type where `findmnt` is available. It does not label all
Linux storage ext4, assume NVMe or claim a storage appliance's caches are cold.
Sufficient free capacity for the declared KV corpus is checked before population.

The default run records **24 recompute and 24 restore samples**, in a seeded
shuffled order. Every request generates eight greedy tokens, even if it reaches
an end-of-sequence token. Each prefix has exactly 4,096 tokens; the suffix length
depends on the actual tokenizer and is recorded separately. Synthetic text
fixtures contain unique reference IDs and repeated narrative. These are not news
articles or a retrieval/financial accuracy test.

Each model snapshot file is hashed once outside measurement. Initialization,
prefix-state population and write costs are recorded separately. GPU model
weights remain resident. Between requests, this process releases all prefix KV
references and unused GPU allocator blocks. **This is explicit state release,
not a natural LRU eviction test or proof that the corpus exceeds physical GPU
memory.** No unrelated process is stopped and no global file cache is flushed.

## Read the results

Open `runs/linux-001/report.html`, or inspect:

| Output | Contents |
|---|---|
| `summary.json` | p50/p95 and counts, measured speedup, hardware/software/model identities, byte totals and limits |
| `samples.jsonl` | Every timing, generated token IDs, correctness result, peak allocated GPU memory and storage receipt |
| `population.json` | Prefix generation, exact KV fingerprints and storage-write costs |
| `cases.json` | Exact synthetic input token IDs for every prefix and suffix |
| State directory `io.jsonl` | Read/write bytes, purge time and file-page residency before/during/after I/O |
| `failure.json` | Failure details when a run does not complete acceptance |

Before accepting a sample, the benchmark compares **every generated token ID**
and the **SHA-256 of every full prefix KV tensor** with its reference. Tensor
hashing copies state to CPU after the measured generation and reports its cost
separately. It never claims the whole application avoids RAM. The filesystem
restore headline includes its file hash verification and deserialization.

First-token time ends when the first newly selected token is available to the
local process. It is **not network streaming latency or complete-answer time**.
Completion time covers the fixed generated-token count. See
[the complete method](docs/METHOD.md) before presenting results.

## Offline correctness check

```bash
gpu-state-bench smoke --output runs/smoke-001 --state-root state
```

This builds a tiny random model in memory, saves/restores actual KV tensors and
verifies identical continuation tokens. No trained model or internet is needed.
It can run on CPU for development. Linux also exercises real page-cache purge;
other development hosts use an explicitly labeled buffered diagnostic. A smoke
report has `performance_run: false` and **no speedup claim**.

## Add your own backend separately

An external package implements three public methods: `describe`, `write` and
`read`. It receives tensor mappings rather than an imposed HTTP/byte transport,
so it can implement host staging or a separately qualified GPU-direct path.

```bash
pip install /path/to/YOUR-OWN-ADAPTER
gpu-state-bench run --model /models/YOUR-LOCAL-SNAPSHOT \
  --state-root /srv/adapter-work --output runs/external-001 \
  --backend your_adapter.module:YourBackend \
  --backend-options /private/backend-options.json
```

The module name is illustrative, not an included integration. The loader does
not discover cloud credentials, contact a vendor endpoint or fall back silently.
Examples such as DDN native storage, AWS S3, MinIO or NetApp belong in separately
distributed adapter packages with their own dependencies and terms. No client
implementation for any of them is included here. See
[the external-backend contract](docs/BACKENDS.md).

Before loading a model, an installed adapter can run `gpu-state-bench check-backend`
with synthetic tensors. It tests multiple keys, exact bytes/shape/dtype, metadata,
repeated reads and missing-object handling. Use CPU for a host-staged adapter or
CUDA for a native GPU-only adapter. A passing check is state-correctness evidence,
not a performance result or independent proof of the backend's network path.

## Distribution and provenance

This is newly written source using public APIs. It does not copy the earlier
mixed-workload benchmark or include its history, code, runtime images or results.
The numerical simulation is newly authored synthetic GBM, not the earlier Heston
kernel. News RAG, concurrent serving orchestration and proprietary model patches
remain excluded. Optional vLLM uses public upstream packages in a separate
environment. Read [PROVENANCE.md](PROVENANCE.md) and [LICENSE](LICENSE).

MIT covers this source, not licenses for dependencies, model weights, storage
services or separately installed adapters. This is a public open-source project;
anyone can clone and use its original source under MIT. Keep its copyright and
license notice when redistributing copies or substantial portions. No particular
hardware performance is guaranteed.

## Attribution and citation

Original project author, copyright holder and maintainer: **Moiz Kohari**.
Use [CITATION.cff](CITATION.cff) or GitHub's “Cite this repository” button when
citing the project. Identify the commit, configuration and measured results when
publishing a benchmark; a citation is not evidence of a particular speedup.

MIT preserves the copyright/license notice in redistributed source or substantial
portions; it does not require every use to include a paper citation, advertising
credit or a backlink. Citation is requested for scholarly and benchmark reporting.
Third-party packages and separately downloaded models retain their own authors,
copyrights and licenses. See [AUTHORS.md](AUTHORS.md) and [PROVENANCE.md](PROVENANCE.md).
