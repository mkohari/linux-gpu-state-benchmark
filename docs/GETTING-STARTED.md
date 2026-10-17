# Start here: from an empty Linux machine to a results page

Owner and author: Moiz Kohari. This guide belongs to the new independent Linux
reference benchmark. It does not install or require the FSI pipeline.

## What you will get

You will run two small experiments, one after the other:

1. **Language model:** process a synthetic document again, or load the model's
   saved working state and answer the same question. An answer is generated anew
   in both cases; a saved answer is never substituted.
2. **Monte Carlo:** generate possible price paths for a fictional asset again,
   or load saved paths and calculate the same two option prices anew. This part
   produces numbers, not language-model tokens.

Both ask whether reading intermediate work is faster than repeating it. A
speedup depends on your GPU, model, storage and chosen workload. There is no
guaranteed 3.5× result. The tests use explicit state release, not natural GPU
eviction, and do not recreate the original concurrent FSI/RAG experiment.

An optional, separate vLLM serving exercise is described at the end.

## Before you start

- Use Linux, preferably Ubuntu 24.04 with Python 3.12. Python 3.10–3.13 is
  supported by the reference setup; the optional vLLM recipe needs 3.10–3.12.
- Have Git and Python's `venv` module available. If Python reports that `venv`
  is unavailable, ask your administrator to install the matching Python venv
  package. The scripts never use administrator privileges.
- For actual GPU measurements, use an NVIDIA CUDA-capable GPU and a compatible
  installed driver. A 16 GiB GPU is a reasonable starting choice for the small
  public model, **not a tested minimum or a promise of fit**. Do not use a GPU
  already occupied by another important job.
- Allow roughly 30 GiB free disk for the reference environment, model, package
  downloads and first results; larger corpora need more. Optional vLLM has a
  separate environment and needs additional space. Host RAM must accommodate
  the model load, temporary serialization and restored state.
- Internet access is needed to clone and install packages. The trained model is
  a separate explicit download, about 3 GiB. After setup and download the
  reference experiments use local files and need no internet.
- This repository is public and MIT licensed. Cloning needs no GitHub account
  or project access grant. A source ZIP works too. Model/service dependencies
  remain subject to their own terms.

## Step 1 — Get the source

Open a terminal and enter:

```bash
git clone https://github.com/mkohari/linux-gpu-state-benchmark.git
cd linux-gpu-state-benchmark
```

Public HTTPS cloning requires no GitHub authentication. For a ZIP, extract it,
open a terminal in the extracted project folder, and continue below.

## Step 2 — Install the reference environment

For a Linux GPU machine:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The script creates a project-local Python environment, installs pinned public
PyTorch 2.9.0 CUDA 12.8, Transformers 4.57.1 and safetensors 0.6.2, installs the
benchmark, and checks dependency compatibility. Package downloads can be large.
It neither changes the NVIDIA driver nor installs a system CUDA toolkit. A
compatible driver is still required. See the
[official PyTorch release installation choices](https://pytorch.org/get-started/previous-versions/#v290)
if CUDA 12.8 wheels are unsuitable for your hardware. Older GPUs are not
necessarily supported by these wheels.

For a Linux machine with **no GPU**, use this instead:

```bash
bash scripts/setup.sh --cpu
source .venv/bin/activate
```

CPU mode teaches the workflow and checks correctness. It cannot produce a GPU
speedup claim. To switch that environment to a GPU build later, rerun setup
without `--cpu`. The setup script replaces PyTorch only within its project
environment. Mac development users can also use `--cpu`; macOS uses the public
PyPI wheel rather than the Linux CPU wheel index. Actual measurements require Linux.

Each new terminal needs `cd` into this folder and `source .venv/bin/activate`.
If you see “command not found”, the environment usually has not been activated.

## Step 3 — Try the offline demonstration first

```bash
gpu-state-bench demo
```

This checks dependencies and storage, constructs a tiny random model in memory,
saves and restores its real KV tensors, and runs a small synthetic Monte Carlo
case. **It downloads no model.** It prints the full path to `index.html` inside a
new uniquely named `runs/` folder. Open that file in a browser.

Success means both correctness checks passed. The random model's token output
is not meaningful prose. CPU times are deliberately not reported as GPU speedup.
On Linux this also tests real file-page purge. A Mac demo is labeled a buffered
development diagnostic because macOS is not the measured filesystem path.

Result folders are never overwritten. The demo leaves its own saved state files
in a uniquely named folder under `state/`, so you can inspect the I/O audit.

## Step 4 — Check the actual storage and GPU

Choose the filesystem you want to measure. If the default `state/` directory is
on the disk you want, simply run:

```bash
gpu-state-bench doctor
```

For another mounted filesystem:

```bash
gpu-state-bench doctor --state-root /mnt/my-disk/benchmark-state
```

`PASS` means that particular readiness check worked. `FIX` explains what needs
attention. The storage check creates and removes only a small owned probe file.
On Linux it verifies exact readback and that the file's reusable pages are
purged after writing and reading. It never flushes the machine's global cache.
Readiness does not certify speed, free memory under other jobs, or cold disk/
appliance caches. The full experiment still has to pass its correctness gates.

To choose GPU 1 rather than GPU 0, set this **before** starting:

```bash
export CUDA_VISIBLE_DEVICES=1
```

That selected GPU becomes `cuda:0` within this process. We do not stop or pause
other applications. The standard start command runs sequentially on that GPU.

## Step 5 — Download the public model

```bash
gpu-state-bench download-model
```

The helper downloads **Qwen/Qwen2.5-1.5B-Instruct** from Hugging Face at fixed
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, into
`models/qwen2.5-1.5b-instruct`. It keeps the model's Apache-2.0 license and model
card, and downloads no Python model-repository code. Review those upstream
terms. [Model card and files](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).

The model is **not packaged in Git or the source ZIP**. It is not the original
49B model or its private runtime. An interrupted download can be resumed by
repeating this command. The helper refuses a directory containing someone
else's files. It does not ask for or install a vendor license key.

## Step 6 — Run both GPU experiments

```bash
gpu-state-bench start
```

For a particular filesystem and readable result name:

```bash
gpu-state-bench start \
  --state-root /mnt/my-disk/benchmark-state \
  --output runs/my-first-gpu-test
```

The optional `--model /path/to/snapshot` selects a different authorized local
full-attention Llama/Qwen2 snapshot. Begin with the pinned public default before
changing models. The start helper checks the machine, runs language and numerical
experiments in separate processes, checks their reports, and prints one results
page. It does not start vLLM. Model loading, file hashing, population and verification
can make total runtime longer than the per-request numbers shown in the report.
There is no fixed finish-time promise.

Default real language workload: 8 synthetic 4,096-token prefixes, 3 repetitions,
8 generated tokens per request. That is **24 requests per mode, 48 measured
requests total**, plus untimed warmup and separately recorded population.

Default numerical workload in `start`: 2 independently seeded scenarios, each
with 65,536 paths × 64 monitoring dates × 8 FP64 bytes = **32 MiB** of saved
state. Both scenarios use the same fictional asset parameters. Three repeats
produce **6 requests per mode, 12 measured requests total**. Each request prices
both strikes and generates zero text tokens. The separate `mc` command defaults
to 252 dates (126 MiB per scenario); explicit options are recorded in the report.

These are approachable starting sizes, not a reproduction of the old 512-context/
110 GiB numerical corpus. Scale gradually using the individual `run` and `mc`
commands after reading [the workloads](WORKLOADS.md) and [operator runbook](RUNBOOK.md).

## Step 7 — Understand the page

| Item | Plain-language meaning |
|---|---|
| Recompute | Do the expensive intermediate work again before answering |
| Restore | Read saved intermediate work, then calculate a new answer |
| Language first token | Time until this local program selects the first new output token; not a whole sentence or network response |
| Numerical query | Obtain the price paths and calculate both option prices; not a language-model question |
| p50 | The middle measured time: half the requests were faster and half slower |
| p95 | An interpolated slow-tail percentile; with few samples it is descriptive, not a production latency guarantee |
| Speedup | Recompute median divided by restore median; 2 means restore used about half the time; below 1 means restore was slower |

Look for `correctness_passed: true` and, for real GPU tests, `performance_run:
true` in each `summary.json`. CPU demo reports always have `performance_run:
false` and a null speedup. The suite's `suite.json` says `completed: true` only
after both child reports pass. Any failure stops the suite and retains its
evidence; do not present partial results as a completed comparison.

Keep the whole results folder, printed state-directory paths and the source
commit (`git rev-parse HEAD`). Individual reports retain configuration, software,
source/model fingerprints, bytes, timing distributions and limitations. Models,
saved state and results are intentionally excluded from Git.

## Optional: learn vLLM serving

Follow [the separate vLLM guide](OPTIONAL-VLLM.md). It installs a separate public
serving environment and runs an HTTP streaming probe. This does **not** export/
restore vLLM's KV cache through our storage backend and is not the filesystem
speedup experiment. Do not run it at the same time on the same GPU as `start`.

## Common first-run problems

| Problem | Next step |
|---|---|
| GitHub clone fails | Check the public URL and network/proxy access, or use GitHub's source ZIP download |
| Python/venv not found | Install Python 3.12 and its venv package through your normal administrator process |
| CUDA unavailable | Check the NVIDIA driver; ensure setup was run without `--cpu`; rerun `doctor` |
| Download stopped | Repeat `download-model`; do not delete an unrelated model folder |
| GPU out of memory | Stop this benchmark, choose a free GPU or reduce the individual `run` prefix/chunk size; never kill another user's process |
| Purge check fails | Use a suitable Linux filesystem and consult the full runbook; there is no silent buffered fallback for GPU measurements |
| Result folder exists | Choose a new `--output` name or omit it to generate a unique one |
| Restore slower than recompute | This is a valid outcome, especially for small models/state and slow storage; inspect bytes and cost components |

For repeatable comparisons, retain the same model revision, dtype, token budget,
simulation parameters and request counts across storage targets. Do not compare
the old mixed-workload result with this new implementation as if only storage changed.
