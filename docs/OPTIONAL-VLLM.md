# Optional public vLLM serving exercise

This is for learning how to run a serving engine and send streaming requests.
The language filesystem experiment uses Transformers because it explicitly
saves/restores KV tensors through the benchmark's backend interface. **The HTTP
probe below does not export/restore vLLM KV through that interface.** It provides
no filesystem-vs-prefill speedup, real-news RAG, retrieval or cold-cache proof.

## Install once, in a separate environment

Use Linux, Python 3.10–3.12, an available NVIDIA GPU and compatible driver. First
complete the reference setup and public model download in
[GETTING-STARTED.md](GETTING-STARTED.md).

```bash
bash scripts/setup-vllm.sh
```

This installs public `vllm==0.11.0` into `.venv-vllm`. It does not alter `.venv`.
The separate environment is intentional: vLLM 0.11.0's upstream CUDA dependency
set pins PyTorch 2.8.0, while the reference benchmark pins 2.9.0.
[Upstream requirements](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt).
The script runs the package dependency check. Downloaded wheels have their own
terms. There is no DDN image, server modification or private model patch.

This is a pinned optional recipe, not a claim that the current source has been
validated on every GPU. The client has local HTTP protocol tests. Actual
vLLM/GPU installation and performance require validation on your target host.

## Start the server in terminal A

Do this after `start` finishes; do not occupy its GPU concurrently.

```bash
bash scripts/start-vllm.sh
```

The launcher serves the already downloaded local model, listens only on
`127.0.0.1:8000`, names it `benchmark-public-model`, requests FP16, an 8,192-token
window, at most one active sequence, eager execution and 70% GPU memory use.
It explicitly disables automatic prefix caching and model-card generation
defaults. It runs in the foreground; wait for the server-ready message and
leave this terminal open. Ctrl-C stops this server. It never stops another job.
The memory setting does not guarantee model fit. If the port is occupied, it
fails; it does not kill the process using that port.

To choose GPU 1 and port 8001:

```bash
BENCH_GPU=1 BENCH_VLLM_PORT=8001 bash scripts/start-vllm.sh
```

## Send the probe from terminal B

```bash
cd linux-gpu-state-benchmark
source .venv/bin/activate
gpu-state-bench vllm-probe --output runs/vllm-001
```

For port 8001, append `--base-url http://127.0.0.1:8001/v1`.
The command uses the local model tokenizer to build 2 synthetic 1,024-token
prefixes plus their question suffix, and sends each request 3 times sequentially
to `/v1/completions`. Six measured requests each request exactly 16 greedy output
tokens with `ignore_eos=true`. No chat template is added because it sends the
exact token IDs to the completions API. This is a short fixed-budget test;
answers may not be complete sentences. See upstream's
[server API](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)
and [cache/engine arguments](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html).

`samples.jsonl` retains every generated text, prompt/output token counts and
timings. The client requires a completed stream, visible non-whitespace answer
text and matching token counts. `summary.json` gives p50/p95 and sample counts.
`cases.json` retains exact input IDs. These are serving samples, not matched
storage arms. The first request includes any server warmup left after startup.

**First visible text** starts just before the HTTP request and ends when the
client receives non-whitespace generated text. It includes HTTP overhead, server
queueing, prefill and generation until visible text. A single token can decode
to empty text/whitespace, so this differs from local first-token timing. Complete
request time ends after the stream's completion marker. The client cannot split
these times into retrieval/storage/queueing causes; there is no retrieval here.

The server must expose exactly one model, and the supplied `--model` tokenizer
must match it. The client hashes its local snapshot but **cannot attest server
weights or settings over HTTP**. Retain your server command, logs, model revision
and package versions if sharing results. An already running external server may
have different caching; the probe makes no cold/revisit assertion about it.

Optional authenticated endpoints can use your own `BENCH_VLLM_API_KEY` environment
variable. The key is used in the request header and is not written to reports.
Never put credentials in the base URL or commit environment files.

For actual vLLM storage-offload measurement, a separately implemented and
qualified public KV-connector integration would be required. Installing vLLM or
obtaining a fast repeated HTTP response does not establish storage-state reuse.
