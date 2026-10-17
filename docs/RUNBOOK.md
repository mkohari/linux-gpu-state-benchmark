# Operator runbook

Use this to install, validate and run the Linux reference, then compare a
separately supplied storage adapter. Only the Linux filesystem backend ships
here. Adapter authors should follow [BACKENDS.md](BACKENDS.md) and
[ADAPTER-RECIPES.md](ADAPTER-RECIPES.md). Those guides do not supply vendor clients.

## Completion criteria

| Stage | Successful output | What it establishes |
|---|---|---|
| Automated tests | Test log | Reference contracts and model/state correctness; Linux tests exercise actual file-page purge |
| `smoke` | `summary.json`, `performance_run: false` | Tiny random model saves/restores actual KV and produces matching tokens |
| `check-backend` | `backend-check.json` | Installed adapter preserves multiple keys, bytes/layout/dtype and metadata; missing key fails |
| Adapter-specific qualification | Your fault-test and independent transport evidence | Claimed service access, CPU/GPU transport and failure behavior |
| `run` | `summary.json`, `performance_run: true` | Matched trained-model timings and full KV/output correctness |

A contract pass is not storage-traffic proof. A CPU check is not a CUDA check.
External transport remains unqualified by this core, even after a model run.
No stage requires a 3.5x speedup. Restore can legitimately be slower.

## 1. Prepare the host and choose a model

You need Linux, Python 3.10+, a compatible CUDA GPU/driver, enough host RAM and
enough VRAM for the selected model plus one request's cache/working buffers.
Select a device you are authorized to use. The program does not stop other
applications or reserve a GPU. Record background load during each measurement.

Obtain a local, authorized safetensors model/tokenizer snapshot separately. The
engine supports full-attention Llama/Qwen2 in float16/bfloat16/float32. Arbitrary
quantized, hybrid, sliding-window or distributed models require engine work.
Run commands use local models and disable custom model-repository code. The
explicit `download-model` helper can obtain the pinned public Qwen snapshot.
New users should follow [GETTING-STARTED.md](GETTING-STARTED.md); separate
[workload definitions](WORKLOADS.md) cover the new Monte Carlo experiment and
[OPTIONAL-VLLM.md](OPTIONAL-VLLM.md) covers the isolated serving exercise. This
runbook's detailed measurement steps below describe the reference prefix-KV path.

The context window must fit **prefix + actual question suffix + output**. A
2,048-token model cannot use the default 4,096-token prefix. Select a suitable
model/window or disclose a consistently reduced workload for all backends.

```bash
git clone https://github.com/mkohari/linux-gpu-state-benchmark.git
cd linux-gpu-state-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[run]'
gpu-state-bench --help
nvidia-smi
python - <<'PY'
import torch, transformers, safetensors
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Transformers:', transformers.__version__, 'safetensors:', safetensors.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable in this Python environment'
print('GPU 0:', torch.cuda.get_device_name(0))
PY
```

The reference pins PyTorch 2.9.0, Transformers 4.57.1 and safetensors 0.6.2. Use
the [official PyTorch instructions](https://pytorch.org/get-started/locally/) to
select a compatible CUDA wheel/driver while preserving the reference version.
CPU-only PyTorch cannot measure a GPU run. No DDN image, NIXL, native client or
FSI deployment is needed for the Linux reference.

## 2. Select private run locations and capture the environment

Replace the first two values with real paths. The timestamp gives each execution
a new label; use fresh output paths for every run.

```bash
export BENCH_MODEL=/models/YOUR-LOCAL-SNAPSHOT
export BENCH_STATE=/mnt/YOUR-FILESYSTEM/kv-state
export BENCH_LABEL="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BENCH_STATE" "operator-evidence/$BENCH_LABEL"
findmnt --target "$BENCH_STATE"
df -h "$BENCH_STATE"
nvidia-smi > "operator-evidence/$BENCH_LABEL/gpu-before.txt"
uname -a > "operator-evidence/$BENCH_LABEL/kernel.txt"
git rev-parse HEAD > "operator-evidence/$BENCH_LABEL/source-commit.txt"
python -m pip freeze > "operator-evidence/$BENCH_LABEL/python-packages.txt"
```

Retain the filesystem/mount settings, driver, physical storage/topology, caches
and background load. The program records model hashes, software versions, GPU
identity/memory and filesystem type where available; it cannot infer every
controller, gateway or remote-server cache setting.

The program prints its new `run-<UUID>` state path at startup and completion.
Keep that path for audit/cleanup. Existing results/state are not overwritten.
The disk check estimates the chosen KV corpus; there is no fixed multi-terabyte
requirement. GPU OOM remains possible, so start with a small model qualification.

## 3. Check installation and the actual filesystem target

```bash
python -m unittest discover -s tests -v
gpu-state-bench smoke \
  --state-root "$BENCH_STATE" --output "runs/$BENCH_LABEL-smoke"
gpu-state-bench check-backend \
  --backend filesystem --device cpu \
  --state-root "$BENCH_STATE" --output "runs/$BENCH_LABEL-filesystem-contract"
```

On Linux these use real targeted purge/residency checks. Inspect the printed
state directory's `io.jsonl`: successful reads/writes have positive bytes and
zero `resident_pages_before` and `resident_pages_after`. Pages may populate
during buffered I/O. Purge time belongs to the measured file boundary.

Neither command establishes trained-model performance. The smoke uses a tiny
random model. The contract check uses synthetic tensors without any model.

```bash
python - <<'PY'
import json, os
p = 'runs/' + os.environ['BENCH_LABEL'] + '-filesystem-contract/backend-check.json'
r = json.load(open(p))
assert r['correctness_passed'] and not r['performance_run'] and not r['transport_qualified']
print(r['checks'])
PY
```

If purge fails, qualify or change the target mount. Do not disable the gate,
flush global caches or relabel a buffered diagnostic as the measured reference.
Local page purge does not flush a storage controller or server cache.

## 4. Run a small trained-model qualification

These examples use GPU device 0. If assigned another device, change `--device`
consistently for every test and retain the actual GPU UUID in the report.

```bash
gpu-state-bench run \
  --model "$BENCH_MODEL" --device cuda:0 \
  --state-root "$BENCH_STATE" --output "runs/$BENCH_LABEL-linux-small" \
  --contexts 2 --prefix-tokens 128 --chunk-tokens 128 \
  --repeats 3 --new-tokens 4 --dtype float16 --attention sdpa
```

Passing establishes that this model/runtime/path works and matches the reference.
Its short-prefix speed ratio is not the default long-prefix result. Snapshot
hashing and initialization can take time and occur outside measured requests.

## 5. Run the Linux reference

```bash
gpu-state-bench run \
  --model "$BENCH_MODEL" --device cuda:0 \
  --state-root "$BENCH_STATE" --output "runs/$BENCH_LABEL-linux" \
  --contexts 8 --prefix-tokens 4096 --chunk-tokens 2048 \
  --repeats 3 --new-tokens 8 --dtype float16 --attention sdpa --order-seed 1234
```

This yields 24 recompute and 24 restore requests in a seeded shuffled order, each
with eight greedy output tokens. Earlier prefix state is explicitly released
between requests. This is not natural LRU eviction or corpus-greater-than-GPU
proof. Model weights remain resident. All expected token IDs and full prefix-KV
hashes must match. [METHOD.md](METHOD.md) defines the timing boundaries.

Open `report.html` locally, or regenerate it without model/GPU work:

```bash
gpu-state-bench report "runs/$BENCH_LABEL-linux/summary.json"
python - <<'PY'
import json, os
r = json.load(open('runs/' + os.environ['BENCH_LABEL'] + '-linux/summary.json'))
assert r['correctness_passed'] and r['performance_run']
for mode in ('recompute', 'restore'):
    m = r['summary'][mode]['ttft_seconds']
    assert m['n'] == 24
    print(mode, 'samples:', m['n'], 'p50 ms:', m['p50'] * 1000, 'p95 ms:', m['p95'] * 1000)
print('Measured median recompute/restore ratio:', r['median_ttft_speedup'])
print('Storage bytes and purge cost:', r['storage_totals'])
PY
```

First-token time ends when the first new token is available to this local
process. It is not network streaming time or complete-answer latency. Restore
includes file read/purge/hash/deserialization/staging and suffix processing;
full KV hash diagnostics run afterward and copy tensors to CPU separately.

## 6. Install and qualify an external adapter

An author must first implement the external package in the adapter guide. No
vendor package ships here. Replace the paths and `module:Class` with real values.

```bash
python -m pip install -e /path/to/YOUR-ADAPTER-PACKAGE
chmod 600 /private/backend-options.json
```

Use its supported external credential mechanism and a dedicated benchmark
namespace. Record the complete adapter commit, SDK/dependency inventory and
non-secret transport/site configuration. The core does not deploy a MinIO
server, provision AWS resources or configure an Infinia cluster.

Start with the appropriate **alternative** below. A host-staged S3 adapter can
test CPU roundtrip. A GPU-only native adapter must test the intended CUDA device
and may correctly reject CPU inputs.

```bash
# Host-staged adapter:
gpu-state-bench check-backend \
  --backend your_state_adapter.adapter:YourBackend \
  --backend-options /private/backend-options.json \
  --state-root /srv/benchmark-control --device cpu \
  --output "runs/$BENCH_LABEL-external-contract-cpu"

# Native GPU-only adapter:
gpu-state-bench check-backend \
  --backend your_state_adapter.adapter:YourBackend \
  --backend-options /private/backend-options.json \
  --state-root /srv/benchmark-control --device cuda:0 \
  --output "runs/$BENCH_LABEL-external-contract-cuda"
```

The check writes three distinct states and performs six repeated/out-of-order
reads, checks full bytes/names/shapes/dtypes and metadata, then requires an explicit
missing-key failure. It does not inject authentication, corruption, timeout or
partial-transfer faults; run your adapter's own negative tests for those. Its
`transport_qualified: false` is intentional. Corroborate service requests and the
actual CPU/GPU path with independent transport evidence.

Run the small model qualification from step 4 with the external flags. Once it
passes, use the exact long-prefix settings of the Linux reference:

```bash
gpu-state-bench run \
  --model "$BENCH_MODEL" --device cuda:0 \
  --backend your_state_adapter.adapter:YourBackend \
  --backend-options /private/backend-options.json \
  --state-root /srv/benchmark-control --output "runs/$BENCH_LABEL-external" \
  --contexts 8 --prefix-tokens 4096 --chunk-tokens 2048 \
  --repeats 3 --new-tokens 8 --dtype float16 --attention sdpa --order-seed 1234
```

For a remote backend, `--state-root` is its local working/ownership-manifest
location. The adapter's options select the actual remote target. Changing this
local path does not configure a bucket or endpoint. Each model run still includes
recomputation as its matched control.

## 7. Check comparability before presenting a chart

```bash
python - <<'PY'
import json, os
from pathlib import Path
base, label = Path('runs'), os.environ['BENCH_LABEL']
a, b = base / (label + '-linux'), base / (label + '-external')
x, y = (json.loads((p / 'summary.json').read_text()) for p in (a, b))
assert all(r['correctness_passed'] and r['performance_run'] for r in (x, y))
assert json.loads((a / 'cases.json').read_text()) == json.loads((b / 'cases.json').read_text())
assert x['model']['sha256'] == y['model']['sha256']
for field in ('repeats', 'new_tokens', 'order_seed'):
    assert x[field] == y[field], field
for field in ('gpu', 'dtype', 'torch', 'transformers', 'cuda_runtime',
              'prefill_chunk_tokens', 'attention_implementation'):
    assert x['environment'][field] == y['environment'][field], field
print('Recorded inputs/runtime match. Review background load and transport proof separately.')
PY
```

The program hashes the class-defining adapter file, not the entire external
package or SDK. Retain that complete provenance separately. Repeat executions
to understand variation and keep slower/unfavorable results too. A ratio within
one run does not establish a production SLA, universal speedup or maximum throughput.

## 8. Retain evidence and clean up only owned data

Retain each run's summary, samples, input cases, population records, HTML report,
I/O audit and operator evidence. Keep adapter contract/fault-test/transport
receipts as separate evidence. Keep credentials and private options outside Git.
Review raw SDK logs before sharing; arbitrary third-party exceptions can contain
private endpoints, credentials or signed URLs.

State is deliberately left for inspection. A remote adapter should save an
owned-object manifest in its printed working directory. Before removal, verify
the exact local/remote namespace belongs to this run and is no longer needed.
Use the authorized provider procedure scoped to that namespace. The core supplies
no bucket/dataset-wide deletion command and never deletes unrelated objects.

An interrupted run is not resumed as accepted measurement. Preserve partial
evidence and `failure.json`, resolve owned-state cleanup, then start a fresh
output/namespace. Never merge partial rows into a later run.

## Troubleshooting

| Symptom | Action |
|---|---|
| CUDA unavailable | Check active Python, PyTorch build, compatible driver and assigned device; CPU smoke is not a replacement measurement |
| Unsupported cache/model/dtype | Use the declared architectures and precision; another layout requires engine work |
| Context exceeds model limit | Reduce prefix/output or select a larger window; apply the same change to all compared paths |
| GPU OOM | Select a smaller supported model or shorter prefix/chunks, then rerun matched settings |
| Slow snapshot hashing | It reads real local weights once outside measurement; check file access and size |
| Output/state already exists | Choose a fresh path/namespace; preserve existing evidence |
| Purge leaves resident pages | Qualify the mount and filesystem behavior; do not silently disable the gate |
| Adapter cannot import | Install it into this same environment and check its exact `module:Class` |
| Missing adapter options | Use its actual documented schema; there are no automatic provider defaults |
| Authentication/TLS failure | Correct credentials, scoped permissions, endpoint and CA trust; no fallback or silently disabled TLS |
| Missing key returns old state | Fix the adapter before loading a model |
| Hash/shape/dtype/token mismatch | Treat as failed correctness; inspect identity, packing, serialization, cache positions and completed transfers |
| Native GPU capability unavailable | Qualify the external native stack; a CPU/S3 path does not satisfy native-only mode |
| Contract passes but no service reads appear | Investigate hidden payload caching or inadequate receipts; transport remains unqualified |
| Restore slower than recompute | A valid outcome; inspect I/O/staging/hash costs and workload suitability, and retain the result |

## Evidence needed for a presentation

Identify model, hardware/runtime, exact input/output workload, p50/p95 and counts,
actual ratio, population and verification boundaries, real bytes/cache behavior,
passed correctness and independently proven external transport where applicable.
State that this is a sequential single-request state-release microbenchmark.
Do not carry a number from an older mixed Monte Carlo/RAG test into this result.
