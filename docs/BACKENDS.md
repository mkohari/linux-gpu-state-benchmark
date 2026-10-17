# Build an external storage adapter

Only `FilesystemBackend` ships in this repo. Storage implementations for DDN,
AWS, MinIO, NetApp or another provider belong in separately installed packages.
The names here are examples, not claims of implemented compatibility.

**For an operator:** follow [RUNBOOK.md](RUNBOOK.md) to install, validate and run
the Linux reference or a separately supplied adapter. **For an adapter author:**
use this guide for the Python package/interface, then choose a storage workflow
from [ADAPTER-RECIPES.md](ADAPTER-RECIPES.md). No vendor client is bundled by either
guide. A developer still has to implement and qualify the actual storage calls.

## Interface

Subclass `gpu_state_bench.backend.Backend`. Its constructor receives a new owned
working-directory path and a private options dictionary. Implement:

```python
from gpu_state_bench.backend import Backend, Restored

class YourBackend(Backend):
    def describe(self):
        return {
            "name": "your-implementation",
            "payload_transport": "describe actual memory and network path",
            "reusable_host_payload_cache": False,
        }

    def write(self, key, tensors, metadata):
        # Persist every tensor plus metadata using your separately supplied code.
        # Finish the write and synchronization before returning its receipt.
        raise NotImplementedError("Implement storage write outside this repository")

    def read(self, key, device):
        # Actually fetch the original payload, create exact tensors on `device`,
        # synchronize completion, and return Restored(tensors, metadata, receipt).
        raise NotImplementedError("Implement storage read outside this repository")
```

This skeleton intentionally contains no vendor API, client, endpoint, license or
credential. Merely copying it does not produce a working backend. An import or
transport failure stops the run, without falling back to filesystem.

## Tensor and identity requirements

- Input is a mapping of `layer.key` and `layer.value` names to CUDA tensors.
  The supplied full-attention engine uses `[1, kv_heads, prefix_tokens, head_dim]`.
- Preserve tensor names, shape, dtype and **all bytes**. Do not quantize, truncate
  or reconstruct a different state while labeling it a restore.
- Persist and return the metadata unchanged, including model snapshot, input
  token identity, schema, tensor hashes and exact payload tensor bytes.
- Keys are SHA-256 identities under a fresh run namespace. Do not read another
  run's data or delete unrelated objects. Do not include credentials in metadata,
  receipts, `describe()` or exception messages.
- The returned tensors must be ready on the requested device. Await async I/O
  and GPU copies. Wall-clock timing must cover actual completion.
- Do not keep reusable host/GPU payloads between requests. Metadata/presence
  caches are permitted if identified. Temporary staging buffers are permitted
  when their existence and size are declared.

The interface accepts tensors so a user can implement either CPU staging or a
direct GPU transport. A byte-only wire format is not forced on native adapters.
No third-party plugin is discovered or activated by brand name.

## Receipts and qualification

Return JSON-compatible receipts containing actual `bytes`, operation, I/O time,
serialization/deserialization, copy and synchronization costs, retry counts and
other backend-specific observations. State whether byte counts describe the
application payload, physical wire traffic, or both. Protocol retries and
encoding overhead can make those different quantities. Describe disk, CPU, GPU,
client and server caches accurately. A client-side file purge does not flush an
object server's cache.

Full KV hashes and continuation token equality test state correctness. They do
not prove requests went to the claimed storage service, crossed a NIC or avoided
CPU staging. External implementations require their own receipts, negative tests
and real hardware qualification. Generic reports retain the explicit limitation
that an external backend's transport has not been certified by this core repo.
Installing a library named NIXL, S3 or anything else is not transport proof.

For a direct GPU claim, establish actual supported GPU-buffer registration,
transfer descriptors, selected backend/library identity and lack of hidden host
payload bounce. CPU control work and the benchmark's separate hash diagnostics
still remain. Do not silently use another path when a requested capability fails.

## Integration sequence

1. Implement and unit-test the adapter in its own repository, with its own
   distribution license and separately obtained dependencies.
2. Test exact state roundtrip, missing objects, wrong credentials, short/corrupt
   payloads, timeouts, async completion and cleanup ownership.
3. Install the package into the same Python environment. Pass its explicit
   `module:Class` and a private JSON options file through the CLI.
4. Run a small correctness test, then a Linux/CUDA benchmark with independently
   verified transport receipts and matched model/input settings.
5. Keep the new source and backend identities with the results. Do not replace
   existing Linux results or reuse a previous implementation's headline.

Use the same prefix cases, dtype, model, prefill chunks, output count, repetition
count and hardware when comparing backends. Deliberately state any difference in
client caches, compression, network, concurrency or transport. This repository
does not provide an S3 credentials format or guess native storage configuration.

## 1. Create your own package outside this repository

Use a separate directory/repository. Install this benchmark first in your active
virtual environment. It is not currently published on PyPI.

```text
your-state-adapter/
  pyproject.toml
  src/
    your_state_adapter/
      __init__.py
      adapter.py
  tests/
```

Minimal `pyproject.toml` for that separate project:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "your-state-adapter"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["linux-gpu-state-benchmark==0.2.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

Declare your adapter's own license and add its own required SDK dependencies in
that project. Do not add them to the benchmark distribution. This benchmark's
public source can be cloned without an access grant; install it locally first
because it is not published on PyPI.

In `adapter.py`, implement the class shown above. In its constructor call
`super().__init__(root, options)` and create `self.root` with `mkdir(mode=0o700)`.
The loader supplies a **fresh nonexistent path**. For remote objects, build a
fresh namespace from your configured benchmark prefix and `root.name` (a new
`run-...` UUID), then save its public ownership manifest under `self.root`.
Two invocations must never reuse remote keys or a previous dataset accidentally.

Treat `options` as private input. Validate required fields and reject unknown
ones. Read credentials through your supported external credential mechanism.
Do not copy options wholesale into `describe()`, receipts or logs. `describe()`
should identify your provider/backend, protocol, actual memory path, client-cache
behavior and non-secret software identities. Give it a distinct name, not
`filesystem`, unless it is the supplied filesystem implementation.

```bash
# Run with the benchmark virtual environment active.
python -m pip install -e /path/to/your-state-adapter
python -c "from your_state_adapter.adapter import YourBackend; print(YourBackend.__name__)"
```

The module path to pass to the benchmark is
`your_state_adapter.adapter:YourBackend`. The earlier skeleton raises
`NotImplementedError` until you supply its storage implementation. Installing
the skeleton alone does not create an Infinia, AWS or MinIO adapter.

## 2. Define a persistent state envelope

For a host-staged object adapter, a practical layout under the owned namespace is:

```text
<owned-namespace>/<key>/payload.safetensors
<owned-namespace>/<key>/manifest.json
```

The manifest contains an envelope version, payload byte length and SHA-256,
the original benchmark `metadata` unchanged, and any layout information needed
by the adapter. Write the payload fully, finish any multipart upload, and write
the manifest last as the committed-state indicator. A read without a complete
manifest/payload pair is an error, not a cache hit or a reason to recompute.

A native GPU adapter can instead pack tensors on GPU into one or more objects.
Its manifest must preserve tensor names, shapes, dtypes, offsets and logical
lengths. Report alignment padding separately. Never change the original KV bytes.

`metadata` is control data. It may travel through CPU memory. Tensor payloads
may also use temporary CPU staging if declared; native GPU-only payload claims
require the distinct workflow in the recipes guide. Small metadata/presence caches
and reusable scratch allocations are permitted. Every external read must still
fetch the saved payload; old scratch contents must never become a hidden cache.

## 3. Return usable operation receipts

At minimum, each successful operation must return a JSON dictionary with:

| Field | Meaning |
|---|---|
| `operation` | Exactly `write` or `read` |
| `bytes` | Positive integer: actual successfully transferred encoded payload bytes; include archive headers/padding if transferred |
| `bytes_scope` | Your precise definition, e.g. `encoded application payload; excludes control objects and protocol overhead` |
| `logical_tensor_bytes` | Sum of original tensor lengths before encoding/padding/compression |
| `io_seconds` | Completed payload I/O time, not queue submission time |
| `retries` | Actual retry count; use 0 only when none occurred |
| `host_payload_staging` | Boolean describing your real transport, not the benchmark's separate hash diagnostic |

Add serialization, CPU/GPU copy, completion-wait and metadata costs where they
occur. State whether stage durations overlap; do not add overlapping times and
call the sum end-to-end latency. Report physical `wire_bytes` only when a reliable
counter measures it; otherwise omit it or use null. A tensor's nominal size alone
does not prove a transfer. Server-side caches can remain warm if documented.

The supplied filesystem backend uses its existing receipts (including file bytes,
purge costs and residency fields). The new checker enforces the mandatory
`operation`/positive `bytes` fields and JSON validity; the richer external fields
above require adapter-specific review. It cannot certify a third-party SDK's
network path merely by accepting a receipt dictionary.

## 4. Run the backend contract check before loading a model

Create an options JSON file in a private directory outside Git, using the schema
implemented by your own adapter. Mode 0600 is recommended. Then:

```bash
gpu-state-bench check-backend \
  --backend your_state_adapter.adapter:YourBackend \
  --backend-options /private/backend-options.json \
  --state-root /srv/benchmark-control \
  --device cpu \
  --output runs/adapter-contract-cpu-001
```

For a GPU-only native adapter, select `--device cuda:0` instead. A CPU-only test
does not establish CUDA support, and a native adapter may correctly reject CPU
inputs. The command writes three distinct synthetic tensor states, then reads
them six times in a different/repeated order. It checks float32, float16 and
bfloat16, non-contiguous input, exact tensor bytes/names/shapes/dtypes, unchanged
metadata, ready tensors on the requested device and an intentionally missing key.
Translate missing objects to `FileNotFoundError` or `KeyError`; do not translate
authentication or general transport failures into a successful empty read.

Successful output is `backend-check.json` with `correctness_passed: true`,
`performance_run: false` and `transport_qualified: false`. Retain its receipts.
It is **not** a full model test, performance result or proof of storage traffic.
All created data remains in this invocation's fresh namespace for inspection.

## 5. Add adapter-specific negative tests

| Injected condition | Required outcome |
|---|---|
| Missing key | Explicit missing-object exception, no stale data or recompute |
| Wrong key/model metadata | Identity check fails |
| Truncated or altered payload | Length or full hash check fails |
| Wrong/expired credentials | Explicit authentication failure, no anonymous/fallback success |
| Delayed async transfer | Method waits for completion or times out; never returns partial state |
| Retried operation | Actual retries/bytes recorded, bounded deadline honored |
| Another run's namespace | Ownership prevents reading/deleting unrelated state |
| GPU-direct requested but unsupported | Initialization/transfer fails, no hidden CPU path |

The generic checker covers the missing-key and successful roundtrip cases. Inject
the other faults in your adapter test suite or a controlled storage sandbox.
For real traffic proof, corroborate client receipts with provider request logs,
backend counters or transport instrumentation. Preserve versions and timestamps.

## 6. Run the model benchmark and retain the implementation

Follow the external-backend section in [RUNBOOK.md](RUNBOOK.md). Match the Linux
baseline settings exactly. Retain your adapter's complete Git commit, dependency
lock/inventory, public configuration, transport evidence and owned-object manifest.
The benchmark hashes the class-defining source file, **not the complete external
package or SDK**, so that extra provenance must be recorded by the adapter author.
