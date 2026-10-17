# External backend contract

Only `FilesystemBackend` ships in this repo. Storage implementations for DDN,
AWS, MinIO, NetApp or another provider belong in separately installed packages.
The names here are examples, not claims of implemented compatibility.

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
