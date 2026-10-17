# Storage adapter implementation recipes

These are engineering workflows for independently supplied adapters, not shipped
vendor implementations. Use [BACKENDS.md](BACKENDS.md) for the package and Python
contract, and [RUNBOOK.md](RUNBOOK.md) for execution. No SDK, binary, proprietary
source, customer configuration or credential is included here.

## Choose the actual access path first

| Target | Candidate adapter path | Host payload staging | What must be supplied externally |
|---|---|---|---|
| AWS S3 | S3 object client | Normally yes for a conventional SDK workflow | Authorized account/bucket, credential mechanism, compatible SDK and network |
| MinIO or another S3-compatible service | S3 object client with explicit endpoint | Normally yes for a conventional SDK workflow | Service endpoint/bucket, supported authentication, TLS trust and compatible client |
| DDN Infinia native | Supported native GPU transport | No, if independently proven for that deployment | Authorized native software/plugin, fabric, cluster/site inputs and credentials |
| Any vendor exposing an S3 endpoint | S3 workflow against that endpoint | Depends on the implemented client/path | Explicit service/API compatibility and qualification |
| Mounted file storage | Supplied `filesystem` backend | Yes | Mount that passes the Linux purge/byte checks |

The brand does not identify the protocol. A native Infinia measurement and an
S3 request against a vendor endpoint are distinct implementations. NIXL itself
also does not imply GPU-direct transport: its object backend distinguishes
CPU-memory engines from accelerated GPU-capable paths with extra prerequisites.
Pin and verify the actual deployed version rather than assuming current upstream
documentation describes an older installed plugin.

## Recipe A: conventional S3-compatible object adapter

### Inputs your adapter should define

Choose and document your own options schema. The benchmark passes it through;
it does not interpret bucket, region or endpoint fields. Typical non-secret
inputs include bucket/container, benchmark-only namespace prefix, region, an
optional custom endpoint, addressing mode, TLS CA bundle, request timeout,
bounded retry policy and allowed concurrency. Keep credentials outside that
JSON when your SDK supports roles, profiles or environment-based authentication.

An **illustrative adapter-defined** JSON file might contain:

```json
{
  "bucket": "operator-owned-benchmark-bucket",
  "namespace_prefix": "gpu-state-tests",
  "endpoint_url": "https://objects.example.invalid",
  "region": "operator-selected-region",
  "request_timeout_seconds": 60,
  "max_attempts": 3
}
```

This is not a runnable core configuration. Its names must match your adapter.
For AWS, the SDK may derive its service endpoint from the region instead of
using a custom URL. For MinIO or another private service, validate the exact
endpoint, signing region, addressing style and trusted certificate chain.
Do not make disabling TLS verification a benchmark default.

### `write` workflow

1. Confirm that the caller's key belongs to your fresh run namespace.
2. Copy each tensor to temporary CPU staging, preserving dtype and values, and
   make a contiguous tensor archive using an appropriate public serialization
   library. Record logical tensor bytes, archive length and archive SHA-256.
3. Upload the complete archive under that namespace/key. Await full upload
   completion; finish multipart upload before considering the object committed.
4. Upload a manifest with the archive length/hash and original benchmark metadata
   unchanged. Publish the manifest only after payload success. Respect your
   service's consistency contract and fail if committed data cannot be retrieved.
5. Return actual payload-byte/copy/serialization/upload/retry receipts. Release
   staging contents; do not retain a payload cache for the later `read`.

### `read` workflow

1. Look up committed manifest information for this exact key. Metadata caching
   is allowed when immutable and declared. A missing/incomplete state is an error.
2. Issue a real object fetch for the saved payload on **every** external read.
   Consume the entire response and close its stream. Check the expected length.
3. Verify an explicit archive SHA-256. Do not assume an S3 ETag is a full-file
   SHA-256 or even an MD5 for every upload/encryption method.
4. Deserialize without pickle, rebuild the exact tensor mapping, copy it to the
   requested device and synchronize the device copy before returning `Restored`.
5. Return the original metadata and actual read/copy/hash/deserialization costs.
   Release temporary CPU payloads after the call. Reuse of allocation capacity is
   fine; satisfying the next read from old payload contents is not.

Do not serialize tensors by converting their numeric dtype, turn a KV lookup
into a cached final answer, or suppress read failures by recomputing the prefix.
File-page purging is relevant only if your client writes local staging files.
For an in-memory SDK stream, prove repeated service reads and absence of a
reusable client payload cache instead. Server caching is a separate declared
property, not something `POSIX_FADV_DONTNEED` can control.

### Service-specific qualification

For AWS, record region, compute/storage placement, bucket type/storage class,
encryption, SDK version and credential mode. S3 Standard and S3 Express One Zone
are different configurations; do not combine their measurements under one label.
Do not use an archive class requiring a separate restore workflow for a direct
online-read test without explicitly adding and timing that workflow.

For MinIO/other compatible services, pin server/client versions and test the
specific operations your adapter uses. Record placement, TLS, object layout,
concurrency and any gateway cache. A single-node development deployment does not
stand in for a production cluster. Do not assume every S3-compatible service
supports every AWS option, checksum feature or authentication mechanism.

These object operations can incur service/request/storage charges and leave data
behind. Use a dedicated namespace and an operator-controlled retention/cleanup
plan. Never issue a broad bucket deletion from the benchmark.

## Recipe B: native Infinia or another qualified GPU-direct adapter

### Deployment prerequisites

Obtain the authorized native client/plugin and its supported installation
instructions outside this repo. Confirm the exact runtime/plugin/API versions,
GPU and NIC/fabric support, cluster reachability, authentication, tenant and an
owned benchmark namespace/dataset. Record actual library identities in your
adapter receipts. The core does not log in, provision a native dataset, install
a plugin or choose an endpoint for you.

Native-only adapters may reject `--device cpu`. Use the generic checker with
`--device cuda:0`, then separately qualify the real transport before a model run.

### `write` workflow

1. Validate the requested GPU-buffer capability before starting work. Native-only
   mode must fail if that capability is unavailable.
2. Preserve the input tensor mapping. If the native API needs contiguous/aligned
   buffers, pack tensors **on GPU** and record layout, dtype, offsets and any
   padding as CPU control metadata. Handle non-contiguous inputs correctly.
3. Register the actual CUDA payload buffers and create owned remote object/state
   descriptors through your supported native API. Retain registrations and
   buffers until completion. CPU metadata is permitted.
4. Submit writes, poll/await their actual completion with a bounded deadline and
   check error states. Commit the manifest after the data is complete.
5. Return transfer/registration/completion receipts and byte counts. Never switch
   to a host/S3/filesystem payload path to make a native-only test pass.

### `read` workflow

1. Validate key/manifest identity and allocate or reuse suitable GPU scratch
   buffers. Previous buffer contents must not satisfy the read.
2. Register/describe those CUDA buffers, issue real native reads and wait for
   completion. Account for all objects/chunks and retries.
3. Reconstruct the tensor mapping on GPU using the recorded layout. Return
   ready tensors and unchanged metadata, with all work completed before return.
4. Retain descriptor/library/traffic proof that identifies the actual backend,
   CUDA payload memory and completion status. Correlate it with the requested
   keys and actual bytes.

Calls that copy tensor payloads to CPU for transport invalidate a GPU-only payload
claim. GPU-to-GPU packing is different from host staging. The benchmark's full
KV SHA-256 diagnostic is a **separate CPU copy after measured generation**, so
the application as a whole still uses CPU RAM even with native transport.

Run deliberate wrong-credential, missing-object, unsupported-capability,
corruption, timeout and partial-transfer tests using your authorized sandbox.
The generic checker cannot inject those vendor-specific faults. Passing its
hash checks alone is not proof that a native NIC/storage path was exercised.

## Acceptance before describing a provider as supported

- Python contract checks pass on the intended CPU/CUDA device.
- Adapter-specific negative tests fail correctly, with no fallback.
- Independent receipts prove the claimed service reads and actual memory path.
- A matched trained-model run passes all KV and output-token checks.
- Source/runtime/configuration identities and actual bytes are retained.
- An owner can identify and remove only that run's objects afterward.

Until these hold, say “adapter interface available” or “adapter under test,” not
“provider supported” or “GPU-direct verified.” No provider performance has been
measured by adding these instructions.

## Primary references

- [AWS S3 concepts and storage classes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
- [AWS credential environment configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html)
- [S3 object checksum/ETag semantics](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html)
- [MinIO Python client documentation](https://github.com/minio/minio-py)
- [Public NIXL object-backend architecture and memory capabilities](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/obj/README.md)

Use your licensed Infinia/native-client documentation for its deployed version's
installation and site-specific setup. These references do not redistribute those
components or replace version-specific qualification.
