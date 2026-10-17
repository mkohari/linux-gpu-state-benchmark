# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
"""Exercise an installed backend without model weights or vendor-specific code."""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
from .backend import Restored
from .engine import tensor_hashes
from .runner import write_json


def checked_receipt(value, operation):
    if not isinstance(value, dict):
        raise ValueError("A storage receipt must be a dictionary")
    json.dumps(value, allow_nan=False)
    if value.get("operation") != operation:
        raise ValueError("Receipt operation does not match the completed operation")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Receipt requires a positive actual transferred-byte count")
    return json.loads(json.dumps(value, allow_nan=False))


def specs(tensors):
    hashes = tensor_hashes(tensors)
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashes[name]}
            for name, value in tensors.items()}


def check(backend, device, output, backend_identity=None):
    """Test keys, exact tensor bytes/layout/dtype, metadata and repeated reads.

    This is deliberately not proof of network traffic, GPU-direct transport or
    performance. A caching implementation can pass state-correctness checks.
    """
    import torch
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or a CUDA device for adapter checks")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    references, writes, reads = {}, [], []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for index, dtype in enumerate((torch.float32, torch.float16, torch.bfloat16)):
            key = hashlib.sha256(f"adapter-contract-case-{index}".encode()).hexdigest()
            # Transpose makes the incoming mapping non-contiguous. Values are
            # exactly representable in all three dtypes; no tolerance is needed.
            tensors = {f"{layer}.{kind}": (torch.arange(272).reshape(1, 2, 8, 17) % 31 + index + layer).to(
                device=device, dtype=dtype).transpose(-1, -2)
                for layer in range(2) for kind in ("key", "value")}
            expected = specs(tensors)
            metadata = {"schema": 1, "purpose": "synthetic adapter contract; not a model result",
                "prefix_tokens": 17, "case": index, "tensor_spec": expected,
                "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}
            independent = json.loads(json.dumps(metadata))
            expected = independent["tensor_spec"]
            begin = time.perf_counter()
            receipt = checked_receipt(backend.write(key, tensors, metadata), "write")
            if specs(tensors) != expected:
                raise ValueError("Backend write modified the supplied tensors")
            # Private immutable copies prevent backend mutation of the input
            # metadata from changing the independent reference.
            references[index] = {"key": key, "spec": expected,
                                 "metadata": independent}
            writes.append({"case": index, "receipt": receipt, "wall_seconds": time.perf_counter() - begin})
            del tensors
        gc.collect()
        for index in (2, 0, 1, 2, 1, 0):
            ref = references[index]
            begin = time.perf_counter()
            result = backend.read(ref["key"], str(device))
            if not isinstance(result, Restored):
                raise ValueError("Backend read must return Restored")
            receipt = checked_receipt(result.receipt, "read")
            if result.metadata != ref["metadata"]:
                raise ValueError("Restored metadata differs from the original key")
            if any(t.device != device for t in result.tensors.values()):
                raise ValueError("Backend returned a tensor on the wrong device")
            if specs(result.tensors) != ref["spec"]:
                raise ValueError("Restored tensor names, shape, dtype or exact bytes differ")
            reads.append({"case": index, "receipt": receipt, "wall_seconds": time.perf_counter() - begin})
            del result
        missing = hashlib.sha256(b"adapter-contract-intentionally-missing-key").hexdigest()
        try:
            backend.read(missing, str(device))
        except (FileNotFoundError, KeyError):
            pass
        else:
            raise ValueError("Missing key must raise FileNotFoundError or KeyError, never return stale state")
        report = {"schema": 1, "correctness_passed": True, "performance_run": False,
            "transport_qualified": False, "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device), "backend": backend.describe(), "backend_source": backend_identity,
            "checks": {"distinct_keys": 3, "repeated_reads": 6, "float32_float16_bfloat16": True,
                "non_contiguous_input": True, "exact_tensor_hashes": True,
                "shape_and_dtype": True, "metadata_identity": True, "missing_key_rejected": True},
            "writes": writes, "reads": reads,
            "limits": ["No model weights or output generation used", "No GPU speedup claim",
                "Backend receipts require independent transport evidence",
                "This does not inject auth failures, corruption, timeouts or partial transfers",
                "Objects remain in this fresh run namespace; cleanup is an operator action"]}
        write_json(output / "backend-check.json", report)
        return report
    except BaseException as exc:
        # An external SDK exception can contain a private endpoint or signed URL.
        write_json(output / "failure.json", {"passed": False, "type": type(exc).__name__,
            "message": "Backend contract check failed; inspect local exception without publishing credentials"})
        raise
