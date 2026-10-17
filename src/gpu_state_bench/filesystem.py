"""The sole included storage implementation: Linux files with temporary RAM staging."""
import hashlib
import json
from pathlib import Path
import re
import time
from .backend import Backend, Restored
from .file_io import append_audit, transfer


class FilesystemBackend(Backend):
    def __init__(self, root, options):
        super().__init__(root, options)
        if set(options) - {"purged"}:
            raise ValueError("Unknown filesystem option")
        self.purged = options.get("purged", True)
        if not isinstance(self.purged, bool):
            raise ValueError("purged must be a boolean")
        self.root.mkdir(mode=0o700)

    def describe(self):
        return {"name": "filesystem", "payload_transport": "filesystem through temporary CPU RAM to GPU",
            "purged": self.purged, "reusable_host_payload_cache": False,
            "server_or_device_cache_flushed": False}

    def paths(self, key):
        if not re.fullmatch("[a-f0-9]{64}", key):
            raise ValueError("State key must be a SHA-256 hex string")
        return self.root / (key + ".safetensors"), self.root / (key + ".json")

    def write(self, key, tensors, metadata):
        from safetensors.torch import save
        start = time.perf_counter()
        payload = save({k: v.detach().cpu().contiguous() for k, v in tensors.items()})
        staging_seconds = time.perf_counter() - start
        path, sidecar = self.paths(key)
        _, receipt = transfer(path, payload=payload, purged=self.purged)
        with sidecar.open("x", encoding="utf8") as f:
            json.dump({"schema": 1, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                       "metadata": metadata}, f, allow_nan=False)
        receipt["serialization_and_device_to_host_seconds"] = staging_seconds
        append_audit(self.root / "io.jsonl", key, receipt)
        return receipt

    def read(self, key, device):
        import torch
        from safetensors.torch import load
        path, sidecar = self.paths(key)
        control = json.loads(sidecar.read_text())
        if control.get("schema") != 1:
            raise ValueError("Unknown stored state schema")
        payload, receipt = transfer(path, purged=self.purged)
        start = time.perf_counter()
        if hashlib.sha256(payload).hexdigest() != control["payload_sha256"]:
            raise ValueError("State file SHA-256 mismatch")
        receipt["payload_hash_check_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        host = load(payload)
        receipt["deserialize_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        tensors = {name: value.to(device) for name, value in host.items()}
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        receipt["host_to_device_seconds"] = time.perf_counter() - start
        append_audit(self.root / "io.jsonl", key, receipt)
        return Restored(tensors, control["metadata"], receipt)
