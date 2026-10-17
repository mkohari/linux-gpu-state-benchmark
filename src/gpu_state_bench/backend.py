"""A tensor-state interface for separately distributed backend implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import inspect
import hashlib
from pathlib import Path
from typing import Any, Mapping


@dataclass
class Restored:
    tensors: Mapping[str, Any]
    metadata: dict
    receipt: dict


class Backend(ABC):
    """write must finish before returning; read must return ready device tensors.

    Tensor values are PyTorch tensors, intentionally typed without importing GPU
    software. Implementations own their transport and must report its real path.
    No implementation may serve a retained CPU/GPU payload as an external read.
    """
    def __init__(self, root: Path, options: dict):
        self.root, self.options = Path(root), options

    @abstractmethod
    def describe(self) -> dict:
        """Public transport identity, excluding private credentials/options."""

    @abstractmethod
    def write(self, key: str, tensors: Mapping[str, Any], metadata: dict) -> dict:
        """Persist the exact tensor bytes and metadata. Return byte/cost evidence."""

    @abstractmethod
    def read(self, key: str, device: str) -> Restored:
        """Read the saved state anew; synchronize all device copies before return."""


def load_backend(spec, root, options):
    if spec == "filesystem":
        from .filesystem import FilesystemBackend
        cls = FilesystemBackend
    else:
        module, separator, name = spec.partition(":")
        if not separator or not module or not name:
            raise ValueError("Backend must be filesystem or an installed module:Class")
        cls = getattr(importlib.import_module(module), name)
    if not inspect.isclass(cls) or not issubclass(cls, Backend):
        raise ValueError("External backend must implement the Backend interface")
    source = Path(inspect.getfile(cls))
    return cls(Path(root), options), {
        "class": cls.__module__ + ":" + cls.__name__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
