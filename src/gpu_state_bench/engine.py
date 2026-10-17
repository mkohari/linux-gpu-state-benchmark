"""Public Transformers DynamicCache integration for full-attention Llama/Qwen2."""
import hashlib
import inspect
import json
from pathlib import Path


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def snapshot_identity(path):
    """Hash actual local model/tokenizer files once, outside measured requests."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    selected = [p for p in files if p.suffix in (".json", ".safetensors", ".model", ".txt")]
    if not selected or not any(p.suffix == ".safetensors" for p in selected):
        raise ValueError("Supply a local model snapshot with safetensors weights")
    records = []
    for p in selected:
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                digest.update(chunk)
        records.append({"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest.hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": records}


def tensor_hashes(tensors):
    """Explicit CPU diagnostics, never counted as filesystem transport proof."""
    import torch
    result = {}
    for name, tensor in sorted(tensors.items()):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


class Engine:
    def __init__(self, model, device, chunk_tokens=256):
        import torch
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.torch, self.model, self.device = torch, model.eval(), str(resolved)
        self.chunk_tokens = chunk_tokens
        cfg = model.config
        if cfg.model_type not in ("llama", "qwen2"):
            raise ValueError("This engine supports full-attention Llama/Qwen2 only; add another engine explicitly")
        if getattr(cfg, "use_sliding_window", False) or (cfg.model_type == "llama" and getattr(cfg, "sliding_window", None)):
            raise ValueError("Sliding-window caches need a separately validated engine")
        if any(kind != "full_attention" for kind in (getattr(cfg, "layer_types", None) or [])):
            raise ValueError("Hybrid/sliding attention layers need a separately validated engine")
        if getattr(cfg, "quantization_config", None):
            raise ValueError("Quantized models need a separately validated engine")
        if "logits_to_keep" not in inspect.signature(model.forward).parameters:
            raise ValueError("Selected model lacks the qualified logits_to_keep interface")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")

    def sync(self):
        if self.torch.device(self.device).type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def forward(self, cache, tokens, sample=True):
        t = self.torch
        offset = cache.get_seq_length()
        ids = t.tensor([tokens], dtype=t.long, device=self.device)
        mask = t.ones((1, offset + len(tokens)), dtype=t.long, device=self.device)
        positions = t.arange(offset, offset + len(tokens), device=self.device)
        with t.inference_mode():
            result = self.model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                cache_position=positions, use_cache=True, logits_to_keep=1)
        return result.past_key_values, int(result.logits[0, -1].argmax().item()) if sample else None

    def prefill(self, tokens):
        from transformers import DynamicCache
        cache = DynamicCache()
        for offset in range(0, len(tokens), self.chunk_tokens):
            cache, _ = self.forward(cache, tokens[offset:offset + self.chunk_tokens], sample=False)
        return cache

    def export(self, cache):
        legacy = cache.to_legacy_cache()
        if len(legacy) != self.model.config.num_hidden_layers:
            raise ValueError("Unexpected cache layer count")
        return {f"{i}.{kind}": tensor for i, pair in enumerate(legacy) for kind, tensor in zip(("key", "value"), pair)}

    def restore(self, tensors, prefix_tokens):
        from transformers import DynamicCache
        layers = self.model.config.num_hidden_layers
        expected = {f"{i}.{kind}" for i in range(layers) for kind in ("key", "value")}
        if set(tensors) != expected:
            raise ValueError("Restored state has the wrong tensor names/layer count")
        for value in tensors.values():
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[-2] != prefix_tokens:
                raise ValueError("Restored state has the wrong cache shape")
            if value.device != self.torch.device(self.device):
                raise ValueError("Backend returned a tensor on the wrong device")
            if value.dtype != next(self.model.parameters()).dtype:
                raise ValueError("Backend changed cache dtype")
        return DynamicCache.from_legacy_cache(tuple((tensors[f"{i}.key"], tensors[f"{i}.value"]) for i in range(layers)))

    def first(self, cache, suffix):
        return self.forward(cache, suffix)

    def finish(self, cache, first, count):
        tokens = [first]
        for _ in range(count - 1):
            cache, next_token = self.forward(cache, [tokens[-1]])
            tokens.append(next_token)
        self.sync()
        return tokens


def cases(tokenizer, count, prefix_tokens):
    suffix = tokenizer.encode("\nQuestion: Summarize the briefing in one sentence.\nAnswer:", add_special_tokens=False)
    result = []
    for i in range(count):
        intro = tokenizer.encode(f"Synthetic briefing {i:06d}. Reference code {100000 + i}.\n", add_special_tokens=True)
        body = tokenizer.encode("This synthetic briefing discusses research capacity and the reuse of earlier calculations. ", add_special_tokens=False)
        if not body or len(intro) >= prefix_tokens:
            raise ValueError("Tokenizer or prefix budget cannot represent the synthetic fixture")
        prefix = (intro + body * ((prefix_tokens - len(intro)) // len(body) + 1))[:prefix_tokens]
        result.append({"id": i, "prefix": prefix, "suffix": suffix})
    if len({token_hash(c["prefix"]) for c in result}) != count:
        raise ValueError("Synthetic prefixes must have distinct token identities")
    return result
