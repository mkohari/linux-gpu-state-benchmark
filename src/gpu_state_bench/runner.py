# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
"""Matched request pairs, explicit state release, honest timing and correctness gates."""
import gc
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from .engine import token_hash, tensor_hashes


def distribution(values):
    if not values:
        raise ValueError("No measured samples")
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"n": len(values), "p50": statistics.median(values), "p95": percentile(.95)}


def release(engine):
    gc.collect()
    engine.sync()
    if engine.torch.device(engine.device).type == "cuda":
        engine.torch.cuda.empty_cache()
        engine.torch.cuda.reset_peak_memory_stats(engine.device)


def prepare(engine, backend, case, model_sha, new_tokens):
    start = time.perf_counter()
    cache = engine.prefill(case["prefix"])
    engine.sync()
    prefill_seconds = time.perf_counter() - start
    tensors = engine.export(cache)
    hashes = tensor_hashes(tensors)
    metadata = {"schema": 1, "model_sha256": model_sha, "prefix_sha256": token_hash(case["prefix"]),
        "prefix_tokens": len(case["prefix"]), "tensor_sha256": hashes,
        "tensor_bytes": sum(x.numel() * x.element_size() for x in tensors.values())}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    start = time.perf_counter()
    receipt = backend.write(key, tensors, metadata)
    write_seconds = time.perf_counter() - start
    cache, first = engine.first(cache, case["suffix"])
    expected = engine.finish(cache, first, new_tokens)
    return {"key": key, "metadata": metadata, "expected_tokens": expected,
        "prefill_seconds": prefill_seconds, "write_seconds": write_seconds, "write_receipt": receipt}


def measure(engine, backend, case, reference, mode, new_tokens):
    release(engine)
    start = time.perf_counter()
    receipt = None
    if mode == "recompute":
        cache = engine.prefill(case["prefix"])
        original = engine.export(cache)
    else:
        restored = backend.read(reference["key"], engine.device)
        if restored.metadata != reference["metadata"]:
            raise ValueError("Backend returned metadata for a different model/prefix/state")
        original = restored.tensors
        cache = engine.restore(original, len(case["prefix"]))
        receipt = restored.receipt
    cache, first = engine.first(cache, case["suffix"])
    engine.sync()
    ttft = time.perf_counter() - start
    output = engine.finish(cache, first, new_tokens)
    complete = time.perf_counter() - start
    validation_start = time.perf_counter()
    if output != reference["expected_tokens"]:
        raise ValueError("Generated token IDs differ from the matched recomputation reference")
    if tensor_hashes(original) != reference["metadata"]["tensor_sha256"]:
        raise ValueError("Full KV tensor hash differs from the prefix reference")
    validation_seconds = time.perf_counter() - validation_start
    peak = engine.torch.cuda.max_memory_allocated(engine.device) if engine.torch.device(engine.device).type == "cuda" else None
    return {"mode": mode, "case": case["id"], "ttft_seconds": ttft,
        "completion_seconds": complete, "validation_seconds": validation_seconds,
        "input_tokens": len(case["prefix"]) + len(case["suffix"]),
        "prefix_tokens": len(case["prefix"]), "suffix_tokens": len(case["suffix"]),
        "generated_tokens": len(output), "output_token_ids": output,
        "peak_allocated_gpu_bytes": peak, "storage_receipt": receipt, "validated": True}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(engine, backend, case_list, model_identity, output, repeats, new_tokens, seed=1234, smoke=False, backend_identity=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    started_utc = datetime.now(timezone.utc).isoformat()
    write_json(output / "cases.json", case_list)
    if len(case_list) < 1 or repeats < 1 or new_tokens < 1:
        raise ValueError("Positive case/repeat/output-token counts are required")
    if not smoke and (platform.system() != "Linux" or engine.torch.device(engine.device).type != "cuda" or repeats < 3):
        raise ValueError("Performance runs require Linux, CUDA and at least three repetitions")
    references, population = [], []
    try:
        # An untimed first case warms the same prefill/suffix/decode shapes.
        warm_cache = engine.prefill(case_list[0]["prefix"])
        warm_cache, first = engine.first(warm_cache, case_list[0]["suffix"])
        engine.finish(warm_cache, first, new_tokens)
        del warm_cache
        release(engine)
        for case in case_list:
            ref = prepare(engine, backend, case, model_identity["sha256"], new_tokens)
            references.append(ref)
            population.append({"case": case["id"], **ref})
            release(engine)
        write_json(output / "population.json", population)
        jobs = [(repeat, case, mode) for repeat in range(repeats)
                for case in range(len(case_list)) for mode in ("recompute", "restore")]
        random.Random(seed).shuffle(jobs)
        rows = []
        with (output / "samples.jsonl").open("x") as f:
            for repeat, index, mode in jobs:
                row = measure(engine, backend, case_list[index], references[index], mode, new_tokens)
                row["repeat"] = repeat
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                rows.append(row)
        summary = {mode: {metric: distribution([r[metric] for r in rows if r["mode"] == mode])
                    for metric in ("ttft_seconds", "completion_seconds", "validation_seconds")}
                   for mode in ("recompute", "restore")}
        manifest = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.rglob("*.py")}
        device = engine.torch.device(engine.device)
        gpu = None
        if device.type == "cuda":
            props = engine.torch.cuda.get_device_properties(device)
            gpu = {"name": props.name, "total_memory_bytes": props.total_memory,
                "uuid": str(getattr(props, "uuid", "unavailable"))}
        try:
            command = ["findmnt", "--json", "--target", str(backend.root), "--output", "FSTYPE"]
            filesystem = json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True))
        except (OSError, ValueError, subprocess.CalledProcessError):
            filesystem = None
        description = backend.describe()
        if description.get("name") == "filesystem" and not smoke:
            if not description.get("purged"):
                raise ValueError("An accepted filesystem measurement requires purged I/O")
            reads = [r["storage_receipt"] for r in rows if r["mode"] == "restore"]
            writes = [r["write_receipt"] for r in population]
            for receipt in reads + writes:
                if receipt["mode"] != "purged" or receipt["bytes"] <= 0 or receipt["resident_pages_before"] != 0 or receipt["resident_pages_after"] != 0:
                    raise ValueError("Incomplete filesystem purge/byte evidence")
        speedup = summary["recompute"]["ttft_seconds"]["p50"] / summary["restore"]["ttft_seconds"]["p50"]
        reads = [r["storage_receipt"] for r in rows if r["mode"] == "restore"]
        writes = [r["write_receipt"] for r in population]
        report = {"schema": 1, "correctness_passed": True, "performance_run": not smoke,
            "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Explicitly released prefix state; matched sequential single-request microbenchmark",
            "backend": description, "backend_source": backend_identity, "filesystem": filesystem,
            "transport_qualification": "Core filesystem byte/purge checks" if description.get("name") == "filesystem" and not smoke else "External transport unqualified by core; smoke is correctness only",
            "summary": summary, "median_ttft_speedup": None if smoke else speedup,
            "contexts": len(case_list), "repeats": repeats, "new_tokens": new_tokens,
            "storage_totals": {"read_operations": len(reads), "write_operations": len(writes),
                "read_bytes": sum(r.get("bytes", 0) for r in reads),
                "write_bytes": sum(r.get("bytes", 0) for r in writes),
                "purge_seconds": sum(r.get("purge_seconds", 0) for r in reads + writes)},
            "order_seed": seed, "model": model_identity, "source_sha256": manifest,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                "torch": engine.torch.__version__, "transformers": __import__("transformers").__version__,
                "cuda_runtime": engine.torch.version.cuda, "gpu": gpu,
                "dtype": str(next(engine.model.parameters()).dtype), "prefill_chunk_tokens": engine.chunk_tokens,
                "attention_implementation": engine.model.config._attn_implementation},
            "limits": ["No speedup threshold or fixed 3.5x claim", "No natural LRU eviction or corpus-greater-than-GPU claim",
                "Not a serving throughput, RAG, Monte Carlo or production SLA benchmark",
                "All generated token IDs and full prefix KV hashes must match the reference",
                "Full tensor-hash diagnostics copy state to CPU after measured generation",
                "Filesystem restore timing includes read, purge, file hash, deserialization and CPU-to-GPU staging",
                "External backends require independent transport qualification; correctness alone does not prove GPU-direct I/O"]}
        write_json(output / "summary.json", report)
        render_report(report, output / "report.html")
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"passed": False, "type": type(exc).__name__, "message": str(exc)})
        raise


def render_report(report, path):
    title = "CPU correctness smoke test" if not report["performance_run"] else "Linux GPU prefix state benchmark"
    ratio = report["median_ttft_speedup"]
    claim = "No GPU performance claim." if ratio is None else f"Median recompute / restore time: {ratio:.2f}×. Values below 1 mean restore was slower."
    rows = "".join("<tr><th>" + mode + "</th><td>" + str(report["summary"][mode]["ttft_seconds"]["n"]) + "</td>" +
        "".join(f"<td>{report['summary'][mode]['ttft_seconds'][p]*1000:,.2f} ms</td>" for p in ("p50", "p95")) + "</tr>" for mode in ("recompute", "restore"))
    page = "<!doctype html><meta charset='utf-8'><title>" + title + "</title><style>body{font:18px system-ui;max-width:1000px;margin:48px auto;padding:0 24px;color:#202a35}td,th{padding:15px;text-align:left;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;font-size:13px}</style>"
    page += "<h1>" + title + "</h1><p>" + claim + "</p><p>New answers from the same prefix and suffix. Lower latency is better.</p><table><tr><th>Path</th><th>Samples</th><th>First token p50</th><th>First token p95</th></tr>" + rows + "</table>"
    page += "<p>Full output-token and KV-byte checks passed. Population and verification costs remain separate. Explicit state release, not natural cache pressure.</p><details><summary>Complete measurements and limits</summary><pre>" + html.escape(json.dumps(report, indent=2)) + "</pre></details>"
    Path(path).write_text(page)
