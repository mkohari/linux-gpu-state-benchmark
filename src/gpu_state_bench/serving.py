"""Optional HTTP serving probe; never represented as a filesystem KV experiment."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

from .runner import distribution, write_json


def validate_url(base_url):
    url = urllib.parse.urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an HTTP(S) base URL without embedded credentials, query or fragment")
    return base_url.rstrip("/")


def request(base_url, path, payload=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("BENCH_VLLM_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    return urllib.request.Request(base_url + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)


def stream(base_url, model, token_ids, new_tokens, timeout=300):
    payload = {"model": model, "prompt": token_ids, "max_tokens": new_tokens,
               "temperature": 0, "seed": 1234, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first_text = None
    text, usage, finished, done = "", None, False, False
    with urllib.request.urlopen(request(base_url, "/completions", payload), timeout=timeout) as response:
        for line in response:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("Serving probe exceeded its total request budget")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError("Serving endpoint returned a generation error")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                part = choice.get("text", "")
                text += part
                if first_text is None and text.strip():
                    first_text = time.perf_counter() - started
                finished = finished or choice.get("finish_reason") is not None
    complete = time.perf_counter() - started
    if not done or not finished or first_text is None or not usage:
        raise ValueError("Stream lacked completion, visible answer text or token usage")
    if usage.get("prompt_tokens") != len(token_ids) or usage.get("completion_tokens") != new_tokens:
        raise ValueError("Server token counts did not match the declared fixed request")
    return {"first_visible_text_seconds": first_text, "completion_seconds": complete,
            "usage": usage, "generated_text": text, "input_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()}


def probe(base_url, model_path, output, contexts=2, prefix_tokens=1024, new_tokens=16, repeats=3):
    from transformers import AutoTokenizer
    from .engine import cases, snapshot_identity
    if min(contexts, repeats, new_tokens) < 1 or prefix_tokens < 32:
        raise ValueError("Use positive contexts/repeats/new tokens and >=32 prefix tokens")
    base_url = validate_url(base_url)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    fixture = cases(tokenizer, contexts, prefix_tokens)
    model_identity = snapshot_identity(model_path)
    with urllib.request.urlopen(request(base_url, "/models"), timeout=30) as response:
        models = json.load(response).get("data", [])
    if len(models) != 1 or not models[0].get("id"):
        raise ValueError("Probe requires a server exposing exactly one model")
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "cases.json", fixture)
    rows = []
    try:
        with (output / "samples.jsonl").open("x") as log:
            for repeat in range(repeats):
                for case in fixture:
                    row = stream(base_url, models[0]["id"], case["prefix"] + case["suffix"], new_tokens)
                    row.update({"case": case["id"], "repeat": repeat, "prefix_tokens": len(case["prefix"]),
                                "suffix_tokens": len(case["suffix"])})
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    rows.append(row)
        report = {"schema": 1, "workload": "optional-http-generation-probe", "completed": True,
                  "filesystem_state_comparison": False, "server_model_id": models[0]["id"],
                  "client_tokenizer_snapshot": model_identity, "server_weights_verified": False,
                  "summary": {m: distribution([r[m] for r in rows]) for m in ("first_visible_text_seconds", "completion_seconds")},
                  "limits": ["HTTP generation only: no retrieval, real-news corpus, storage adapter, KV export or restore",
                      "Local tokenizer snapshot hashed; HTTP API cannot attest server weights or cache configuration",
                      "Repeated prompts may encounter server caches; no cold/revisit or storage speedup claim",
                      "First visible non-whitespace answer text includes HTTP/server queueing; not local first-token timing",
                      "Synthetic input IDs are fixed; eight/sixteen output tokens may not form a complete sentence"]}
        write_json(output / "summary.json", report)
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"passed": False, "type": type(exc).__name__, "message": "HTTP probe failed; check server availability, tokenizer identity and fixed token budget"})
        raise
