# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
"""Small, explicit setup helpers. No privileged changes or automatic GPU takeover."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_MODEL = Path("models/qwen2.5-1.5b-instruct")


def doctor(state_root, cpu=False):
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    add("Python", sys.version_info >= (3, 10), platform.python_version())
    for package, pinned in (("torch", "2.9.0"), ("transformers", "4.57.1"), ("safetensors", "0.6.2")):
        try:
            version = importlib.metadata.version(package)
            add(package, version.split("+")[0] == pinned, f"Installed {version}; reference {pinned}")
        except importlib.metadata.PackageNotFoundError:
            add(package, False, "Missing: run bash scripts/setup.sh" + (" --cpu" if cpu else ""))
    add("Operating system", cpu or sys.platform == "linux", platform.system() + ("; CPU demo only" if cpu else "; GPU measurements require Linux"))
    try:
        import torch
        add("Compute device", cpu or torch.cuda.is_available(), "CPU demo" if cpu else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable: check the NVIDIA driver and install the CUDA build of PyTorch"))
        if not cpu and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            add("GPU memory information", True, f"{free/2**30:.1f} GiB free of {total/2**30:.1f}; this is information, not a model-fit guarantee")
    except (ImportError, OSError, RuntimeError):
        add("Compute device", False, "PyTorch cannot initialize; repeat setup and check its installation")
    try:
        state_root = Path(state_root)
        state_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(state_root).free
        add("Storage free space", free >= (16 * 2**20 if cpu else 2 * 2**30), f"{free/2**30:.1f} GiB free at {state_root.resolve()}")
        from .file_io import transfer
        with tempfile.TemporaryDirectory(prefix="doctor-", dir=state_root) as directory:
            path = Path(directory) / "owned-probe"
            payload = b"benchmark-storage-check" * 4096
            purged = sys.platform == "linux"
            _, write = transfer(path, payload=payload, purged=purged)
            restored, read = transfer(path, purged=purged)
            add("Storage readback and purge", restored == payload, "Exact readback; zero reusable file pages after I/O" if purged else "Buffered development diagnostic; Linux required for purge")
    except (OSError, ValueError, RuntimeError):
        add("Storage readback and purge", False, "Target failed owned-file read/write/purge check; choose another path and consult docs/RUNBOOK.md")
    for row in checks:
        print(("PASS " if row["passed"] else "FIX  ") + row["check"] + ": " + row["detail"], flush=True)
    report = {"schema": 1, "ready": all(c["passed"] for c in checks), "cpu_demo": cpu, "checks": checks,
              "limits": "Readiness is not GPU performance validation, a guarantee of model fit, or proof that appliance caches are cold"}
    return report


def download_model(destination):
    from huggingface_hub import snapshot_download
    destination = Path(destination)
    marker = destination / "BENCHMARK_MODEL.json"
    identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "license": "Apache-2.0; review the downloaded LICENSE and upstream model card",
                "source": "https://huggingface.co/" + MODEL_ID + "/tree/" + MODEL_REVISION}
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError("Destination contains other files; choose an empty model directory")
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("Downloading the public 1.5B model (about 3 GiB) at a fixed revision. Model weights are not in this repository.", flush=True)
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "README.md", "LICENSE"])
    for required in ("config.json", "tokenizer_config.json", "model.safetensors", "LICENSE"):
        if not (destination / required).is_file():
            raise ValueError("Model download incomplete: missing " + required + "; rerun download-model to resume")
    print("Model ready: " + str(destination.resolve()), flush=True)


def guided(output, state_root, model, cpu=False):
    from .runner import write_json
    output = Path(output) if output else Path("runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if output.exists():
        raise ValueError("Use a fresh output directory; existing results are never overwritten")
    if not cpu and not (Path(model) / "config.json").is_file():
        raise ValueError("Model not ready. Run gpu-state-bench download-model, or supply --model with your local snapshot")
    if not doctor(state_root, cpu)["ready"]:
        raise ValueError("Fix the checks above before starting; no measurement was started")
    output.mkdir(parents=True, mode=0o700)
    common = ["--state-root", str(state_root)]
    commands = [["smoke", "--output", str(output / "language"), *common],
                ["mc", "--cpu-demo", "--output", str(output / "numerical"), *common]] if cpu else [
        ["run", "--model", str(model), "--output", str(output / "language"), *common,
         "--contexts", "8", "--prefix-tokens", "4096", "--repeats", "3", "--new-tokens", "8"],
        ["mc", "--paths", "65536", "--steps", "64", "--scenarios", "2", "--repeats", "3",
         "--output", str(output / "numerical"), *common]]
    manifest = {"schema": 1, "cpu_demo": cpu, "completed": False, "commands": commands,
                "scope": "Two sequential experiments; no simultaneous workload, real-news RAG or vLLM state-offload claim"}
    try:
        for command in commands:
            print("Starting " + ("language-model state check" if command[0] in ("smoke", "run") else "Monte Carlo state check") + "...", flush=True)
            subprocess.run([sys.executable, "-m", "gpu_state_bench.cli", *command], check=True)
        reports = [json.loads((output / name / "summary.json").read_text()) for name in ("language", "numerical")]
        if not all(r["correctness_passed"] and r["performance_run"] == (not cpu) for r in reports):
            raise ValueError("A child experiment did not pass the expected validation")
        manifest["completed"] = True
        (output / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Benchmark results</title>"
            "<style>body{font:20px system-ui;max-width:850px;margin:60px auto;padding:24px;line-height:1.6}</style>"
            "<h1>" + ("CPU demonstration completed" if cpu else "GPU experiments completed") + "</h1>"
            "<p>Both experiments passed their correctness checks. " + ("This demonstration makes no GPU performance claim." if cpu else "Open each report for measured speedup, sample counts and limitations.") + "</p>"
            "<ol><li><a href='language/report.html'>Language-model prefix state</a>: processing a document again versus restoring its saved KV state.</li>"
            "<li><a href='numerical/report.html'>Monte Carlo price-path state</a>: simulating paths again versus restoring them and calculating option prices.</li></ol>"
            "<p>These run one after another. They do not reproduce the earlier simultaneous FSI benchmark. Model weights stay resident; request state is explicitly released. No natural eviction or guaranteed 3.5× result.</p>")
    finally:
        write_json(output / "suite.json", manifest)
    print("Open this results page: " + str((output / "index.html").resolve()), flush=True)
    return output
