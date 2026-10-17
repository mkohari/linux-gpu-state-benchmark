"""New synthetic GBM/Asian-option experiment; no historical FSI kernel or data."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import random
import time

from .engine import tensor_hashes
from .runner import distribution, write_json


@dataclass(frozen=True)
class Simulation:
    paths: int = 65536
    steps: int = 252
    scenarios: int = 2
    repeats: int = 3
    seed: int = 20261017
    spot: float = 100.0
    rate: float = .03
    volatility: float = .20
    years: float = 1.0
    strikes: tuple = (95.0, 105.0)

    def validate(self):
        if self.paths < 2 or self.steps < 1 or self.scenarios < 1 or self.repeats < 1:
            raise ValueError("Need >=2 paths and positive steps, scenarios and repeats")
        if self.seed < 0 or self.seed + 1009 * (self.scenarios - 1) >= 2**63:
            raise ValueError("Scenario seeds must fit positive signed 64-bit integers")
        if not all(math.isfinite(x) for x in (self.spot, self.rate, self.volatility, self.years, *self.strikes)):
            raise ValueError("Simulation parameters must be finite")
        if min(self.spot, self.volatility, self.years, *self.strikes) <= 0 or not self.strikes:
            raise ValueError("Spot, volatility, years and strikes must be positive")


def sync(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def release(device):
    import torch
    gc.collect()
    sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def simulate(config, seed, device):
    """Exact GBM at equally spaced dates under a synthetic risk-neutral measure."""
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)
    dt = config.years / config.steps
    noise = torch.randn((config.paths, config.steps), generator=generator,
                        dtype=torch.float64, device=device)
    noise.mul_(config.volatility * math.sqrt(dt)).add_((config.rate - .5 * config.volatility**2) * dt)
    prices = noise.cumsum(dim=1)
    del noise
    return prices.exp_().mul_(config.spot)


def price(config, state):
    """Each strike asks a fresh Asian-call question using all monitoring dates."""
    import torch
    averages = state.mean(dim=1)
    discount = math.exp(-config.rate * config.years)
    answers = []
    for strike in config.strikes:
        payoff = (averages - strike).clamp_min(0) * discount
        answers.append({"strike": strike, "asian_call_price": payoff.mean().item(),
                        "standard_error": (payoff.std(unbiased=True) / math.sqrt(config.paths)).item()})
    return answers


def analytic_control(config, state):
    """European terminal payoff has a closed form; Asian payoff does not here."""
    import torch
    strike = config.spot
    sigma_t = config.volatility * math.sqrt(config.years)
    d1 = (math.log(config.spot / strike) + (config.rate + .5 * config.volatility**2) * config.years) / sigma_t
    d2 = d1 - sigma_t
    normal = lambda z: .5 * (1 + math.erf(z / math.sqrt(2)))
    expected = config.spot * normal(d1) - strike * math.exp(-config.rate * config.years) * normal(d2)
    payoff = (state[:, -1] - strike).clamp_min(0) * math.exp(-config.rate * config.years)
    estimate = payoff.mean().item()
    error = (payoff.std(unbiased=True) / math.sqrt(config.paths)).item()
    if not torch.isfinite(state).all().item() or not (state > 0).all().item():
        raise ValueError("Non-finite or non-positive simulated price")
    # A stochastic sanity gate, not an accuracy guarantee or market calibration.
    if abs(estimate - expected) > max(6 * error, 1e-9):
        raise ValueError("Terminal European-call control is outside six estimated standard errors")
    return {"strike": strike, "black_scholes_price": expected, "monte_carlo_price": estimate,
            "standard_error": error, "tolerance_standard_errors": 6, "passed": True}


def check_receipt(receipt, operation, purged):
    if receipt.get("operation") != operation or type(receipt.get("bytes")) is not int or receipt["bytes"] <= 0:
        raise ValueError("Backend must report actual positive read/write byte counts")
    if purged and any((receipt.get("mode") != "purged", receipt.get("resident_pages_before") != 0,
                       receipt.get("resident_pages_after") != 0)):
        raise ValueError("Missing zero-residency filesystem purge evidence")


def run(config, backend, identity, output, device="cuda:0", cpu_demo=False):
    import torch
    config.validate()
    if cpu_demo:
        if torch.device(device).type != "cpu":
            raise ValueError("CPU demo requires the CPU")
    elif platform.system() != "Linux" or torch.device(device).type != "cuda" or not torch.cuda.is_available() or config.repeats < 3:
        raise ValueError("Performance runs require Linux, CUDA and >=3 repeats; use --cpu-demo for a small correctness check")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    started = datetime.now(timezone.utc).isoformat()
    purged = backend.describe().get("name") == "filesystem" and not cpu_demo
    if purged and backend.describe().get("purged") is not True:
        raise ValueError("GPU filesystem measurements require purged I/O")
    rows, population = [], []
    try:
        warm = simulate(config, config.seed, device)
        price(config, warm)
        del warm
        release(device)
        for scenario in range(config.scenarios):
            scenario_seed = config.seed + 1009 * scenario
            start = time.perf_counter()
            state = simulate(config, scenario_seed, device)
            sync(device)
            compute_seconds = time.perf_counter() - start
            control = analytic_control(config, state)
            expected = price(config, state)
            check_start = time.perf_counter()
            fingerprints = tensor_hashes({"prices": state})
            verification_seconds = time.perf_counter() - check_start
            metadata = {"schema": 1, "experiment": "synthetic-gbm-asian-call-v1", "config": json.loads(json.dumps(asdict(config))),
                        "seed": scenario_seed, "shape": list(state.shape), "dtype": str(state.dtype),
                        "tensor_sha256": fingerprints, "tensor_bytes": state.numel() * state.element_size()}
            key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            start = time.perf_counter()
            receipt = backend.write(key, {"prices": state}, metadata)
            sync(device)
            write_seconds = time.perf_counter() - start
            check_receipt(receipt, "write", purged)
            population.append({"scenario": scenario, "key": key, "metadata": metadata,
                "expected_prices": expected, "analytic_control": control, "simulation_seconds": compute_seconds,
                "verification_seconds": verification_seconds, "write_seconds": write_seconds, "write_receipt": receipt})
            del state
            release(device)
        write_json(output / "population.json", population)
        jobs = [(repeat, scenario, mode) for repeat in range(config.repeats)
                for scenario in range(config.scenarios) for mode in ("recompute", "restore")]
        random.Random(config.seed).shuffle(jobs)
        with (output / "samples.jsonl").open("x") as log:
            for repeat, scenario, mode in jobs:
                ref = population[scenario]
                release(device)
                begin = time.perf_counter()
                receipt = None
                if mode == "recompute":
                    state = simulate(config, ref["metadata"]["seed"], device)
                else:
                    restored = backend.read(ref["key"], device)
                    if restored.metadata != ref["metadata"] or set(restored.tensors) != {"prices"}:
                        raise ValueError("Wrong numerical state identity")
                    state = restored.tensors["prices"]
                    receipt = restored.receipt
                    check_receipt(receipt, "read", purged)
                    del restored
                if list(state.shape) != ref["metadata"]["shape"] or state.dtype != torch.float64 or state.device != torch.empty(0, device=device).device:
                    raise ValueError("Wrong numerical state shape, dtype or target device")
                sync(device)
                ready = time.perf_counter() - begin
                answers = price(config, state)
                sync(device)
                complete = time.perf_counter() - begin
                peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
                check_start = time.perf_counter()
                if tensor_hashes({"prices": state}) != ref["metadata"]["tensor_sha256"] or answers != ref["expected_prices"]:
                    raise ValueError("Numerical state or repricing differs from the exact reference")
                verification = time.perf_counter() - check_start
                row = {"scenario": scenario, "seed": ref["metadata"]["seed"], "repeat": repeat, "mode": mode,
                       "state_ready_seconds": ready, "reprice_seconds": complete - ready, "query_seconds": complete,
                       "verification_seconds": verification, "prices": answers, "generated_tokens": 0,
                       "peak_allocated_gpu_bytes": peak, "storage_receipt": receipt, "validated": True}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                rows.append(row)
                del state
        stats = {mode: {metric: distribution([r[metric] for r in rows if r["mode"] == mode])
                       for metric in ("state_ready_seconds", "reprice_seconds", "query_seconds", "verification_seconds")}
                 for mode in ("recompute", "restore")}
        receipts = [p["write_receipt"] for p in population] + [r["storage_receipt"] for r in rows if r["mode"] == "restore"]
        gpu = torch.cuda.get_device_properties(device) if torch.device(device).type == "cuda" else None
        report = {"schema": 1, "workload": "synthetic-gbm-asian-call", "correctness_passed": True,
                  "performance_run": not cpu_demo, "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "configuration": asdict(config), "state_bytes_per_scenario": config.paths * config.steps * 8,
                  "corpus_tensor_bytes": config.paths * config.steps * 8 * config.scenarios,
                  "summary": stats, "median_query_speedup": None if cpu_demo else stats["recompute"]["query_seconds"]["p50"] / stats["restore"]["query_seconds"]["p50"],
                  "total_measured_requests": len(rows), "measured_requests_per_mode": config.scenarios * config.repeats,
                  "generated_tokens": 0, "backend": backend.describe(), "backend_source": identity,
                  "storage_totals": {"read_bytes": sum(r["bytes"] for r in receipts if r["operation"] == "read"),
                      "write_bytes": sum(r["bytes"] for r in receipts if r["operation"] == "write"),
                      "purge_seconds": sum(r.get("purge_seconds", 0) for r in receipts)},
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                  "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda, "device": device, "gpu": None if gpu is None else {"name": gpu.name, "total_memory_bytes": gpu.total_memory}},
                  "limits": ["Synthetic one-asset GBM; no real equity, market dataset, Heston model or calibration",
                      "Sequential matched requests with explicit state release; no natural eviction or overlap claim",
                      "Saved state is all FP64 price paths, not an LLM token cache or saved answer",
                      "Every query calculates both strikes anew; output has zero language-model tokens",
                      "Seed reproducibility is checked within this software/device environment, not across CPU/CUDA versions",
                      "Full tensor diagnostics copy to CPU outside query time; filesystem file checks remain inside restore time",
                      "External transport is not independently qualified; speedup is measured, never guaranteed"]}
        write_json(output / "summary.json", report)
        render_report(report, output / "report.html")
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"passed": False, "type": type(exc).__name__, "message": str(exc)})
        raise


def render_report(report, path):
    ratio = report["median_query_speedup"]
    claim = "CPU correctness only; no GPU speedup claim." if ratio is None else f"Median recompute / restore query time: {ratio:.2f}×. Below 1 means restore was slower."
    rows = "".join(f"<tr><th>{mode}</th><td>{report['summary'][mode]['query_seconds']['n']}</td>" +
        "".join(f"<td>{report['summary'][mode]['query_seconds'][p]*1000:,.2f} ms</td>" for p in ("p50", "p95")) + "</tr>" for mode in ("recompute", "restore"))
    Path(path).write_text("<!doctype html><meta charset='utf-8'><title>Monte Carlo state reuse</title>"
        "<style>body{font:18px system-ui;max-width:1000px;margin:48px auto;padding:24px;color:#202a35}th,td{padding:16px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;font-size:13px}</style>"
        "<h1>Monte Carlo state reuse</h1><p>" + claim + "</p><p>Each request obtains simulated price paths and calculates two Asian-call prices. No language-model tokens are generated.</p>"
        "<table><tr><th>Path</th><th>Requests</th><th>Query p50</th><th>Query p95</th></tr>" + rows + "</table>"
        "<p>Exact state hashes, exact repricing and a separate European-call analytic sanity check passed. State population and verification are reported separately.</p>"
        "<details><summary>All measurements and limits</summary><pre>" + html.escape(json.dumps(report, indent=2)) + "</pre></details>")
