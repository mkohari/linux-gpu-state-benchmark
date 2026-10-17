import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from gpu_state_bench.filesystem import FilesystemBackend
from gpu_state_bench.numerical import Simulation, analytic_control, price, run, simulate
from gpu_state_bench.onboarding import MODEL_ID, MODEL_REVISION, download_model, guided
from gpu_state_bench.serving import stream, validate_url


class NumericalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_seed_repeats_same_paths_but_changes_shocks_not_parameters(self):
        config = Simulation(paths=4096, steps=12)
        a = simulate(config, 1234, "cpu")
        b = simulate(config, 1234, "cpu")
        c = simulate(config, 1235, "cpu")
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(a.shape, c.shape)
        self.assertTrue(analytic_control(config, a)["passed"])
        self.assertGreater(price(config, a)[0]["asian_call_price"], price(config, a)[1]["asian_call_price"])

    def test_same_shocks_converge_to_exact_deterministic_growth_with_zero_noise(self):
        config = Simulation(paths=64, steps=4)
        with patch("torch.randn", return_value=torch.zeros(64, 4, dtype=torch.float64)):
            state = simulate(config, 123, "cpu")
        dates = torch.arange(1, 5, dtype=torch.float64) / 4
        expected = config.spot * ((config.rate - .5 * config.volatility**2) * dates).exp()
        torch.testing.assert_close(state, expected.expand_as(state), rtol=1e-14, atol=0)

    def test_corrupted_restored_state_rejects_run_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FilesystemBackend(root / "state", {"purged": sys.platform == "linux"})
            original = backend.read
            def corrupt(key, device):
                result = original(key, device)
                result.tensors["prices"].add_(1)
                return result
            backend.read = corrupt
            with self.assertRaisesRegex(ValueError, "differs from the exact reference"):
                run(Simulation(paths=512, steps=8, scenarios=1, repeats=1), backend, {}, root / "output", "cpu", True)
            self.assertFalse((root / "output" / "summary.json").exists())
            self.assertFalse(json.loads((root / "output" / "failure.json").read_text())["passed"])

    def test_cpu_numerical_reports_balanced_counts_actual_bytes_and_no_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FilesystemBackend(root / "state", {"purged": sys.platform == "linux"})
            report = run(Simulation(paths=512, steps=8, scenarios=2, repeats=2), backend, {}, root / "output", "cpu", True)
            self.assertTrue(report["correctness_passed"])
            self.assertFalse(report["performance_run"])
            self.assertIsNone(report["median_query_speedup"])
            self.assertEqual(report["total_measured_requests"], 8)
            self.assertEqual(report["measured_requests_per_mode"], 4)
            self.assertEqual(report["state_bytes_per_scenario"], 512 * 8 * 8)
            self.assertEqual(report["generated_tokens"], 0)
            self.assertGreater(report["storage_totals"]["read_bytes"], 0)
            rows = [json.loads(line) for line in (root / "output" / "samples.jsonl").read_text().splitlines()]
            self.assertEqual({r["seed"] for r in rows}, {20261017, 20262026})
            if sys.platform == "linux":
                for row in rows:
                    if row["mode"] == "restore":
                        self.assertEqual(row["storage_receipt"]["resident_pages_after"], 0)

    def test_gpu_claim_cannot_be_made_by_cpu_path(self):
        with self.assertRaisesRegex(ValueError, "Linux, CUDA"):
            run(Simulation(), None, {}, Path("not-created"), "cpu", False)


class SetupChecks(unittest.TestCase):
    def test_download_pins_revision_keeps_license_and_resumes_only_its_own_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model"
            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], MODEL_ID)
                self.assertEqual(kwargs["revision"], MODEL_REVISION)
                self.assertNotIn("*.py", kwargs["allow_patterns"])
                for name in ("config.json", "tokenizer_config.json", "model.safetensors", "LICENSE"):
                    (target / name).write_text("fixture")
            with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as download:
                download_model(target)
                download_model(target)
                self.assertEqual(download.call_count, 2)
            marker = json.loads((target / "BENCHMARK_MODEL.json").read_text())
            self.assertEqual(marker["revision"], MODEL_REVISION)
            other = Path(directory) / "other"
            other.mkdir()
            (other / "my-data").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "contains other files"):
                download_model(other)
            self.assertEqual((other / "my-data").read_text(), "preserve")

    def test_guided_failure_cannot_leave_a_completed_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("gpu_state_bench.onboarding.doctor", return_value={"ready": True}), \
                 patch("gpu_state_bench.onboarding.subprocess.run", side_effect=RuntimeError("fixture failure")):
                with self.assertRaises(RuntimeError):
                    guided(root / "output", root / "state", root / "unused", cpu=True)
            self.assertFalse(json.loads((root / "output" / "suite.json").read_text())["completed"])
            self.assertFalse((root / "output" / "index.html").exists())


class StreamingChecks(unittest.TestCase):
    @contextlib.contextmanager
    def endpoint(self, truncated=False, wrong_counts=False):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert self.path == "/v1/completions"
                assert payload["ignore_eos"] and payload["stream_options"]["include_usage"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [{"choices": [{"text": " ", "finish_reason": None}]},
                          {"choices": [{"text": "answer", "finish_reason": "length"}]},
                          {"choices": [], "usage": {"prompt_tokens": len(payload["prompt"]),
                              "completion_tokens": payload["max_tokens"] + int(wrong_counts)}}]
                for event in events:
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                if not truncated:
                    self.wfile.write(b"data: [DONE]\n\n")
        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield "http://127.0.0.1:" + str(server.server_port) + "/v1"
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_real_http_stream_counts_text_completion_and_never_claims_kv_restore(self):
        with self.endpoint() as url:
            result = stream(url, "fixture", [1, 2, 3], 8)
        self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 8})
        self.assertEqual(result["generated_text"], " answer")
        self.assertGreater(result["first_visible_text_seconds"], 0)
        self.assertGreaterEqual(result["completion_seconds"], result["first_visible_text_seconds"])
        self.assertNotIn("speedup", result)

    def test_partial_stream_or_wrong_token_budget_is_a_failure(self):
        for options in ({"truncated": True}, {"wrong_counts": True}):
            with self.endpoint(**options) as url:
                with self.assertRaises(ValueError):
                    stream(url, "fixture", [1, 2], 8)

    def test_embedded_credentials_and_non_http_urls_rejected(self):
        for url in ("file:///tmp/foo", "http://user:secret@host/v1", "http://host/v1?token=private"):
            with self.assertRaises(ValueError):
                validate_url(url)
