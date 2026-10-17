# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from gpu_state_bench.backend import load_backend
from gpu_state_bench.file_io import transfer
from gpu_state_bench.filesystem import FilesystemBackend
from gpu_state_bench.runner import distribution

HAS_MODEL_LIBS = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "safetensors"))


class Contracts(unittest.TestCase):
    def test_distribution_keeps_sample_counts_and_units(self):
        result = distribution([1, 2, 3, 4])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["p50"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)

    def test_empty_distribution_rejected(self):
        with self.assertRaises(ValueError):
            distribution([])

    def test_filesystem_never_accepts_path_traversal_or_unknown_options(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                FilesystemBackend(Path(d) / "bad", {"endpoint": "unimplemented"})
            backend = FilesystemBackend(Path(d) / "state", {"purged": False})
            with self.assertRaises(ValueError):
                backend.paths("../other-file")

    def test_existing_payload_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state"
            transfer(p, payload=b"state", purged=False)
            with self.assertRaises(FileExistsError):
                transfer(p, payload=b"replacement", purged=False)
            self.assertEqual(p.read_bytes(), b"state")

    def test_buffered_diagnostic_records_actual_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state"
            _, write = transfer(p, payload=b"payload" * 5000, purged=False)
            data, read = transfer(p, purged=False)
            self.assertEqual(data, b"payload" * 5000)
            self.assertEqual(read["bytes"], write["bytes"])
            self.assertEqual(read["mode"], "buffered-diagnostic")
            self.assertIsNone(read["resident_pages_after"])

    @unittest.skipUnless(sys.platform == "linux", "Real Linux purge is checked on Linux CI")
    def test_actual_linux_reads_and_writes_purge_owned_pages(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state"
            payload = b"q" * (1024**2 + 71)
            _, write = transfer(p, payload=payload)
            data, read = transfer(p)
            self.assertEqual(data, payload)
            for row in (write, read):
                self.assertEqual(row["resident_pages_before"], 0)
                self.assertEqual(row["resident_pages_after"], 0)
                self.assertEqual(row["bytes"], len(payload))

    def test_third_party_backend_is_explicit_and_missing_module_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_backend("not-a-configured-backend", Path(d) / "state", {})
            with self.assertRaises(ModuleNotFoundError):
                load_backend("uninstalled_external_package:Backend", Path(d) / "state", {})


@unittest.skipUnless(HAS_MODEL_LIBS, "Install .[run] for numerical/model integration checks")
class TensorAndModelIntegration(unittest.TestCase):
    def test_qwen2_full_attention_roundtrip_and_sliding_rejection(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from gpu_state_bench.engine import Engine
        from gpu_state_bench.runner import prepare, measure
        torch.set_num_threads(1)
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, use_sliding_window=False)
        model = Qwen2ForCausalLM(cfg)
        engine = Engine(model, "cpu", 8)
        case = {"id": 0, "prefix": [1, 2, 3, 4] * 4, "suffix": [5, 6]}
        with tempfile.TemporaryDirectory() as d:
            backend = FilesystemBackend(Path(d) / "state", {"purged": False})
            reference = prepare(engine, backend, case, "b" * 64, 3)
            row = measure(engine, backend, case, reference, "restore", 3)
            self.assertTrue(row["validated"])
        model.config.use_sliding_window = True
        with self.assertRaisesRegex(ValueError, "Sliding-window"):
            Engine(model, "cpu")

    def test_safetensors_roundtrip_and_corruption_detection(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            backend = FilesystemBackend(Path(d) / "state", {"purged": sys.platform == "linux"})
            key = hashlib.sha256(b"fixture").hexdigest()
            original = {"0.key": torch.arange(48).reshape(1, 2, 6, 4).float()}
            meta = {"model": "synthetic-test", "prefix_tokens": 6}
            backend.write(key, original, meta)
            result = backend.read(key, "cpu")
            self.assertTrue(torch.equal(result.tensors["0.key"], original["0.key"]))
            self.assertEqual(result.metadata, meta)
            path, _ = backend.paths(key)
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                backend.read(key, "cpu")

    def test_end_to_end_uses_real_cache_not_saved_answer_and_makes_no_cpu_speed_claim(self):
        from gpu_state_bench.cli import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "new-parent" / "output"
            with patch.object(sys, "argv", ["gpu-state-bench", "smoke", "--output", str(output), "--state-root", str(Path(d) / "state")]):
                main()
            report = json.loads((output / "summary.json").read_text())
            self.assertTrue(report["correctness_passed"])
            self.assertFalse(report["performance_run"])
            self.assertIsNone(report["median_ttft_speedup"])
            self.assertEqual(report["storage_totals"]["read_operations"], 2)
            self.assertGreater(report["storage_totals"]["read_bytes"], 0)
            rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r["generated_tokens"] == 4 and r["validated"] for r in rows))

    def test_wrong_state_bytes_fail_even_when_metadata_looks_correct(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from gpu_state_bench.engine import Engine
        from gpu_state_bench.runner import prepare, measure
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64))
        engine = Engine(model, "cpu", 8)
        case = {"id": 0, "prefix": [1, 2, 3, 4] * 4, "suffix": [5, 6]}
        with tempfile.TemporaryDirectory() as d:
            backend = FilesystemBackend(Path(d) / "state", {"purged": False})
            reference = prepare(engine, backend, case, "a" * 64, 3)
            actual_read = backend.read
            def bad_read(key, device):
                restored = actual_read(key, device)
                restored.tensors = {k: v + 1 for k, v in restored.tensors.items()}
                return restored
            backend.read = bad_read
            with self.assertRaises(ValueError):
                measure(engine, backend, case, reference, "restore", 3)


if __name__ == "__main__":
    unittest.main()
