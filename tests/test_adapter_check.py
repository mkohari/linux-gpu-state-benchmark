import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from gpu_state_bench.check_backend import checked_receipt

HAS_TENSORS = all(importlib.util.find_spec(n) for n in ("torch", "safetensors"))


class Receipts(unittest.TestCase):
    def test_false_empty_nan_and_mismatched_receipts_fail(self):
        for row in ({"operation": "read", "bytes": True}, {"operation": "write", "bytes": 1},
                    {"operation": "read", "bytes": 0}, {"operation": "read", "bytes": 1, "seconds": float("nan")}):
            with self.assertRaises(ValueError):
                checked_receipt(row, "read")

    def test_receipts_are_independent_snapshots(self):
        original = {"operation": "read", "bytes": 100, "details": {"attempts": 1}}
        copy = checked_receipt(original, "read")
        original["details"]["attempts"] = 7
        self.assertEqual(copy["details"]["attempts"], 1)


@unittest.skipUnless(HAS_TENSORS, "Install .[run] for tensor adapter checks")
class AdapterChecks(unittest.TestCase):
    def backend(self, directory):
        from gpu_state_bench.filesystem import FilesystemBackend
        return FilesystemBackend(Path(directory) / "payload", {"purged": sys.platform == "linux"})

    def test_three_dtypes_multiple_keys_and_repeated_reads(self):
        from gpu_state_bench.check_backend import check
        with tempfile.TemporaryDirectory() as d:
            report = check(self.backend(d), "cpu", Path(d) / "output")
            self.assertTrue(report["correctness_passed"])
            self.assertEqual(len(report["writes"]), 3)
            self.assertEqual(len(report["reads"]), 6)
            self.assertFalse(report["performance_run"])
            self.assertFalse(report["transport_qualified"])

    def test_metadata_mutation_cannot_change_independent_reference(self):
        from gpu_state_bench.check_backend import check
        with tempfile.TemporaryDirectory() as d:
            backend = self.backend(d)
            original = backend.write
            def mutated(key, tensors, metadata):
                metadata["case"] = 999
                return original(key, tensors, metadata)
            backend.write = mutated
            with self.assertRaisesRegex(ValueError, "metadata"):
                check(backend, "cpu", Path(d) / "output")
            self.assertFalse((Path(d) / "output/backend-check.json").exists())

    def test_missing_key_cannot_return_a_previous_success(self):
        from gpu_state_bench.check_backend import check
        with tempfile.TemporaryDirectory() as d:
            backend = self.backend(d)
            original = backend.read
            previous = []
            def stale(key, device):
                try:
                    result = original(key, device)
                    previous[:] = [result]
                    return result
                except FileNotFoundError:
                    return previous[0]
            backend.read = stale
            with self.assertRaisesRegex(ValueError, "Missing key"):
                check(backend, "cpu", Path(d) / "output")

    def test_cli_loads_separately_supplied_module_and_redacts_options(self):
        from gpu_state_bench.cli import main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            module = root / "example_external_adapter.py"
            module.write_text("from gpu_state_bench.filesystem import FilesystemBackend\n"
                "class CustomBackend(FilesystemBackend):\n"
                "    def __init__(self, root, options):\n"
                "        self.private_value = options.pop('private_test_value')\n"
                "        super().__init__(root, options)\n"
                "    def describe(self):\n"
                "        return {**super().describe(), 'name': 'external-test-filesystem'}\n")
            options = root / "options.json"
            options.write_text(json.dumps({"purged": sys.platform == "linux", "private_test_value": "DO_NOT_EXPORT_TEST_VALUE"}))
            output = root / "output"
            argv = ["gpu-state-bench", "check-backend", "--backend", "example_external_adapter:CustomBackend",
                "--backend-options", str(options), "--state-root", str(root / "state"), "--output", str(output)]
            with patch.object(sys, "path", [str(root)] + sys.path), patch.object(sys, "argv", argv):
                main()
            report = (output / "backend-check.json").read_text()
            self.assertNotIn("DO_NOT_EXPORT_TEST_VALUE", report)
            self.assertIn("example_external_adapter:CustomBackend", report)
            sys.modules.pop("example_external_adapter", None)


if __name__ == "__main__":
    unittest.main()
