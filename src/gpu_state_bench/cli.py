import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid
from .backend import load_backend
from .engine import Engine, cases, snapshot_identity, tensor_hashes
from .runner import run, render_report


def parser():
    p = argparse.ArgumentParser(description="Compare GPU prefix recomputation with saved KV state. No vendor SDK included.")
    subs = p.add_subparsers(dest="command", required=True)
    m = subs.add_parser("run", help="Actual Linux/CUDA measurement, not a guaranteed speedup")
    m.add_argument("--model", required=True, type=Path, help="Existing local safetensors model/tokenizer snapshot")
    m.add_argument("--output", required=True, type=Path, help="Fresh results directory")
    m.add_argument("--state-root", required=True, type=Path, help="Filesystem mount or external backend working directory")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    m.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    m.add_argument("--contexts", type=int, default=8)
    m.add_argument("--prefix-tokens", type=int, default=4096)
    m.add_argument("--chunk-tokens", type=int, default=2048)
    m.add_argument("--new-tokens", type=int, default=8)
    m.add_argument("--repeats", type=int, default=3)
    m.add_argument("--order-seed", type=int, default=1234)
    m.add_argument("--backend", default="filesystem", help="filesystem or installed external.module:Class")
    m.add_argument("--backend-options", type=Path, help="Private JSON options file, never copied into reports")
    s = subs.add_parser("smoke", help="Offline tiny random model; correctness only, never a performance claim")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--state-root", required=True, type=Path)
    r = subs.add_parser("report", help="Render a saved summary as HTML")
    r.add_argument("summary", type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == "report":
        report = json.loads(args.summary.read_text())
        render_report(report, args.summary.with_name("report.html"))
        return
    if args.output.exists():
        p.error("Use a fresh output directory; existing results are never overwritten")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(9)
    torch.set_num_threads(1)
    smoke = args.command == "smoke"
    if not smoke:
        if sys.platform != "linux" or torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            p.error("Performance measurements require Linux and an available CUDA device; use smoke for CPU correctness")
        if args.contexts < 2 or args.repeats < 3 or args.prefix_tokens < 32 or args.new_tokens < 1 or args.chunk_tokens < 1:
            p.error("Use >=2 contexts, >=3 repetitions, >=32 prefix tokens and positive token budgets")
        if not args.model.is_dir():
            p.error("--model must be an existing local snapshot")
        cfg = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        if cfg.model_type not in ("llama", "qwen2"):
            p.error("Only full-attention Llama/Qwen2 are qualified by this engine")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        case_list = cases(tokenizer, args.contexts, args.prefix_tokens)
        maximum = max(len(c["prefix"]) + len(c["suffix"]) + args.new_tokens for c in case_list)
        if maximum > cfg.max_position_embeddings:
            p.error("Requested tokens exceed the model position limit")
        print("Hashing the local model snapshot outside measurement...", flush=True)
        identity = snapshot_identity(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention).to(args.device)
        engine = Engine(model, args.device, args.chunk_tokens)
        repeats, new_tokens = args.repeats, args.new_tokens
        options = json.loads(args.backend_options.read_text()) if args.backend_options else {}
        if not isinstance(options, dict):
            p.error("Backend options must be a JSON object")
        if args.backend == "filesystem" and options.get("purged", True) is not True:
            p.error("Performance filesystem runs require purged=true")
        spec = args.backend
    else:
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        model = LlamaForCausalLM(cfg)
        engine = Engine(model, "cpu", chunk_tokens=16)
        identity = {"kind": "random offline correctness fixture; no trained model",
            "sha256": hashlib.sha256(json.dumps(tensor_hashes(dict(model.named_parameters())), sort_keys=True).encode()).hexdigest()}
        case_list = [{"id": i, "prefix": [1, 10 + i] + [4, 5, 6, 7] * 8, "suffix": [8, 9, 3]} for i in range(2)]
        repeats, new_tokens, options, spec = 1, 4, {"purged": sys.platform == "linux"}, "filesystem"
    args.state_root.mkdir(parents=True, exist_ok=True)
    if spec == "filesystem":
        first = engine.prefill(case_list[0]["prefix"])
        size = sum(t.numel() * t.element_size() for t in engine.export(first).values())
        del first
        if shutil.disk_usage(args.state_root).free < size * len(case_list) * 1.2:
            p.error("Insufficient free filesystem space for the declared KV corpus")
    root = args.state_root / ("run-" + uuid.uuid4().hex)
    backend, backend_identity = load_backend(spec, root, options)
    report = run(engine, backend, case_list, identity, args.output, repeats, new_tokens,
        seed=getattr(args, "order_seed", 1234), smoke=smoke, backend_identity=backend_identity)
    print(json.dumps({"correctness_passed": report["correctness_passed"], "performance_run": report["performance_run"],
        "median_ttft_speedup": report["median_ttft_speedup"], "report": str(args.output / "report.html"), "state_directory": str(root)}))


if __name__ == "__main__":
    main()
