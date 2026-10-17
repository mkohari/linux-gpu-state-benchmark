# What each workload actually does

## Language prefix-state experiment

`run` uses the public Transformers engine, not vLLM. Synthetic fixtures are
constructed from reference IDs and repeated narrative; there are no downloaded
news articles, search index, embeddings or retrieval service. Each prefix has
exactly the configured token count. The question suffix adds tokens, recorded
separately; total input tokens = prefix + suffix. Each measured request generates
the configured number of new greedy tokens, even after an EOS token.

A prefix is processed once during population, its full KV tensors are written,
and a reference continuation is generated. Between measured requests the process
releases all request KV references and unused CUDA allocator blocks; model
weights remain. In shuffled balanced order it either recomputes the same prefix
or restores its saved tensors, then processes the same suffix and generates a
new answer. Every output token ID and every full prefix-tensor hash must match.
See [METHOD.md](METHOD.md) for exact timing boundaries.

There is no claim that a particular number of requests naturally evicts GPU
memory. Small default corpora can fit in GPU memory; state is released explicitly.
Use the detected card's memory name: GPU memory may be GDDR or HBM. These are
different physical memory technologies, not different benchmark policies.

## New numerical state experiment

`mc` is independently authored synthetic geometric Brownian motion (GBM), not
the earlier NVDA-calibrated Heston simulation. There is **one fictional asset**,
with initial price 100, annual rate 3%, volatility 20%, horizon 1 year, no
dividend, and no imported market observations. These are illustrative parameters.

For equally spaced dates, each step uses an independent standard-normal draw Z:

```text
dt = years / steps
log_return = (rate - 0.5 * volatility²) * dt + volatility * sqrt(dt) * Z
price(t) = initial_price * exp(sum(log_returns through t))
```

A seed initializes the pseudorandom generator. The same seed replays the same
draws in this device/software environment. Scenario i uses `base_seed + 1009*i`.
Changing the seed changes the simulated sequence of shocks; it does not change
the asset, volatility, rate or maturity. Different CPU/CUDA/library versions are
not promised bitwise identical random draws. The run verifies exact replay in
its own environment.

Each state is an FP64 matrix `[paths, steps]` of all monitored prices, excluding
time zero. Bytes = paths × steps × 8. Temporary random draws/cumulative sums add
working memory; measured peak CUDA allocation is recorded separately. The state
is saved once per scenario; the recompute path does not read it. Restore reads
the full matrix anew through the selected backend. No saved option-price answer
is returned in place of calculation.

Each request calculates two arithmetic-average Asian calls at strikes 95 and
105. For each path it averages all monitored prices, takes the positive part of
`average - strike`, discounts at the rate and averages across paths. Estimated
standard error is reported separately from runtime percentiles. Both strikes
share the state and are priced anew on every request.

Example question in words: “For these simulated paths, what are the prices of
one-year average-price calls with strikes 95 and 105?” This is a Python numerical
request, not a natural-language prompt. **There are no input/output text tokens.**

Separate correctness checks require every restored FP64 byte and repriced answer
to equal the scenario reference. A terminal European call at strike 100 is
compared with its Black–Scholes analytic price within six estimated standard
errors. This is a stochastic implementation sanity check, not a guarantee of
financial accuracy or a closed-form check of the Asian price.

With standalone defaults (`mc`): 65,536 paths, 252 dates, 2 scenarios and 3
repetitions. Each scenario state is 132,120,576 bytes = 126 MiB; the corpus is
252 MiB. A “path” contains 252 simulated dates; it is not a pricing request.
Six matched requests per mode each price both strikes. Population/warmup are
additional work, recorded separately rather than inflated into sample counts.

The starter suite deliberately selects 64 dates for a lighter first GPU run:
32 MiB per scenario, 64 MiB total. Neither default is a natural LRU spill test.

```bash
gpu-state-bench mc --state-root /mnt/my-disk/state --output runs/mc-001 \
  --paths 65536 --steps 252 --scenarios 2 --repeats 3
```

`--cpu-demo` runs 512 paths × 16 dates, 2 scenarios, 1 repeat by default. It is
correctness only; GPU performance requires Linux/CUDA and at least 3 repeats.

| Numerical timing | Includes |
|---|---|
| State ready | Simulate paths, or read/check/deserialize/purge/stage saved state until ready on the target device |
| Reprice | Calculate both Asian prices and their standard errors |
| Query | State ready + reprice, including ordinary dispatch/check overhead |
| Verification | Full state hash and exact answer comparison after query timing; full hash copies to CPU |
| Population/write | Initial simulation, correctness controls and saved-state write, outside revisit query timing |

The summary gives per-mode p50/p95/counts for all four measured timing fields.
Write costs, bytes and purge costs remain visible. A restore can be slower;
there is no mandatory speedup. Short runs are demonstrations, not tail-latency SLAs.

## Suite and optional serving scope

`start` runs the two experiments **sequentially in separate processes**. It is
not concurrent orchestration or a RAG pipeline. `vllm-probe` is separately
documented HTTP generation. It is not either storage comparison. Separate
reports prevent a serving/cache result from being mistaken for file-state reuse.
