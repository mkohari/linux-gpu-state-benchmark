# Source boundaries

Owner and project author: Moiz Kohari. Initial implementation: October 16, 2026.

The Python source, tests and documentation in this repository were newly authored
for this filesystem reference experiment. No files or commit history were copied
from an FSI application or the earlier mixed-workload benchmark. This statement
describes the source boundary; it is not a legal opinion about employment or
other agreements.

The implementation uses public PyTorch, Transformers and safetensors APIs. Their
packages are installed separately and are not vendored. Public technical references:

- [Transformers 4.57.1 cache concepts and DynamicCache](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [safetensors tensor serialization API](https://huggingface.co/docs/safetensors/api/torch)
- [Linux posix_fadvise](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
- [Linux mincore](https://man7.org/linux/man-pages/man2/mincore.2.html)
- [PyTorch random generators](https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html)
- [Public Qwen2.5-1.5B-Instruct snapshot and license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
- [Optional vLLM 0.11.0 server](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)

Version 0.2 adds newly written setup helpers, a synthetic GBM/Asian-call numerical
experiment and an HTTP serving probe. The numerical equations use illustrative
parameters and no imported equity dataset, calibration, Heston kernel or historical
result. None of that source was copied from the earlier FSI benchmark. Optional
vLLM is installed from its public distribution into a separate environment; no
private serving patch or storage connector is included. The model helper is an
explicit download, preserving the upstream model license; weights are never in
the source distribution. Dependency and model licenses remain independent of MIT.

The source distribution contains no model weights, vendor SDKs, client/plugin
binaries, copied numerical kernel, commercial EULA, private model build unit,
credential, cloud adapter or historical performance data. Vendor names in the
extension documentation identify possible user integrations, not bundled code,
endorsement or demonstrated compatibility.

No 3.5x or other trained-model GPU speedup has been established by this source
revision. Correctness smoke tests use a randomly initialized tiny model, not a
production language model. New measured performance must remain attributed to
the model, hardware, source revision, filesystem, method and output evidence of
that specific run.
