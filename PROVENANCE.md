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
