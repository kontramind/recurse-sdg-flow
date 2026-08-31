# recurse-sdg-flow

Recursive synthetic data generation pipeline, built with Prefect.

This is a clean, standalone port of the SDG pipeline used in [paper reference
TBD]. It is being assembled incrementally:

1. ✅ Project scaffolding
2. ⏳ Data-prep CLI (`prepare_step7.py`) — builds train/test/population splits
   + encoding config from a source population file
3. ⏳ Minimal Prefect pipeline: encode → train → generate
4. ⏳ Recursive multi-generation loop
5. ⏳ Evaluation stages (statistical, privacy, detection, hallucination, TSTR)

## Setup

```bash
uv sync
```
