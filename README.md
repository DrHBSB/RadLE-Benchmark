# RadLE-Benchmark

Public development workspace for the RadLE 2.0 benchmark runner. This is an early **inference-only** starter for Upasana and collaborators. It currently handles one case at a time. The full 200-case study, correctness adjudication, RadLE scores, and manuscript statistics are not implemented here.

## Start with the Python and Colab pair

Install Python 3.10 or newer, then run `pip install -e .[test]`. Set your own `OPENROUTER_API_KEY` locally and run:

```sh
python -m radle.inference --images image1.png image2.png --model muse_spark_1_3_openrouter --output predictions.csv --case-id case_001
```

The [Colab quickstart](notebooks/RadLE_OpenRouter_Quickstart.ipynb) is an alternative one-case demonstration. Its default is image upload. It reads the API key from Colab Secrets and does not send a paid request until `RUN_LIVE = True`.

The prompt is in [prompts/diagnostic.txt](prompts/diagnostic.txt), model examples in [configs/models.json](configs/models.json), and the current implementation in [src/radle](src/radle). The runner returns a diagnosis, confidence, response status, latency, model ID, and prompt hash. It does **not** assess correctness.

## Upasana's first task

Extend the Python module and Colab to accept a case manifest and ordered image folder, process the approved 50-case development cohort, and write one prediction row per case. Keep the manifest and image root configurable so the final public/private split can be applied later without changing the request contract. Preserve the exact prompt, image order, requested and returned model/provider identity, abstentions, API errors, and raw response in a private run record. Test offline with synthetic images before any paid API run.

The clinical images and case answer key are **not in this GitHub repository**. Obtain the private development folder and manifest directly from the study team. Do not add private images, case-level study results, credentials, or Colab output to Git.

## Status

This repository is a starter, not the finalized NEJM AI methods or a validated 200-case public benchmark. GPT-6 Astra's study Responses API route and batch scoring are not yet supported. See [docs/METHODS.md](docs/METHODS.md) for current limits.
