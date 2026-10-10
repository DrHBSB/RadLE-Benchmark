# Open-model reference calls

These scripts send one ordered case to a server that you have already installed and started. They are small public adaptations of successful study routes, not the original 200-case collectors. They do not set up GPUs, download models, resume runs, repair responses, or calculate correctness. Ordinary Colab hardware is not a supported-runtime promise.

Install this repository with `pip install -e .` and supply your own images in presentation order. From the repository root:

```sh
python advanced/open_models/ollama_one_case.py --model octomed_7b --images 01.png 02.png --output prediction.csv
```

The Ollama script uses `OLLAMA_BASE_URL` or `http://localhost:11434/v1`. Its recorded study-route defaults are:

| Selection | Local model tag | Generation limit | Important qualification |
| --- | --- | ---: | --- |
| `llava_med_mistral_7b` | `z-uo/llava-med-v1.5-mistral-7b_q8_0` | 2,048 | The study route used an Ollama image-conditioned GGUF. Descriptive prose may not yield a valid JSON diagnosis. |
| `octomed_7b` | `hf.co/mradermacher/OctoMed-7B-GGUF:Q8_0` | 8,192 | This Q8 build includes a vision projector. Its reasoning text may consume the output budget or prevent valid JSON. |
| `internvl3_5_8b` | `hf.co/mradermacher/InternVL3_5-8B-GGUF:Q8_0` | 8,192 | The study checked image conditioning on this serving route; a new Ollama build needs its own check. |

The study used temperature `0.01` for these routes. The reference script sends the same prompt bytes and original ordered PNG/JPEG bytes, but uses this repository's strict JSON parser. Its output should not be compared numerically with the private historical run, which used a broader extraction and repair workflow. A server must actually load the model and vision projector before using the script; successful HTTP responses alone do not prove image conditioning.

MedGemma 1.5 4B used a separate GPU Workbench/vLLM route. The recorded completed run used `google/medgemma-1.5-4b-it`, `bfloat16`, maximum model length 4,096, and output limit 2,048. The model is gated; obtain access and serve it yourself, then call:

```sh
python advanced/open_models/medgemma_one_case.py --base-url http://127.0.0.1:8000/v1 --images 01.png --output prediction.csv
```

Both scripts make one call per invocation and write the same CSV columns as the OpenRouter quickstart. They make no claim that these models were all included in the final paper roster. In particular, LLaVA-Med's successful Ollama route is source evidence, not confirmation of its final analysis inclusion.
