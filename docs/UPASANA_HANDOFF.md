# Upasana handoff: first development slice

**Starting point:** this public GitHub repository, particularly `src/radle/inference.py`, `src/radle/inputs.py`, `src/radle/parsing.py`, `notebooks/RadLE_OpenRouter_Quickstart.ipynb`, `prompts/diagnostic.txt`, and `configs/models.json`.

**Private data:** the study team will share a read-only image folder and a 50-case development manifest separately. Keep all clinical images, reference diagnoses, saved responses, API keys, and run outputs outside Git. The case split is being finalized by Subhrankar; do not infer it from the first 50 case IDs.

**First task:** extend the one-case Python and Colab paths to process the development manifest. Each case must retain stable ID and ordered image paths. Write one row per case with requested and returned model/provider identity when available, diagnosis, confidence or deferral, raw response or a private raw-response reference, status, timing, prompt hash, and run timestamp. Failures remain visible as rows; do not silently drop cases or score API errors as wrong diagnoses.

**Acceptance:** a no-network test uses synthetic images to cover one image, multiple ordered images, a valid answer, abstention, malformed JSON, and an API failure. A 50-case dry run validates manifest/image resolution and output row count without paid inference. The study team will authorize and bound any live API pilot separately.

**Later:** plug in the approved public case manifest and image folder after the split and data-release review. Scoring, adjudication, manuscript statistics, and the live private 16-model collector are outside this first slice.
