# Methods and implementation status

This repository is an early public development starter for the RadLE 2.0 inference interface. It does not establish the manuscript's final 200-case methods, scores, statistics, public/private split, or clinical adjudication.

## Current inference contract

The one-case Python runner in `src/radle/inference.py` places the text in `prompts/diagnostic.txt` first in a user-role content array, followed by the case images in the supplied order as unmodified PNG/JPEG bytes encoded as data URLs. It requests a JSON diagnosis and confidence level 0–4, or an "I don't know" response with null confidence. The parser distinguishes a valid answer, abstention, invalid confidence, malformed response, and API error.

`configs/models.json` is a small set of example OpenRouter configurations, not the final manuscript roster. It records requested model IDs, provider pins, reasoning settings where applicable, and output limits. Provider routes, model availability, and privacy properties must be checked at run time. GPT-6 Astra used a separate Responses API route in the study and is not implemented in this starter.

The Python command and Colab currently run **one case**. They do not assess correctness or compute RadLE-A or RadLE-C. No clinical images, answer key, or study results are stored in this repository.

## Development boundary

Upasana's initial development can use the original 50 cases from a separately shared private folder. The runner should accept a case manifest and image root as inputs rather than embedding an absolute Drive path. The final approved public manifest can later replace the development manifest without changing prompt, image ordering, response parsing, or output fields. Public release of any case package needs a separate review of the final split and redistribution terms.

Keep saved predictions distinct from adjudicated correctness and scores. A successful API response is not a validated diagnosis.
