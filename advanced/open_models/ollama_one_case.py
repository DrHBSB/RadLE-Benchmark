"""Reference-only, one-case multimodal call to an existing Ollama server.

This is an adaptation of the study's successful Ollama routes, not its
200-case collector or a claim of numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from radle.inference import PROMPT, PROMPT_SHA256, write_csv
from radle.inputs import content_array
from radle.parsing import parse_response

MODELS = {
    "llava_med_mistral_7b": ("z-uo/llava-med-v1.5-mistral-7b_q8_0", 2048),
    "octomed_7b": ("hf.co/mradermacher/OctoMed-7B-GGUF:Q8_0", 8192),
    "internvl3_5_8b": ("hf.co/mradermacher/InternVL3_5-8B-GGUF:Q8_0", 8192),
}


def request_kwargs(model_name, paths, model_id=None):
    if model_name not in MODELS:
        raise ValueError(f"Unsupported reference model: {model_name}")
    default_id, max_tokens = MODELS[model_name]
    return {
        "model": model_id or default_id,
        "messages": [{"role": "user", "content": content_array(paths, PROMPT)}],
        "temperature": 0.01,
        "max_tokens": max_tokens,
    }


def run_case(paths, model_name, *, case_id="uploaded_case", model_id=None, client=None):
    kwargs = request_kwargs(model_name, paths, model_id)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
    started = time.perf_counter()
    try:
        raw = client.chat.completions.create(**kwargs).choices[0].message.content
        diagnosis, likert, status = parse_response(raw)
    except Exception:
        diagnosis, likert, status = "", None, "api_error"
    return {
        "case_id": case_id,
        "model_id": kwargs["model"],
        "diagnosis": diagnosis,
        "likert_score": likert,
        "status": status,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "prompt_sha256": PROMPT_SHA256,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--model-id", help="Override the local Ollama model tag")
    parser.add_argument("--case-id", default="uploaded_case")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    row = run_case(args.images, args.model, case_id=args.case_id, model_id=args.model_id)
    write_csv(row, args.output)
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["status"] in {"ok", "ok_recovered", "abstained"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
