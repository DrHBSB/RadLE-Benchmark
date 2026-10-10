"""One-case OpenRouter inference using the selected v2 chat request contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from importlib.resources import files
from pathlib import Path

from .inputs import content_array
from .parsing import parse_response

RESOURCES = files("radle").joinpath("resources")
PROMPT = RESOURCES.joinpath("diagnostic.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
COLUMNS = ["case_id", "model_id", "diagnosis", "likert_score", "status", "latency_seconds", "prompt_sha256"]


def models():
    return json.loads(RESOURCES.joinpath("models.json").read_text(encoding="utf-8"))


def request_kwargs(model, paths):
    kwargs = {
        "model": model["model_id"],
        "messages": [{"role": "user", "content": content_array(paths, PROMPT)}],
        "max_tokens": model["max_tokens"],
        "extra_body": {
            "response_format": {"type": "json_object"},
            "provider": {"only": [model["provider_only"]], "allow_fallbacks": False, "require_parameters": True},
        },
    }
    if model.get("reasoning_effort") is not None:
        kwargs["extra_body"]["reasoning"] = {
            "effort": model["reasoning_effort"], "exclude": False
        }
    if model.get("temperature") is not None:
        kwargs["temperature"] = model["temperature"]
    return kwargs


def run_case(paths, model_name, case_id="uploaded_case", client=None):
    registry = models()
    if model_name not in registry:
        raise ValueError(f"Unsupported model: {model_name}")
    kwargs = request_kwargs(registry[model_name], paths)
    if client is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        diagnosis, score, status = parse_response(raw)
    except Exception:
        diagnosis, score, status = "", None, "api_error"
    return {
        "case_id": case_id,
        "model_id": registry[model_name]["model_id"],
        "diagnosis": diagnosis,
        "likert_score": score,
        "status": status,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "prompt_sha256": PROMPT_SHA256,
    }


def write_csv(row, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="Image paths in presentation order")
    parser.add_argument("--model", required=True, choices=sorted(models()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", default="uploaded_case")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is not set")
    row = run_case(args.images, args.model, args.case_id)
    write_csv(row, args.output)
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["status"] in {"ok", "ok_recovered", "abstained"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

