"""Reference-only MedGemma call to an already running OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import time

from radle.inference import PROMPT, PROMPT_SHA256, write_csv
from radle.inputs import content_array
from radle.parsing import parse_response

MODEL_ID = "google/medgemma-1.5-4b-it"


def request_kwargs(paths, model_id=MODEL_ID):
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": content_array(paths, PROMPT)}],
        "temperature": 0.01,
        "max_tokens": 2048,
    }


def run_case(paths, *, client, model_id=MODEL_ID, case_id="uploaded_case"):
    kwargs = request_kwargs(paths, model_id)
    started = time.perf_counter()
    try:
        raw = client.chat.completions.create(**kwargs).choices[0].message.content
        diagnosis, likert, status = parse_response(raw)
    except Exception:
        diagnosis, likert, status = "", None, "api_error"
    return {
        "case_id": case_id,
        "model_id": model_id,
        "diagnosis": diagnosis,
        "likert_score": likert,
        "status": status,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "prompt_sha256": PROMPT_SHA256,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--base-url", required=True, help="URL of your running server's /v1 endpoint")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--api-key", default="local", help="Server credential, if required")
    parser.add_argument("--case-id", default="uploaded_case")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    row = run_case(args.images, client=client, model_id=args.model_id, case_id=args.case_id)
    write_csv(row, args.output)
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["status"] in {"ok", "ok_recovered", "abstained"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
