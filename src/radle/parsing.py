"""Parse the requested two-field response without judging correctness."""

from __future__ import annotations

import json
import re


def parse_response(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "", None, "malformed_response"
    text = raw_text.strip()
    recovered = False
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text, recovered = fenced.group(1), True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "", None, "malformed_response"
    if not isinstance(data, dict):
        return "", None, "malformed_response"
    diagnosis = data.get("diagnosis")
    score = data.get("likert_score")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        return "", None, "malformed_response"
    diagnosis = diagnosis.strip()
    if diagnosis == "I don't know":
        if score is not None:
            return diagnosis, None, "invalid_abstention"
        return diagnosis, None, "abstained"
    if isinstance(score, bool) or not isinstance(score, int) or score not in range(5):
        return diagnosis, None, "invalid_likert"
    return diagnosis, score, "ok_recovered" if recovered else "ok"
