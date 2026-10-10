"""Validate and encode ordered images without altering their bytes."""

from __future__ import annotations

import base64
from pathlib import Path
from PIL import Image


def ordered_images(paths):
    if not paths:
        raise ValueError("Provide at least one image")
    result = []
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        raw = path.read_bytes()
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type: {path.name}")
        with Image.open(path) as img:
            if img.format not in {"PNG", "JPEG"}:
                raise ValueError(f"Unsupported image format: {path.name}")
            img.verify()
        result.append((path, mime, raw))
    return result


def content_array(paths, prompt):
    content = [{"type": "text", "text": prompt}]
    for _, mime, raw in ordered_images(paths):
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        }})
    return content
