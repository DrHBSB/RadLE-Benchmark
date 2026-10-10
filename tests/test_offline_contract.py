import base64
import runpy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from radle.inference import PROMPT, PROMPT_SHA256, request_kwargs, run_case, models
from radle.parsing import parse_response


def test_prompt_and_ordered_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert (root / "prompts" / "diagnostic.txt").read_text(encoding="utf-8") == PROMPT
    assert (root / "configs" / "models.json").read_text(encoding="utf-8") == (
        root / "src" / "radle" / "resources" / "models.json"
    ).read_text(encoding="utf-8")
    a, b = tmp_path / "a.png", tmp_path / "b.jpg"
    Image.new("RGB", (3, 3), "red").save(a)
    Image.new("RGB", (3, 3), "blue").save(b)
    model = models()["muse_spark_1_3_openrouter"]
    kwargs = request_kwargs(model, [b, a])
    items = kwargs["messages"][0]["content"]
    assert PROMPT_SHA256 == "d409f2821f56deb57892f7bffaffd73234c9e5374e8cdfbcc864cc9cfbdbf3ba"
    assert [base64.b64decode(item["image_url"]["url"].split(",", 1)[1]) for item in items[1:]] == [b.read_bytes(), a.read_bytes()]
    assert "temperature" not in kwargs
    assert kwargs["extra_body"]["provider"]["only"] == ["meta"]


def test_response_states():
    cases = [
        ('{"diagnosis":"Tuberculosis","likert_score":3}', ("Tuberculosis", 3, "ok")),
        ('{"diagnosis":"I don\'t know","likert_score":null}', ("I don't know", None, "abstained")),
        ('{"diagnosis":"Tuberculosis","likert_score":7}', ("Tuberculosis", None, "invalid_likert")),
        ('reasoning {"diagnosis":"Tuberculosis","likert_score":3}', ("", None, "malformed_response")),
        ('```json\n{"diagnosis":"Tuberculosis","likert_score":3}\n```', ("Tuberculosis", 3, "ok_recovered")),
        ('no JSON', ("", None, "malformed_response")),
    ]
    for raw, expected in cases:
        assert parse_response(raw) == expected


def test_mock_request(tmp_path):
    image = tmp_path / "one.png"
    Image.new("RGB", (2, 2)).save(image)
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"diagnosis":"Normal","likert_score":1}'))])
    row = run_case([image], "muse_spark_1_3_openrouter", client=Client())
    assert (row["diagnosis"], row["likert_score"], row["status"]) == ("Normal", 1, "ok")


def test_open_model_reference_requests(tmp_path):
    image = tmp_path / "one.png"
    Image.new("RGB", (2, 2)).save(image)
    root = Path(__file__).resolve().parents[1] / "advanced" / "open_models"
    ollama = runpy.run_path(str(root / "ollama_one_case.py"))
    medgemma = runpy.run_path(str(root / "medgemma_one_case.py"))
    octo = ollama["request_kwargs"]("octomed_7b", [image])
    assert octo["model"] == "hf.co/mradermacher/OctoMed-7B-GGUF:Q8_0"
    assert octo["temperature"] == 0.01 and octo["max_tokens"] == 8192
    assert "provider" not in octo
    med = medgemma["request_kwargs"]([image])
    assert med["model"] == "google/medgemma-1.5-4b-it"
    assert med["max_tokens"] == 2048
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"diagnosis":"Normal","likert_score":1}'))])
    assert ollama["run_case"]([image], "octomed_7b", client=Client())["status"] == "ok"
    assert medgemma["run_case"]([image], client=Client())["status"] == "ok"


def test_minimax_official_provider_request(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.jpg"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "blue").save(second)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"diagnosis":"Normal","likert_score":1}')
        )])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))
    row = run_case([second, first], "minimax_m3", client=client)
    kwargs = calls[0]
    assert kwargs["model"] == "minimax/minimax-m3"
    assert kwargs["extra_body"]["provider"] == {
        "only": ["minimax/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert kwargs["temperature"] == 0.01 and kwargs["max_tokens"] == 16384
    assert "reasoning" not in kwargs["extra_body"]
    assert kwargs["extra_body"]["response_format"] == {"type": "json_object"}
    content = kwargs["messages"][0]["content"]
    assert content[0]["text"] == PROMPT
    assert [
        base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
        for item in content[1:]
    ] == [second.read_bytes(), first.read_bytes()]
    assert row["status"] == "ok" and row["model_id"] == "minimax/minimax-m3"
