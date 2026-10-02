"""Exercise the complete trace recorder with native local model state."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import transformers
from demo.record import ANSWER_INSTRUCTION, DOCUMENTS, QUESTIONS, record, respond, save

from rclc import SenderState


def test_demo_records_source_positions_messages_answers_and_json(
    model: Any, senders: list[SenderState], tmp_path: Path
) -> None:
    tokenizer = senders[0].tokenizer
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}\n{% endfor %}assistant: "
    )
    documents = [{**document, "text": document["text"][:240]} for document in DOCUMENTS]
    documents[0]["text"] += " Café 🌱. [1,2]"
    for width in (16, 4):
        target = tmp_path / "trace.json"
        with patch("demo.record.respond", wraps=respond) as responses:
            result = record(model, tokenizer, documents, QUESTIONS, span_size=width, output=target)
        full_ids = torch.tensor(
            [token for source in result["sources"] for token in source["token_ids"]]
        )
        full_rows = model.get_input_embeddings()(full_ids)
        baselines = [
            call for call in responses.call_args_list if call.args[3].shape == full_rows.shape
        ]
        assert len(baselines) == len(QUESTIONS)
        for call, question in zip(baselines, QUESTIONS, strict=True):
            assert torch.equal(call.args[3], full_rows) and call.args[2] == question
            assert call.args[4] == ANSWER_INSTRUCTION and call.args[5] == 1024
        assert json.loads(target.read_text()) == json.loads(json.dumps(result))
        assert result["run"]["device"] == "cpu"
        assert result["run"]["dtype"] == "torch.float32"
        assert result["run"]["max_answer_tokens"] == 1024
        assert result["schema"] == 2 and result["span_size"] == width
        assert result["complete"] and result["expected_cases"] == len(QUESTIONS)
        assert len(result["sources"]) == 3
        assert [case["question"] for case in result["cases"]] == QUESTIONS
        for source, document in zip(result["sources"], documents, strict=True):
            assert source["text"] == document["text"]
            encoded = tokenizer(
                document["text"], add_special_tokens=False, return_offsets_mapping=True
            )
            assert source["offsets"] == encoded["offset_mapping"]
            assert source["token_ids"] == encoded["input_ids"]
        for index, case in enumerate(result["cases"]):
            baseline = case["full_context"]
            assert baseline["positions"] == len(full_ids) and baseline["seconds"] > 0
            assert len(baseline["answer"]["token_ids"]) == 1024
            assert baseline["answer"]["text"] == tokenizer.decode(baseline["answer"]["token_ids"])
            assert baseline["answer"]["stop"] == "limit"
            assert case["order"] == (("text", "rclc") if index % 2 else ("rclc", "text"))
            cap, origins = 0, []
            for hop, source in enumerate(result["sources"]):
                cap += (len(source["token_ids"]) + 3) // 4
                text, rclc = (case[method]["hops"][hop] for method in ("text", "rclc"))
                originals = origins + [[hop, i] for i in range(len(source["token_ids"]))]
                indices = rclc["selected_indices"]
                assert indices == sorted(set(indices)) and indices[0] == 0
                assert all(0 <= i < len(originals) for i in indices)
                origins = [originals[i] for i in indices]
                assert origins == rclc["origins"]
                assert rclc["positions"] == len(origins) == cap == rclc["cap"] == text["cap"]
                message = text["message"]
                assert text["positions"] == len(message["token_ids"]) <= cap
                assert message["text"] == tokenizer.decode(message["token_ids"])
                assert message["stop"] in ("limit", "eos")
            for method in ("text", "rclc"):
                run = case[method]
                assert len(run["hops"]) == 3
                assert all(hop["seconds"] > 0 for hop in run["hops"])
                assert run["answer_seconds"] > 0
                assert (
                    sum(hop["seconds"] for hop in run["hops"]) + run["answer_seconds"]
                    <= run["seconds"]
                )
                answer = run["answer"]
                assert len(answer["token_ids"]) == 1024 and answer["stop"] == "limit"
                assert answer["text"] == tokenizer.decode(answer["token_ids"])
        target = tmp_path / "trace.json"
        save(result, target)
        assert json.loads(target.read_text()) == json.loads(json.dumps(result))
    with pytest.raises(ValueError, match="ratio"):
        record(model, tokenizer, documents, QUESTIONS[:1], ratio=0)
    with pytest.raises(ValueError, match="span_size"):
        record(model, tokenizer, documents, QUESTIONS[:1], span_size=0)


def test_colab_records_offline_from_a_source_archive_and_downloads_a_replay(
    model: Any, senders: list[SenderState], tmp_path: Path
) -> None:
    import os
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "tiny-qwen"
    model.config.max_position_embeddings = 2048
    model.save_pretrained(checkpoint)
    tokenizer = senders[0].tokenizer
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}\n{% endfor %}assistant: "
    )
    tokenizer.save_pretrained(checkpoint)
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for folder in ("src", "demo", "examples"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(root))
        for name in ("pyproject.toml", "README.md"):
            archive.write(root / name, name)
    env = {
        **os.environ,
        "RCC_DEMO_TEST_MODE": "1",
        "RCC_DEMO_MODEL": str(checkpoint),
        "RCC_DEMO_SOURCE_ZIP": str(source_zip),
        "RCC_DEMO_OUTPUT": str(tmp_path / "runs"),
    }
    execute = (
        "import json, sys\n"
        "notebook = json.load(open(sys.argv[1]))\n"
        "namespace = {}\n"
        "for cell in notebook['cells']:\n"
        "    if cell['cell_type'] == 'code':\n"
        "        exec(compile(''.join(cell['source']), '<colab-cell>', 'exec'), namespace)\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", execute, str(root / "demo/colab.ipynb")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    (bundle,) = (tmp_path / "runs").glob("*.zip")
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("source.zip") == source_zip.read_bytes()
        assert json.loads(archive.read("config.json"))["test_mode"] is True
        assert archive.read("demo/index.html") == (root / "demo/index.html").read_bytes()
        assert b"transformers" in archive.read("packages.txt")
        for width, name in ((16, "trace.json"), (4, "trace-w4.json")):
            trace = json.loads(archive.read("demo/" + name))
            assert trace["span_size"] == width and len(trace["cases"]) == 2
            assert trace["complete"] and trace["expected_cases"] == 2
            assert trace["run"]["device"] == "cpu"
            assert trace["run"]["model"] == str(checkpoint)
            assert trace["run"]["transformers"] == transformers.__version__
            assert trace["run"]["max_answer_tokens"] == 1024
            for case in trace["cases"]:
                baseline = case["full_context"]
                assert baseline["positions"] == sum(len(s["token_ids"]) for s in trace["sources"])
                assert baseline["seconds"] > 0 and baseline["answer"]["token_ids"]
            assert b"Saved" in archive.read(f"record-w{width}.log")
            assert b"rclc answer" in archive.read(f"record-w{width}.log")
            assert b"full_context answer" in archive.read(f"record-w{width}.log")

    api_env = {key.replace("RCC_DEMO_", "RCC_API_"): value for key, value in env.items()}
    api_env["RCC_API_OUTPUT"] = str(tmp_path / "api-runs")
    process = subprocess.run(
        [sys.executable, "-c", execute, str(root / "examples/colab.ipynb")],
        env=api_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    (bundle,) = (tmp_path / "api-runs").glob("*.zip")
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("source.zip") == source_zip.read_bytes()
        result = json.loads(archive.read("results.json"))
        assert result["complete"] and len(result["cases"]) == 2
        assert result["config"]["model"] == str(checkpoint)
        assert result["transformers"] == transformers.__version__
        assert result["sender_prefill_seconds"] > 0 and result["transfer_seconds"] > 0
        assert result["config"]["device"] == "cpu"
        assert result["config"]["vllm_pins"] == []
        for case in result["cases"]:
            assert case["token_ids"] and case["receiver_input_positions"] > 0
            assert case["receiver_prefill_and_generation_seconds"] > 0
            assert case["contains_expected_text"] == (
                case["expected_text"].lower() in case["answer"].lower()
            )
        assert b"Saved" in archive.read("run.log")
        assert b"from rclc import bind, transfer" in archive.read("run.py")
