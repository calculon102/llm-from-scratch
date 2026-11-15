"""End-to-end test for the one-command MVP demo (T-606)."""

from __future__ import annotations

import json

from scripts import mvp_demo


def test_mvp_demo_reproduces_flow(tmp_path, capsys):
    rc = mvp_demo.main(
        [
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ]
    )
    output = capsys.readouterr().out
    assert rc == 0, output
    assert "MVP demo complete" in output
    assert "device=cpu" in output
    assert "XOR phase (M1)" in output
    assert "Tokenizer phase (M2)" in output
    assert "MiniGPT pipeline (core M4)" in output

    reports = list(tmp_path.glob("**/pipeline.json"))
    assert len(reports) == 1, reports
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "finished"
    assert {"preprocess", "tokenizer", "gpt_setup", "train", "eval"} <= set(
        report["stages"]
    )

    run_root = reports[0].parents[2]
    assert list(run_root.glob("**/checkpoints/final.pt"))
    assert list(run_root.glob("**/gradcheck_report.txt"))
    assert list(run_root.glob("**/tokenizers/bundle.json"))
    assert list(run_root.glob("**/eval/report.json"))
    assert list(run_root.glob("**/gpt/model.pt"))
    assert list(run_root.glob("**/eval/sample.txt"))
