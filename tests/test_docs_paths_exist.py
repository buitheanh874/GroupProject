from __future__ import annotations

from pathlib import Path


def test_docs_files_exist_and_non_empty():
    docs = [
        Path("docs/MDP_COMPLIANCE.md"),
    ]
    for doc in docs:
        assert doc.exists(), f"{doc} missing"
        assert doc.is_file(), f"{doc} not a file"
        assert doc.read_text(encoding="utf-8").strip(), f"{doc} is empty"
