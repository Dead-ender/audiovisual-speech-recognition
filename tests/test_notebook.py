"""Structural checks for the retained synthetic notebook."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / (
    "Audiovisual Speech Recognition through Multimodal Fusion of Lip Movements and "
    "Audio Spectrograms.ipynb"
)


def test_notebook_has_no_saved_execution_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_notebook_code_cells_parse_and_demo_is_opt_in():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    assert "RUN_DEMO = False" in "\n".join(code)
