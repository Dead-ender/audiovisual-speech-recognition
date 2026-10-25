"""Fast structural checks for the notebook artifact; no model execution."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "Audiovisual Speech Recognition through Multimodal Fusion of Lip Movements and Audio Spectrograms.ipynb"


def test_notebook_is_valid_and_has_no_saved_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_notebook_python_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]))


def test_training_demo_is_opt_in():
    source = "\n".join(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )
    assert "RUN_DEMO = False" in source
    assert "if RUN_DEMO:" in source
