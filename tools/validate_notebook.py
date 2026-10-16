"""Check a Colab notebook is well-formed without executing any cell."""
import json
from pathlib import Path

notebook_path = Path("colab_setup.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4, "Unsupported notebook format"
assert isinstance(nb.get("cells"), list), "Notebook has no cells array"
for i, cell in enumerate(nb["cells"]):
    assert cell.get("cell_type") in ("markdown", "code", "raw"), f"Bad cell {i}"
    assert isinstance(cell.get("source"), list), f"Cell {i} source must be a list"
    if cell["cell_type"] == "code":
        assert cell.get("execution_count") is None, "Do not commit execution counts"
        assert cell.get("outputs") == [], "Do not commit output or personal data"
print("Colab notebook structure OK (nothing executed).")
