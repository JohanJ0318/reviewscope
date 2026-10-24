import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")


def test_serving_never_imports_torch():
    code = "import sys, reviewscope.serve; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
