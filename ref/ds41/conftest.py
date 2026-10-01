"""pytest bootstrap: make `import ref.ds41` work from any cwd (the repo root holds the `ref` namespace package)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
