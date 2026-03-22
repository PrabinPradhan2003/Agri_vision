"""Convenience launcher.

This repo's Flask app lives in the `Flask Deployed App/` folder.
Running `python app.py` from the repo root will start that app.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    target = repo_root / "Flask Deployed App" / "app.py"
    if not target.is_file():
        raise FileNotFoundError(f"Could not find Flask app entrypoint at: {target}")

    # Make local imports inside the Flask app folder work (e.g. `import CNN`).
    flask_dir = str(target.parent)
    if flask_dir not in sys.path:
        sys.path.insert(0, flask_dir)

    # Some setups rely on CWD for ancillary files; keep it consistent.
    os.chdir(flask_dir)

    # Execute the real app as __main__ so its Flask root_path points to the
    # correct folder (templates/static live alongside that file).
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
