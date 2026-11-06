"""Refresh the public release manifest after an intentional code/document change."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "checks",
}


def main():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[0] in ("data", "resources", "runs", "supplementary"):
            raise ValueError(f"Private or separate delivery directory: {relative}")
        if path.suffix in (".npz", ".npy", ".pkl", ".pt", ".pth", ".pdf", ".zip"):
            raise ValueError(f"Unexpected release file: {relative}")
        if path.is_symlink():
            raise ValueError(f"Release files must be regular files: {relative}")
        files[str(relative)] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    value = {
        "model_version": "clinical_memory_dynamic_20261104",
        "package_version": "0.1.0",
        "public_url": "https://github.com/LIZESHENG13/EviDrug",
        "files": files,
        "files_count": len(files),
        "patient_level_data_included": False,
        "supplementary_pdf_included": False,
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(value, indent=2) + "\n")
    print(f"Manifested {len(files)} public release files.")


if __name__ == "__main__":
    main()
