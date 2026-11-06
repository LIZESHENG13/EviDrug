"""Download and verify the three public mapping/interaction resources."""

import json
import urllib.request
from importlib.resources import files
from pathlib import Path

from ..utils import sha, write


def download_resources(directory="resources"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = json.loads(
        files("evidrug.data").joinpath("mappings/resource_checksums.json").read_text()
    )
    for name, reference in expected.items():
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            with urllib.request.urlopen(reference["source"], timeout=60) as response:
                with temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if sha(temporary) != reference["sha256"]:
                temporary.unlink()
                raise ValueError(f"Downloaded resource checksum differs: {name}")
            temporary.replace(path)
        if sha(path) != reference["sha256"]:
            raise ValueError(f"Resource checksum differs: {name}")
        print(f"Verified {name}", flush=True)
    write(directory / "resource_manifest.json", expected)


def prepare_dataset(dataset, input_path, output, resources="resources"):
    input_path, output, resources = (Path(p).resolve() for p in (input_path, output, resources))
    output.mkdir(parents=True, exist_ok=True)
    if dataset == "mimic_iv_full":
        from .mimic_iv import prepare

        prepare(input_path, output / "patients.pkl", resources)
    elif dataset == "nwicu":
        from .nwicu import prepare
        from .nwicu_inspection import inspect

        audit = output / "inspection"
        inspect(input_path, audit, resources)
        prepare(input_path, output, resources, audit / "audit.json")
    elif dataset == "sicdb":
        from .sicdb import prepare

        prepare(input_path, output)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return output / "patients.pkl"
