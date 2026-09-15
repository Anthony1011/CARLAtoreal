"""Check source layout without importing or executing model/pipeline modules."""

import ast
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KNOWN_MISSING = {
    "build_v51_corpus.py", "gen_chroma_maps.py", "gen_depth_moge.py",
    "gen_edges_denoised.py", "gen_light_maps.py", "gen_mapillary_labels_paths.py",
    "make_v50_init.py",
}


def main():
    manifest = json.loads((ROOT / "docs/FILE_MIGRATION.json").read_text(encoding="utf-8"))
    for old, new in manifest["moves"].items():
        assert not (ROOT / old).exists(), f"Unexpected old entry point: {old}"
        assert (ROOT / new).is_file(), f"Missing moved file: {new}"

    source_roots = [ROOT / name for name in ("carla2real", "scripts", "experiments", "pix2pixHD")]
    python_files = [p for folder in source_roots for p in folder.rglob("*.py")
                    if not {"checkpoints", "results", "__pycache__"}.intersection(p.relative_to(folder).parts)]
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in ("config", "vidcodec"), f"Old import in {path}: {node.module}"
                if node.module.startswith("carla2real."):
                    target = ROOT.joinpath(*node.module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), target

    shell_files = [p for name in ("configs", "scripts", "experiments") for p in (ROOT / name).rglob("*.sh")]
    missing = set()
    for path in shell_files:
        source = path.read_text(encoding="utf-8")
        for module in re.findall(r"-m (carla2real\.[\w.]+)", source):
            assert (ROOT / (module.replace(".", "/") + ".py")).is_file(), (path, module)
        for target in re.findall(r"\$BASE/([\w/.-]+\.(?:py|sh))", source):
            if not (ROOT / target).is_file():
                missing.add(target)
        for relative in re.findall(r'\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)([^"\s]+)', source):
            if relative.endswith(".sh"):
                assert (path.parent / relative.lstrip("/")).resolve().is_file(), (path, relative)
    assert missing <= KNOWN_MISSING, f"New unresolved helpers: {missing - KNOWN_MISSING}"

    bash = shutil.which("bash")
    if not bash:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            bash = str(candidate)
    if bash:
        for path in shell_files:
            subprocess.run([bash, "-n", path.as_posix()], check=True, capture_output=True)
        print(f"PASS: Bash syntax ({len(shell_files)} files)")
    else:
        print("SKIP: Bash syntax (Bash not installed)")
    print(f"PASS: {len(manifest['moves'])} moves, {len(python_files)} Python syntax checks, local references")
    print("KNOWN MISSING (not fixed by reorganization): " + ", ".join(sorted(missing)))
    print("NOT RUN: GPU inference, data processing or external perception stack")


if __name__ == "__main__":
    main()
