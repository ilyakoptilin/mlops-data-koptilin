"""ClearML Dataset part of the practice (steps 9-13).

    python clearml_datasets.py --surname ivanov create    # steps 10-12: 1.0.0, 1.1.0, 2.0.0
    python clearml_datasets.py --surname ivanov verify    # step 13: download by ID, hashes, lineage

No credentials are stored here: the SDK reads them from ~/clearml.conf (clearml-init).
Dataset IDs are saved to clearml_ids.json (local file, ignored by Git) and
the results of `verify` are written into the CLEARML block of REPORT.md.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from clearml import Dataset

ROOT = Path(__file__).resolve().parent
STAGING = ROOT / "clearml_staging"      # what we upload (not submitted)
DOWNLOADS = ROOT / "clearml_downloads"  # what we download (not submitted)
IDS_FILE = ROOT / "clearml_ids.json"
REPORT = ROOT / "REPORT.md"
BEGIN, END = "<!-- CLEARML:BEGIN -->", "<!-- CLEARML:END -->"

VERSIONS = {  # key -> (version, tag)
    "BASE_ID": "1.0.0",
    "WINE_ID": "1.1.0",
    "IRIS_FEATURES_ID": "2.0.0",
}


def names(surname: str):
    project = f"MIIT/MLOps/practice_3/{surname}"
    name = f"sklearn-tabular-{surname}"
    if name in ("demo_dataset",) or "Seminar3" in project:
        sys.exit("forbidden shared names")
    return project, name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def show_server():
    """Step 9: make sure the SDK points to the right server (print host only, never keys)."""
    try:
        from clearml.backend_api import Session
        print("ClearML API server:", Session.get_api_server_host())
    except Exception as exc:  # noqa: BLE001
        print("could not determine server host:", exc)


def stage(mode: str, folder: str):
    """Generate one data state with prepare_data.py into clearml_staging/<folder>/."""
    target = STAGING / folder
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run([sys.executable, str(ROOT / "prepare_data.py"), mode], cwd=target, check=True)
    return target / "data"


def stage_all():
    if STAGING.exists():
        shutil.rmtree(STAGING)
    base = stage("base", "base")                    # iris.csv
    wine = stage("wine", "wine")                    # wine.csv only
    feat = stage("iris_features", "iris_features")  # iris.csv with new columns
    return base, wine, feat


def upload_version(project, name, version, folder, description, tags, parents=None):
    ds = Dataset.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        parent_datasets=parents,
        description=description,
    )
    ds.add_files(path=str(folder))   # only files that belong to this step
    if tags:
        ds.add_tags(tags)
    ds.upload()                      # upload BEFORE finalize
    ds.finalize()                    # closes the version; it does NOT create the next one
    print(f"created {version}: id={ds.id} parents={parents or []}")
    return ds


def cmd_create(args):
    if IDS_FILE.exists() and not args.force:
        sys.exit(f"{IDS_FILE.name} already exists - versions were created. Use --force only if you really want duplicates.")
    project, name = names(args.surname)
    show_server()
    base_dir, wine_dir, feat_dir = stage_all()

    base = upload_version(
        project, name, "1.0.0", base_dir,
        "Original scikit-learn Iris (iris.csv), no extra columns", ["base"],
    )
    wine = upload_version(
        project, name, "1.1.0", wine_dir,
        "Child of 1.0.0: adds wine.csv (original scikit-learn Wine)", ["wine"],
        parents=[base.id],
    )
    feat = upload_version(
        project, name, "2.0.0", feat_dir,
        "Child of 1.0.0 (NOT of 1.1.0): iris.csv with sepal_area and petal_area", ["iris-features"],
        parents=[base.id],
    )
    IDS_FILE.write_text(json.dumps(
        {"BASE_ID": base.id, "WINE_ID": wine.id, "IRIS_FEATURES_ID": feat.id}, indent=2))
    print(json.dumps(json.loads(IDS_FILE.read_text()), indent=2))


def describe(key, dataset_id, out_dir):
    ds = Dataset.get(dataset_id=dataset_id, only_completed=True)   # strictly by ID
    if out_dir.exists():
        shutil.rmtree(out_dir)
    local = Path(ds.get_mutable_local_copy(target_folder=str(out_dir), overwrite=False))
    graph = ds.get_dependency_graph()   # {id: [parent ids]} for this version and all ancestors
    files = sorted(p for p in local.rglob("*") if p.is_file())
    tables = {}
    for p in files:
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            tables[p.name] = (df.shape, list(df.columns))
    return {
        "key": key, "id": dataset_id, "version": ds.version, "status": getattr(ds, "status", None) or "completed",
        "tags": ds.tags, "parents": graph.get(dataset_id, []), "graph": graph,
        "files": sorted(ds.list_files()),
        "added": sorted(ds.list_added_files()),
        "modified": sorted(ds.list_modified_files()),
        "removed": sorted(ds.list_removed_files()),
        "hashes": {p.relative_to(local).as_posix(): sha256(p) for p in files},
        "tables": tables, "dir": out_dir.name,
    }


def cmd_verify(args):
    if not IDS_FILE.exists():
        sys.exit("run `create` first")
    ids = json.loads(IDS_FILE.read_text())
    show_server()
    DOWNLOADS.mkdir(exist_ok=True)
    info = {k: describe(k, ids[k], DOWNLOADS / v) for k, v in VERSIONS.items()}

    base, wine, feat = info["BASE_ID"], info["WINE_ID"], info["IRIS_FEATURES_ID"]
    checks = {
        "1.1.0 parent == BASE_ID": wine["parents"] == [base["id"]],
        "2.0.0 parent == BASE_ID": feat["parents"] == [base["id"]],
        "WINE_ID is NOT an ancestor of 2.0.0": wine["id"] not in feat["graph"],
        "1.0.0 contains only iris.csv": base["files"] == ["iris.csv"],
        "1.1.0 contains iris.csv + wine.csv": wine["files"] == ["iris.csv", "wine.csv"],
        "1.1.0 added == [wine.csv]": wine["added"] == ["wine.csv"],
        "2.0.0 contains only iris.csv": feat["files"] == ["iris.csv"],
        "2.0.0 modified == [iris.csv]": feat["modified"] == ["iris.csv"],
        "iris.csv in 1.0.0 and 1.1.0 identical": base["hashes"]["iris.csv"] == wine["hashes"]["iris.csv"],
        "iris.csv in 2.0.0 differs from base": base["hashes"]["iris.csv"] != feat["hashes"]["iris.csv"],
    }

    lines = []
    for item in (base, wine, feat):
        lines += [
            f"#### {item['key']} - version {item['version']}",
            "",
            f"- Dataset ID: `{item['id']}` (status: {item['status']}, tags: {item['tags']})",
            f"- parent IDs: {item['parents'] or 'none'}",
            f"- files: {item['files']}",
            f"- added: {item['added']}; modified: {item['modified']}; removed: {item['removed']}",
            f"- downloaded to: `clearml_downloads/{item['dir']}/`",
            "",
            "| file | SHA-256 | shape | columns |",
            "|---|---|---|---|",
        ]
        for f, h in item["hashes"].items():
            shape, cols = item["tables"].get(f, ("-", "-"))
            lines.append(f"| {f} | `{h}` | {shape} | {cols} |")
        lines.append("")
    lines += ["#### Dependency graph (`get_dependency_graph()` of 2.0.0 and 1.1.0)", "", "```json",
              json.dumps({"2.0.0": feat["graph"], "1.1.0": wine["graph"]}, indent=2), "```", "",
              "#### Automatic checks", ""]
    lines += [f"- [{'x' if ok else ' '}] {text}" for text, ok in checks.items()]
    block = "\n".join(lines)

    print(block)
    if REPORT.exists() and BEGIN in REPORT.read_text():
        text = REPORT.read_text()
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        REPORT.write_text(f"{head}{BEGIN}\n{block}\n{END}{tail}")
        print("\nREPORT.md updated")
    if not all(checks.values()):
        sys.exit("SOME CHECKS FAILED - see the list above")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surname", required=True, help="latin surname, used in project/dataset names")
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create")
    create.add_argument("--force", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    {"create": cmd_create, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
