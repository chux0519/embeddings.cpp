#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / ".vendor" / "ggml-upstream.json"
GGML_PATH = ROOT / "ggml"


def load_meta() -> dict[str, Any]:
    with META_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta: dict[str, Any]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")


def git(args: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def status(_: argparse.Namespace) -> None:
    meta = load_meta()
    current = meta["current"]
    upstream = meta["upstream"]
    print(f"vendored path: {meta['vendored_path']}")
    print(f"upstream repo: {upstream['repository']}")
    print(f"upstream subdir: {upstream['subdirectory']}")
    print(f"current ref: {current['ref']}")
    print(f"current commit: {current['commit']}")
    print(f"imported in: {current.get('imported_in', 'unknown')}")
    print()
    print("local ggml changes:")
    diff = git(["status", "--short", "--", "ggml"])
    print(diff or "none")


def inspect_source(args: argparse.Namespace) -> None:
    source_root = Path(args.source).expanduser().resolve()
    source_ggml = source_root / "ggml"
    if not source_ggml.is_dir():
        raise SystemExit(f"missing source ggml directory: {source_ggml}")

    print(f"source: {source_root}")
    try:
        print(f"commit: {git(['rev-parse', 'HEAD'], cwd=source_root)}")
        print(f"describe: {git(['describe', '--tags', '--always', '--dirty'], cwd=source_root)}")
        status_text = git(["status", "--short"], cwd=source_root)
        print("status:")
        print(status_text or "clean")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"failed to inspect git checkout: {exc}") from exc

    cmake = source_ggml / "CMakeLists.txt"
    print()
    print("ggml/CMakeLists.txt header:")
    with cmake.open("r", encoding="utf-8") as f:
        for _, line in zip(range(12), f):
            print(line.rstrip())


def set_current(args: argparse.Namespace) -> None:
    meta = load_meta()
    meta["current"]["ref"] = args.ref or "unknown"
    meta["current"]["commit"] = args.commit or "unknown"
    meta["current"]["date"] = args.date or date.today().isoformat()
    try:
        meta["current"]["imported_in"] = git(["rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        pass
    meta["current"]["notes"] = args.notes or "Exact upstream ref recorded."
    save_meta(meta)
    print(f"recorded ggml upstream {meta['current']['ref']} ({meta['current']['commit']})")


def import_from_checkout(args: argparse.Namespace) -> None:
    source_root = Path(args.source).expanduser().resolve()
    source_ggml = source_root / "ggml"
    if not source_ggml.is_dir():
        raise SystemExit(f"missing source ggml directory: {source_ggml}")

    if args.commit == "auto":
        args.commit = git(["rev-parse", "HEAD"], cwd=source_root)
    if args.ref == "auto":
        args.ref = git(["describe", "--tags", "--always"], cwd=source_root)

    dirty = git(["status", "--short", "--", "ggml"])
    if dirty and not args.force:
        raise SystemExit(
            "refusing to overwrite local ggml changes; commit/stash them or pass --force"
        )

    if GGML_PATH.exists():
        shutil.rmtree(GGML_PATH)
    ignore = shutil.ignore_patterns(".git", "build", "__pycache__")
    shutil.copytree(source_ggml, GGML_PATH, ignore=ignore)

    set_current(args)
    print(f"imported {source_ggml} -> {GGML_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track the vendored ggml upstream ref.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="print recorded upstream state")
    status_parser.set_defaults(func=status)

    inspect_parser = subparsers.add_parser("inspect-source", help="inspect a llama.cpp checkout")
    inspect_parser.add_argument("--source", required=True, help="path to a llama.cpp checkout")
    inspect_parser.set_defaults(func=inspect_source)

    set_parser = subparsers.add_parser("set", help="record the current upstream ref")
    set_parser.add_argument("--ref", required=True, help="upstream tag or branch")
    set_parser.add_argument("--commit", required=True, help="resolved upstream commit")
    set_parser.add_argument("--date", help="import date, YYYY-MM-DD")
    set_parser.add_argument("--notes", help="free-form provenance notes")
    set_parser.set_defaults(func=set_current)

    import_parser = subparsers.add_parser("import", help="copy ggml from a llama.cpp checkout")
    import_parser.add_argument("--source", required=True, help="path to a llama.cpp checkout")
    import_parser.add_argument("--ref", default="auto", help="upstream tag or branch, or auto")
    import_parser.add_argument("--commit", default="auto", help="resolved upstream commit, or auto")
    import_parser.add_argument("--date", help="import date, YYYY-MM-DD")
    import_parser.add_argument("--notes", help="free-form provenance notes")
    import_parser.add_argument("--force", action="store_true", help="overwrite dirty local ggml")
    import_parser.set_defaults(func=import_from_checkout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
