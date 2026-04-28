#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")


def normalize_tag(tag: str) -> str | None:
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    if tag.startswith("web-v"):
        version = tag.removeprefix("web-v")
    elif tag.startswith("v"):
        version = tag.removeprefix("v")
    else:
        return None
    return version if VERSION_RE.fullmatch(version) else None


def exact_git_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for tag in result.stdout.splitlines():
        version = normalize_tag(tag.strip())
        if version is not None:
            return version
    return None


def source_version() -> str | None:
    init_py = ROOT / "embeddings_cpp/__init__.py"
    match = re.search(r'__version__ = "([^"]+)"', init_py.read_text(encoding="utf-8"))
    if match and VERSION_RE.fullmatch(match.group(1)):
        return match.group(1)
    return None


def resolve_version() -> str:
    for value in (
        os.environ.get("EMBEDDINGS_CPP_VERSION"),
        normalize_tag(os.environ.get("GITHUB_REF_NAME", "")),
        normalize_tag(os.environ.get("GITHUB_REF", "")),
        exact_git_tag(),
        source_version(),
    ):
        if value:
            return value
    raise SystemExit("could not resolve package version from env, git tag, or source")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the release version from env, git tag, or source.")
    parser.add_argument("--asset", action="store_true", help="Print the browser asset version, e.g. v0.1.2.")
    args = parser.parse_args()

    version = resolve_version()
    print(f"v{version}" if args.asset else version)


if __name__ == "__main__":
    main()
