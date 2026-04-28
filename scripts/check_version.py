#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def match_one(path: str, pattern: str) -> str:
    text = read(path)
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing version pattern in {path}: {pattern}")
    return match.group(1)


def main() -> None:
    errors: list[str] = []

    py_version = match_one("embeddings_cpp/__init__.py", r'__version__ = "([^"]+)"')
    asset_version = f"v{py_version}"

    package_json = json.loads(read("packages/web/package.json"))
    package_lock = json.loads(read("packages/web/package-lock.json"))
    require(package_json["version"] == py_version, "packages/web/package.json version mismatch", errors)
    require(package_lock["version"] == py_version, "packages/web/package-lock.json version mismatch", errors)
    require(
        package_lock["packages"][""]["version"] == py_version,
        "packages/web/package-lock.json root package version mismatch",
        errors,
    )

    src_asset = match_one("packages/web/src/index.ts", r'const RUNTIME_ASSET_VERSION = "([^"]+)";')
    dist_asset = match_one("packages/web/dist/index.js", r'const RUNTIME_ASSET_VERSION = "([^"]+)";')
    require(src_asset == asset_version, f"src web asset version mismatch: {src_asset}", errors)
    require(dist_asset == asset_version, f"dist web asset version mismatch: {dist_asset}", errors)

    paths_with_asset = [
        "packages/web/README.md",
        "docs/SNOWFLAKE_NPM_PACKAGE.md",
        "docs/PRODUCTIZATION.md",
        "packages/web/examples/basic-browser.html",
        "packages/web/examples/demo.html",
        "packages/web/examples/mobile-diagnostics.html",
        "scripts/browser_runtime_regression.mjs",
        "scripts/wasm_encode_page.html",
        "scripts/wasm_persistent_encode_page.html",
    ]
    dynamic_asset_paths = [
        ".github/workflows/upload-web-assets-to-hf.yml",
    ]
    for path in paths_with_asset:
        text = read(path)
        require(
            asset_version in text,
            f"{path} does not mention expected asset version {asset_version}",
            errors,
        )
        stale_webpkg = re.findall(r"webpkg\d+", text)
        require(not stale_webpkg, f"{path} contains stale webpkg marker(s): {stale_webpkg}", errors)

    for path in dynamic_asset_paths:
        text = read(path)
        require(
            "scripts/resolve_version.py --asset" in text,
            f"{path} does not derive the browser asset version dynamically",
            errors,
        )
        stale_webpkg = re.findall(r"webpkg\d+", text)
        require(not stale_webpkg, f"{path} contains stale webpkg marker(s): {stale_webpkg}", errors)

    if errors:
        for error in errors:
            print(f"version check failed: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(f"version check passed: package={py_version} web_asset={asset_version}")


if __name__ == "__main__":
    main()
