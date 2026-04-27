#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace(path: str, pattern: str, replacement: str, *, required: bool = True) -> None:
    original = read(path)
    updated, count = re.subn(pattern, replacement, original)
    if count == 0 and required:
        raise RuntimeError(f"no version match in {path}: {pattern}")
    write(path, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set all public package versions.")
    parser.add_argument("version", help="Semver package version, for example 0.1.0")
    args = parser.parse_args()

    version = args.version
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise SystemExit(f"invalid semver: {version}")
    asset_version = f"v{version}"

    replace(
        "embeddings_cpp/__init__.py",
        r'__version__ = "[^"]+"',
        f'__version__ = "{version}"',
    )

    for path in ["packages/web/package.json", "packages/web/package-lock.json"]:
        package_path = ROOT / path
        data = json.loads(package_path.read_text(encoding="utf-8"))
        data["version"] = version
        if path.endswith("package-lock.json"):
            data["packages"][""]["version"] = version
        package_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    replace(
        "packages/web/src/index.ts",
        r'const RUNTIME_ASSET_VERSION = "[^"]+";',
        f'const RUNTIME_ASSET_VERSION = "{asset_version}";',
    )
    replace(
        "packages/web/dist/index.js",
        r'const RUNTIME_ASSET_VERSION = "[^"]+";',
        f'const RUNTIME_ASSET_VERSION = "{asset_version}";',
    )

    query_paths = [
        "packages/web/examples/basic-browser.html",
        "packages/web/examples/demo.html",
        "packages/web/examples/mobile-diagnostics.html",
        "scripts/browser_runtime_regression.mjs",
        "scripts/wasm_encode_page.html",
        "scripts/wasm_persistent_encode_page.html",
    ]
    for path in query_paths:
        replace(path, r"\?v=[A-Za-z0-9_.+-]+", f"?v={asset_version}", required=False)
        replace(path, r'params\.get\("v"\) \|\| "[^"]+"', f'params.get("v") || "{asset_version}"', required=False)

    docs_paths = [
        "packages/web/README.md",
        "docs/SNOWFLAKE_NPM_PACKAGE.md",
        "docs/PRODUCTIZATION.md",
        ".github/workflows/upload-web-assets-to-hf.yml",
    ]
    for path in docs_paths:
        replace(path, r"browser/[A-Za-z0-9_.+-]+/", f"browser/{asset_version}/", required=False)
        replace(path, r"`v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?`", f"`{asset_version}`", required=False)
        replace(path, r'default: "v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?"', f'default: "{asset_version}"', required=False)

    print(f"set package version {version} and web asset version {asset_version}")


if __name__ == "__main__":
    main()
