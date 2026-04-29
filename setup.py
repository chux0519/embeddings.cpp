import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def _is_riscv64_target() -> bool:
    hints = [
        os.environ.get("CIBW_ARCHS", ""),
        os.environ.get("CIBW_BUILD", ""),
        os.environ.get("AUDITWHEEL_ARCH", ""),
        platform.machine(),
    ]
    return any("riscv64" in hint.lower() for hint in hints)


def _platform_cmake_args() -> list[str]:
    if not _is_riscv64_target():
        return []
    # manylinux riscv64 currently ships binutils that reject the newer "zvfh"
    # ISA extension spelling. Keep RVV enabled, but avoid optional fp16 and
    # cache-hint extensions that are not safe for a portable wheel.
    return [
        "-DGGML_RV_ZFH=OFF",
        "-DGGML_RV_ZVFH=OFF",
        "-DGGML_RV_ZICBOP=OFF",
        "-DGGML_RV_ZIHINTPAUSE=OFF",
    ]


def _cmake_build_jobs() -> str | None:
    jobs = os.environ.get("EMBEDDINGS_CPP_BUILD_JOBS")
    if jobs is not None:
        jobs = jobs.strip()
        return jobs or None
    if _is_riscv64_target():
        # QEMU user-mode builds can get slower or appear stuck when Ninja fans
        # out across all hosted-runner cores. Keep RVV enabled, but use a small
        # parallelism cap for the emulated wheel build.
        return "2"
    return None


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        native = os.environ.get("EMBEDDINGS_CPP_NATIVE", "0").lower() in {"1", "on", "true", "yes"}

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DEMBEDDINGS_CPP_ENABLE_PYBIND=ON",
            f"-DEMBEDDINGS_CPP_BUILD_WASM_TOOLS=OFF",
            f"-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CPU_REPACK=ON",
            "-DGGML_LLAMAFILE=ON",
            "-DGGML_BLAS=OFF",
            "-DGGML_OPENMP=OFF",
            f"-DGGML_NATIVE={'ON' if native else 'OFF'}",
            "-DGGML_CUDA=OFF",
            "-DGGML_VULKAN=OFF",
        ]
        cmake_args += _platform_cmake_args()
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            # if not single_config and not contains_arch:
            #     cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]
            # Enable metal by default
            cmake_args += ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"]
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        # if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
        #     # self.parallel is a Python 3 only way to set parallel jobs by hand
        #     # using -j in the build_ext call, not supported by pip or PyPA-build.
        #     if hasattr(self, "parallel") and self.parallel:
        #         # CMake 3.12+ only.
        #         build_args += [f"-j{self.parallel}"]

        # Compile in parallel by default. RISC-V cibuildwheel uses QEMU, where
        # unconstrained parallelism can be counterproductive.
        build_jobs = _cmake_build_jobs()
        build_args += [f"-j{build_jobs}" if build_jobs else "-j"]

        build_temp = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_temp, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", "--target", "_C", *build_args], cwd=build_temp, check=True)


HERE = Path(__file__).resolve().parent


def _normalize_tag_version(tag: str) -> str | None:
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    if tag.startswith("v"):
        version = tag.removeprefix("v")
    else:
        return None
    return version if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version) else None


def _git_tag_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=HERE,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for tag in result.stdout.splitlines():
        version = _normalize_tag_version(tag.strip())
        if version is not None:
            return version
    return None


def _source_version() -> str:
    version_match = re.search(
        r'__version__ = "(.*?)"',
        (HERE / "embeddings_cpp/__init__.py").read_text(encoding="utf-8"),
    )
    if version_match is None:
        raise RuntimeError("Failed to find version string in embeddings_cpp/__init__.py")
    return version_match.group(1)


version = (
    os.environ.get("EMBEDDINGS_CPP_VERSION")
    or _normalize_tag_version(os.environ.get("GITHUB_REF_NAME", ""))
    or _normalize_tag_version(os.environ.get("GITHUB_REF", ""))
    or _git_tag_version()
    or _source_version()
)
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="embeddings-cpp",
    version=version,
    description="GGML-based text embedding inference with Hugging Face tokenizers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="embeddings.cpp contributors",
    url="https://github.com/chux0519/embeddings.cpp",
    project_urls={
        "Source": "https://github.com/chux0519/embeddings.cpp",
        "Issues": "https://github.com/chux0519/embeddings.cpp/issues",
        "Snowflake GGUF": "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={"embeddings_cpp": ["registry.json"]},
    extras_require={
        "hub": ["huggingface_hub>=0.23"],
        "server": ["fastapi>=0.110", "uvicorn[standard]>=0.27", "huggingface_hub>=0.23"],
        "dev": ["huggingface_hub>=0.23", "fastapi>=0.110", "uvicorn[standard]>=0.27", "psutil", "requests", "numpy"],
    },
    ext_modules=[CMakeExtension("embeddings_cpp._C")],
    cmdclass={"build_ext": CMakeBuild},
)
