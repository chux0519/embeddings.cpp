from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def render_readme(tmp_path: Path, model_id: str) -> str:
    gguf = tmp_path / "artifact.gguf"
    output = tmp_path / "README.md"
    gguf.write_bytes(b"fake gguf payload")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_hf_readme.py"),
            "--model-id",
            model_id,
            "--gguf",
            str(gguf),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert len(result.stdout.strip()) == 64
    return output.read_text(encoding="utf-8")


def test_render_hf_readme_uses_bge_m3_default_template(tmp_path):
    readme = render_readme(tmp_path, "BAAI/bge-m3")

    assert "base_model: BAAI/bge-m3" in readme
    assert "`bge-m3.q8_0.gguf`" in readme
    assert "EMBEDDINGS_CPP_CPU_REPACK=1" in readme
    assert "model = load(\"BAAI/bge-m3\")" in readme
    assert "{{" not in readme


def test_render_hf_readme_keeps_snowflake_default_template_compatible(tmp_path):
    readme = render_readme(tmp_path, "Snowflake/snowflake-arctic-embed-m-v2.0")

    assert "base_model: Snowflake/snowflake-arctic-embed-m-v2.0" in readme
    assert "`snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`" in readme
    assert "mixed `q4_K` MLP + `q8_0` attention" in readme
    assert "{{" not in readme
