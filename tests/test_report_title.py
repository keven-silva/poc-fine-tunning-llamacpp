import importlib.util
from pathlib import Path

REPORT = Path(__file__).parent.parent / "scripts" / "06_report.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("report_script", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_title_drops_the_organisation_prefix():
    assert _load_report().report_title("unsloth/Qwen3-0.6B") == "Qwen3-0.6B persona fine-tune"


def test_title_keeps_a_bare_model_name():
    assert _load_report().report_title("Qwen3-8B") == "Qwen3-8B persona fine-tune"


def test_title_uses_the_last_path_segment_of_a_nested_id():
    assert _load_report().report_title("org/sub/Model-X") == "Model-X persona fine-tune"


def test_report_source_no_longer_hardcodes_a_model_name():
    assert "Qwen3-8B persona" not in REPORT.read_text(encoding="utf-8")
