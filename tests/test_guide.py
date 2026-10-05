import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
GUIDE = ROOT / "docs" / "guides" / "adapting-to-a-new-niche.md"

# Created while the pipeline runs, or tool output folders: not checked into git.
RUNTIME_PREFIXES = ("data/", "outputs/", "vendor/")


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_has_the_eight_sections_in_order():
    headings = re.findall(r"^## (\d)\. ", _text(), re.M)
    assert headings == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_every_file_the_guide_names_exists():
    prose = re.sub(r"```.*?```", "", _text(), flags=re.S)  # fenced blocks may show runtime output
    named = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|yaml|md|sh|toml))`", prose))
    assert named, "the guide should reference real files"
    missing = sorted(
        p for p in named
        if not p.startswith(RUNTIME_PREFIXES) and not (ROOT / p).exists()
    )
    assert missing == [], f"the guide names files that do not exist: {missing}"


def test_guide_covers_both_example_niches_and_the_retrieval_caveat():
    text = _text().lower()
    for needle in ("cyber security", "política", "retrieval"):
        assert needle in text, needle


def test_guide_states_the_core_invariants_a_new_niche_must_keep():
    text = _text()
    for needle in ("/completion", "enable_thinking=False", "PROMPT_VERSION"):
        assert needle in text, needle
