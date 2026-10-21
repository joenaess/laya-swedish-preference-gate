"""Offline tests for the checkpoint-signal helpers."""

from laya_gate import ckpt_eval as C


def test_split_prefix():
    para = (
        "Neurologi är en gren av medicinen som behandlar sjukdomar i nervsystemet. "
        "Dock kan begreppet även syfta på studiet av nervsystemet i helhet och dess "
        "funktioner hos människor och djur i alla åldrar."
    )
    prefix, rest = C.split_prefix(para)
    assert prefix.endswith("nervsystemet.") and rest.startswith("Dock kan")
    assert C.split_prefix("En mening utan fortsättning.") is None
    assert C.split_prefix("Kort. " + "ord " * 30 + ".") is None  # first sentence too short


def test_few_shot_prompt_ends_with_open_slot():
    shots = [{"en": "Hello.", "human": "Hej."}, {"en": "Thanks.", "human": "Tack."}]
    p = C.few_shot_prompt(shots, "Good night.")
    assert p.endswith("English: Good night.\nSwedish:")
    assert p.count("English:") == 3 and "Swedish: Tack." in p


def test_repetition():
    assert C.repetition("det är det är det är det är") > 0.5
    assert C.repetition("en två tre fyra fem sex") == 0.0
    assert C.repetition("kort") == 0.0


def test_labels_and_series():
    assert C.label("goldfish-models/swe_latn_100mb", None) == "goldfish-100mb"
    assert C.label(C.APERTUS_REPO, "step50000-tokens210B") == "apertus-210B"
    assert C.series(C.APERTUS_REPO) == "apertus"
    assert [p for _, _, p in C.APERTUS] == sorted(p for _, _, p in C.APERTUS)
