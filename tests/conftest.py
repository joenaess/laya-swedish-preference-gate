"""Shared fixtures: a tiny offline tokenizer and checkpoint, synthetic data, skip logic."""

from __future__ import annotations

import json
import os
import socket

import pytest
import torch

from laya_gate.utils import CHECKPOINT_DIR

SWEDISH = (
    "Fotosyntesen är den process där växter omvandlar ljusenergi till kemisk energi. "
    "Människor äter blåbär och jordgubbar på sommaren. Ödmjukhet, ärlighet och "
    "ansvarstagande är viktiga värden. Det är när solen lyser på träd och gör magi. "
    "Hur fungerar fotosyntesen? Vad är skillnaden mellan väder och klimat? "
    "Å, ä och ö är svenska bokstäver. Göteborg, Malmö och Västerås ligger i Sverige. "
)


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_network() -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    checks = {
        "gpu": (_has_cuda, "no CUDA device"),
        "network": (_has_network, "no Hugging Face Hub access"),
        "judge": (
            lambda: bool(os.environ.get("API_KEY") and os.environ.get("BASE_URL")),
            "API_KEY/BASE_URL not set",
        ),
    }
    cache: dict[str, bool] = {}
    for item in items:
        for marker, (check, reason) in checks.items():
            if marker in item.keywords:
                if marker not in cache:
                    cache[marker] = check()
                if not cache[marker]:
                    item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A small byte-level BPE trained in memory: offline, and it splits å/ä/ö into bytes."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tk.train_from_iterator([SWEDISH] * 20, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_tokenizer):
    """A randomly initialised 2-layer Laya checkpoint (max_len 256) in the `laya.load` layout."""
    from laya.common import DecisionModel
    from laya.train import save_checkpoint
    from transformers import AutoModel, ModernBertConfig

    torch.manual_seed(0)
    tok = tiny_tokenizer
    ecfg = ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
        local_attention=64,
    )
    model = DecisionModel(AutoModel.from_config(ecfg, attn_implementation="sdpa"), head_layers=1)
    cfg = {
        "encoder": "tiny-modernbert",
        "head_layers": 1,
        "max_len": 256,
        "head_max_len": 64,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
    }
    path = tmp_path_factory.mktemp("tiny_ckpt")
    save_checkpoint(model, tok, cfg, str(path))
    (path / "temperature.json").write_text(json.dumps({"temperature": 1.5}))
    return path


@pytest.fixture(scope="session")
def trained_checkpoint():
    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        pytest.skip("no trained checkpoint at checkpoints/best_model")
    return CHECKPOINT_DIR


@pytest.fixture()
def synthetic_pairs():
    import random

    rng = random.Random(0)
    words = SWEDISH.split()
    pairs = []
    for i in range(1000):
        pairs.append(
            {
                "id": f"p{i:04d}",
                "prompt": f"Fråga {i}: " + " ".join(rng.choices(words, k=8)),
                "chosen": " ".join(rng.choices(words, k=20)),
                "rejected": " ".join(rng.choices(words, k=12)),
            }
        )
    return pairs
