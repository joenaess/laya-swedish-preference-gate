"""Fine-tune the multilingual Laya checkpoint (mmBERT-base backbone) on Swedish DPO pairs.

Adapted from the reference recipe (https://github.com/NandhaKishorM/laya, `laya.train`):
the same item encoding (`build_sequence`), soft cross-entropy over the `choice` options,
AdamW + gradient-norm clipping and the checkpoint layout `laya.load` reads. Differences,
per this project's spec: one learning rate (2e-5) with 10% linear warmup and cosine decay,
bfloat16 autocast instead of fp16 + GradScaler, effective batch 16, best epoch kept by loss on
the calibration half of validation.
"""

from __future__ import annotations

import os

# Must be set before the first CUDA allocation; reduces fragmentation at 1024-token batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import build_items
from .utils import (
    BASE_MODEL_REPO,
    BASE_MODEL_SUBFOLDER,
    CHECKPOINT_DIR,
    HEAD_MAX_LEN,
    MAX_LEN,
    SEED,
    SPLITS_DIR,
    read_jsonl,
    update_metrics,
)


@dataclass
class Hyper:
    epochs: int = 3
    micro_batch: int = 8
    grad_accum: int = 2
    lr: float = 2e-5
    warmup_frac: float = 0.10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False
    eval_batch: int = 32
    seed: int = SEED
    max_len: int = MAX_LEN
    head_max_len: int = HEAD_MAX_LEN
    loss: str = "soft cross-entropy over the two choice options"
    oom_fallback: str = "none"


def encode_all(tok, items):
    from laya.train import encode_item

    out = []
    for it in items:
        enc = encode_item(tok, it, MAX_LEN, HEAD_MAX_LEN)
        enc["label"] = it["label"]
        out.append(enc)
    return out


def batch_loss(model, batch, device):
    """Mean cross-entropy and per-row logits for a collated batch with `target`."""
    args = [
        batch[k].to(device)
        for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model(*args)
    logits = logits[:, :2].float()
    target = batch["target"][:, :2].to(device)
    loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
    return loss, logits


def _collate_train(chunk, pad_id):
    from laya.common import collate_items

    return collate_items([chunk], pad_id)


@torch.no_grad()
def evaluate(model, encoded, pad_id, device, batch_size=32):
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    losses, correct = [], []
    for s in range(0, len(order), batch_size):
        chunk = [encoded[i] for i in order[s : s + batch_size]]
        batch = _collate_train(chunk, pad_id)
        loss, logits = batch_loss(model, batch, device)
        losses.append(loss.item() * len(chunk))
        pred = logits.argmax(-1).cpu().numpy()
        correct.extend((pred == np.array([c["label"] for c in chunk])).tolist())
    return sum(losses) / len(encoded), float(np.mean(correct))


def try_step(model, encoded, pad_id, device, hp: Hyper) -> bool:
    """Run two forward+backward+AdamW steps on the longest items with a throwaway optimizer.

    The second step runs with the optimizer state already allocated, as every real step does.
    Returns False on CUDA OOM. The weights are restored afterwards, so the probe does not train.
    """
    longest = sorted(encoded, key=lambda e: -len(e["ids"]))[: hp.micro_batch]
    snapshot = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    ok = True
    try:
        model.train()
        for _ in range(2):
            loss, _ = batch_loss(model, _collate_train(longest, pad_id), device)
            loss.backward()
            opt.step()
            del loss
    except torch.cuda.OutOfMemoryError:
        ok = False
    finally:
        opt.zero_grad(set_to_none=True)
        del opt
        model.zero_grad(set_to_none=True)
        model.load_state_dict(snapshot)
        torch.cuda.empty_cache()
    return ok


def set_checkpointing(model, on: bool) -> None:
    if on:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.encoder.gradient_checkpointing_disable()
    model.head_checkpointing = on


def choose_memory_config(model, encoded, pad_id, device, hp: Hyper) -> Hyper:
    """Spec fallback chain: 8x2 -> 8x2 + gradient checkpointing -> 4x4 + checkpointing."""
    for micro, accum, ckpt, label in (
        (8, 2, False, "none"),
        (8, 2, True, "gradient_checkpointing"),
        (4, 4, True, "gradient_checkpointing+batch4x4"),
    ):
        hp.micro_batch, hp.grad_accum, hp.gradient_checkpointing = micro, accum, ckpt
        set_checkpointing(model, ckpt)
        print(f"memory probe: micro_batch={micro} accum={accum} ckpt={ckpt}", flush=True)
        if device.type != "cuda" or try_step(model, encoded, pad_id, device, hp):
            if device.type == "cuda":
                peak = torch.cuda.max_memory_reserved() / 2**30
                print(f"  -> fits (peak reserved {peak:.2f} GiB)", flush=True)
            hp.oom_fallback = label
            return hp
        print("  -> OOM", flush=True)
    raise RuntimeError("out of memory even at micro_batch=4 with gradient checkpointing")


def save(model, tok, cfg, hp: Hyper, path, extra) -> None:
    from laya.train import save_checkpoint

    cfg = dict(cfg)
    cfg["max_len"], cfg["head_max_len"] = hp.max_len, hp.head_max_len
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["temperature_by_options"] = {}
    cfg.pop("lang_temperatures", None)
    cfg["training"] = {
        "fine_tuned_from_checkpoint": f"{BASE_MODEL_REPO}/{BASE_MODEL_SUBFOLDER}",
        "laya_swedish_preference_gate": {**asdict(hp), **extra},
    }
    if path.exists():
        shutil.rmtree(path)
    save_checkpoint(model, tok, cfg, str(path))


def main(argv=None) -> None:
    from laya.train import load_checkpoint
    from transformers import get_cosine_schedule_with_warmup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=Hyper.epochs)
    parser.add_argument("--limit", type=int, default=None, help="debug: N train items")
    args = parser.parse_args(argv)

    hp = Hyper(epochs=args.epochs)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, cfg = load_checkpoint(BASE_MODEL_SUBFOLDER)
    model.to(device)
    pad_id = tok.pad_token_id

    splits = {k: read_jsonl(SPLITS_DIR / f"{k}.jsonl") for k in ("train", "val_calib", "val_eval")}
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    encoded = {k: encode_all(tok, build_items(tok, v)[0]) for k, v in splits.items()}
    print({k: len(v) for k, v in encoded.items()}, flush=True)

    base_calib = evaluate(model, encoded["val_calib"], pad_id, device, hp.eval_batch)
    base_eval = evaluate(model, encoded["val_eval"], pad_id, device, hp.eval_batch)
    print(f"zero-shot base: calib loss/acc {base_calib}, eval loss/acc {base_eval}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hp = choose_memory_config(model, encoded["train"], pad_id, device, hp)

    train = encoded["train"]
    micro_steps = math.ceil(len(train) / hp.micro_batch)
    updates_per_epoch = math.ceil(micro_steps / hp.grad_accum)
    total_updates = updates_per_epoch * hp.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(hp.warmup_frac * total_updates), total_updates
    )

    history, best = [], {"val_calib_loss": float("inf"), "epoch": None}
    t_start = time.perf_counter()
    for epoch in range(hp.epochs):
        t_epoch = time.perf_counter()
        model.train()
        order = list(range(len(train)))
        random.Random(hp.seed + epoch).shuffle(order)
        total, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, s in enumerate(range(0, len(order), hp.micro_batch)):
            chunk = [train[i] for i in order[s : s + hp.micro_batch]]
            loss, _ = batch_loss(model, _collate_train(chunk, pad_id), device)
            window_start = (step // hp.grad_accum) * hp.grad_accum
            window = min(hp.grad_accum, micro_steps - window_start)
            (loss / window).backward()
            total += loss.item()
            n += 1
            if (step + 1) % hp.grad_accum == 0 or step + 1 == micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                el = time.perf_counter() - t_epoch
                print(
                    f"epoch {epoch + 1} step {step + 1}/{micro_steps} loss {total / n:.4f} "
                    f"lr {scheduler.get_last_lr()[0]:.2e} {el:.0f}s",
                    flush=True,
                )
        train_time = time.perf_counter() - t_epoch
        c_loss, c_acc = evaluate(model, encoded["val_calib"], pad_id, device, hp.eval_batch)
        e_loss, e_acc = evaluate(model, encoded["val_eval"], pad_id, device, hp.eval_batch)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / n,
            "val_calib_loss": c_loss,
            "val_calib_acc": c_acc,
            "val_eval_loss": e_loss,
            "val_eval_acc": e_acc,
            "val_loss": (c_loss + e_loss) / 2,
            "val_acc": (c_acc + e_acc) / 2,
            "train_seconds": train_time,
            "epoch_seconds": time.perf_counter() - t_epoch,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if c_loss < best["val_calib_loss"]:
            best = {"val_calib_loss": c_loss, "epoch": epoch + 1}
            save(model, tok, cfg, hp, CHECKPOINT_DIR, {"best_epoch": epoch + 1})
            print(f"saved best checkpoint (epoch {epoch + 1})", flush=True)

    total_seconds = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None
    update_metrics(
        "training",
        {
            "base_checkpoint": f"{BASE_MODEL_REPO} (subfolder={BASE_MODEL_SUBFOLDER})",
            "backbone": cfg.get("encoder"),
            "hyperparameters": asdict(hp),
            "effective_batch": hp.micro_batch * hp.grad_accum,
            "optimizer_updates": total_updates,
            "train_items": len(train),
            "history": history,
            "best_epoch": best["epoch"],
            "selection": "lowest loss on the val_calib half; val_eval is never used for selection",
            "best_val_eval_acc": history[best["epoch"] - 1]["val_eval_acc"],
            "best_val_acc": history[best["epoch"] - 1]["val_acc"],
            "zero_shot_base": {
                "val_calib_loss": base_calib[0],
                "val_calib_acc": base_calib[1],
                "val_eval_loss": base_eval[0],
                "val_eval_acc": base_eval[1],
            },
            "total_seconds": total_seconds,
            "per_epoch_seconds": [h["epoch_seconds"] for h in history],
            "peak_gpu_memory_allocated_gib": peak,
            "peak_gpu_memory_reserved_gib": reserved,
            "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
