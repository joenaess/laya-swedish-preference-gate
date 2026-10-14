"""Upload the calibrated checkpoint to the Hugging Face Hub (private) with a model card.

Runs only when HF_TOKEN is set, the licenses allow redistribution, every test passed and
post-calibration ECE meets the target. `huggingface_hub` reads HF_TOKEN from the environment
itself; the token is never passed explicitly, printed or written to disk.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import readme
from .utils import (
    CHECKPOINT_DIR,
    DEFAULT_HF_REPO,
    ECE_TARGET,
    GITHUB_REPO,
    METRICS_PATH,
    load_metrics,
    update_metrics,
)

UPLOAD_FILES = ("model.safetensors", "rl_agent_config.json", "temperature.json")
UPLOAD_DIRS = ("encoder", "tokenizer")


def blockers(m: dict) -> list[str]:
    out = []
    if not os.environ.get("HF_TOKEN"):
        out.append("HF_TOKEN not set")
    if not readme.get(m, "data.licenses.redistribution_allowed"):
        out.append("licenses do not clearly permit redistribution")
    tests = m.get("tests")
    if not (isinstance(tests, dict) and tests.get("all_passed")):
        out.append("not all tests passed (or tests were not run)")
    ece = readme.get(m, "calibration.ece_post")
    if not (isinstance(ece, float) and ece <= ECE_TARGET):
        out.append(f"post-calibration ECE {ece} is not <= {ECE_TARGET}")
    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        out.append("no trained checkpoint")
    return out


def model_card(m: dict, repo_id: str) -> str:
    lic = readme.get(m, "data.licenses", {})
    header = """---
language: [sv]
license: apache-2.0
base_model: convaiinnovations/laya
datasets: [jonasaise/dpo-swedish-interpreted]
pipeline_tag: text-classification
tags: [preference-model, reward-model, dpo, swedish, laya]
---
"""
    quick = readme.quickstart({**m, "publish": {"status": "uploaded", "repo_id": repo_id}})
    return f"""{header}
# Laya Swedish Preference Gate

A fast pairwise judge for Swedish responses: given a Swedish prompt and two candidate
responses it returns a calibrated probability that the first is better, in one forward pass
of a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model (mmBERT-base
backbone, `convaiinnovations/laya` subfolder `multilingual`). It is a "System 1" filter meant to
settle easy comparisons in milliseconds and escalate uncertain ones to an LLM judge.

Code, full results and the training pipeline: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO}).

## Results

{readme.results_table(m)}

### System 1 vs System 2 (Kimi-k3 judge)

{readme.judge_section(m)}

## Quickstart

Install the package from GitHub (`uv pip install git+https://github.com/{GITHUB_REPO}`), then:

{quick}

## Limitations

{readme.limitations({**m, "publish": {"status": "uploaded"}})}

## License

Apache-2.0. Base model: {lic.get("convaiinnovations/laya", "see repo")}; dataset:
{lic.get("jonasaise/dpo-swedish-interpreted", "see repo")}. The dataset is derived from
UltraFeedback; check upstream terms before commercial use.
"""


def main(argv=None) -> None:
    m = load_metrics()
    problems = blockers(m)
    if problems:
        reason = "upload skipped: " + "; ".join(problems)
        print(reason)
        update_metrics("publish", {"status": "skipped", "reason": reason})
        readme.main()
        sys.exit(0)

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_HF_REPO)
    api = HfApi()  # token from HF_TOKEN in the environment
    try:
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        info = api.repo_info(repo_id)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        reason = (
            f"upload skipped: the Hub refused to create `{repo_id}` (HTTP {status}); the "
            "HF_TOKEN in use cannot write to that namespace. Set HF_REPO_ID to a namespace the "
            "token can write to, or use a token for the intended account, and rerun "
            "`python -m laya_gate.publish`."
        )
        print(reason)
        update_metrics("publish", {"status": "skipped", "reason": reason})
        readme.main()
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in UPLOAD_FILES:
            shutil.copy2(CHECKPOINT_DIR / name, stage / name)
        for name in UPLOAD_DIRS:
            shutil.copytree(CHECKPOINT_DIR / name, stage / name)
        (stage / "results").mkdir()
        shutil.copy2(METRICS_PATH, stage / "results" / "metrics.json")
        (stage / "README.md").write_text(model_card(m, repo_id), encoding="utf-8")
        for f in stage.rglob("*"):  # nothing from the judge cache or the environment
            if f.is_file():
                text = f.read_bytes()
                for var in ("BASE_URL", "API_KEY", "HF_TOKEN"):
                    val = os.environ.get(var)
                    if val and val.encode() in text:
                        raise SystemExit(f"refusing to upload: {f.name} contains {var}")
        commit = api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Laya Swedish preference gate checkpoint",
        )

    out = {
        "status": "uploaded",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "visibility": "private" if info.private else "public",
        "commit": getattr(commit, "oid", None),
        "files": sorted([*UPLOAD_FILES, *UPLOAD_DIRS, "results/metrics.json", "README.md"]),
    }
    update_metrics("publish", out)
    readme.main()  # adds the Hub link and Hub quickstart to the GitHub README
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
