"""A word trigram baseline for Swedish minimal pairs ("is this pair n-gram solvable?").

    python -m laya_gate.ngram build   # count trigrams over Europarl-SV sentences (cached)

Stupid backoff (Brants et al., 2007) over lowercased word and punctuation tokens from
Språkbanken's Europarl-SV (CC BY 4.0); a sentence scores its mean log10 probability per token,
so minimal pairs of different lengths compare by perplexity, as in BLiMPPI (arXiv:2610.13607).
Europarl is parliamentary and largely translated Swedish, so this is a domain-mismatched,
cheap reference point, not a strong language model.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import Counter
from pathlib import Path

from .utils import CACHE_DIR

NGRAM_PATH = CACHE_DIR / "ngram" / "europarl_sv_trigram.pkl"
N_SENTENCES = 1_000_000
BACKOFF = 0.4
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens(text: str) -> list[str]:
    return ["<s>", "<s>", *_TOKEN.findall(text.lower()), "</s>"]


class Trigram:
    def __init__(self, counts: dict[int, Counter]):
        self.c = counts
        self.total = sum(counts[1].values())
        self.vocab = len(counts[1])

    @classmethod
    def from_sentences(cls, sentences) -> Trigram:
        counts = {1: Counter(), 2: Counter(), 3: Counter()}
        for s in sentences:
            t = tokens(s)
            for i in range(2, len(t)):
                counts[1][t[i]] += 1
                counts[2][(t[i - 1], t[i])] += 1
                counts[3][(t[i - 2], t[i - 1], t[i])] += 1
            counts[2][("<s>", "<s>")] += 1  # history count for sentence starts
        return cls(counts)

    def score_token(self, u: str, v: str, w: str) -> float:
        c3, c2 = self.c[3].get((u, v, w), 0), self.c[2].get((u, v), 0)
        if c3 and c2:
            return c3 / c2
        c2w, c1 = self.c[2].get((v, w), 0), self.c[1].get(v, 0)
        if c2w and c1:
            return BACKOFF * c2w / c1
        return BACKOFF * BACKOFF * (self.c[1].get(w, 0) + 1) / (self.total + self.vocab)

    def mean_logprob(self, text: str) -> float:
        t = tokens(text)
        lp = [math.log10(self.score_token(t[i - 2], t[i - 1], t[i])) for i in range(2, len(t))]
        return sum(lp) / len(lp)


def europarl_sentences(n: int = N_SENTENCES):
    """Sentences from the Swedish Europarl XML, as space-joined tokens."""
    from .external_eval import EP_DIR, _bz2_lines

    cur, k = None, 0
    for line in _bz2_lines(EP_DIR / "europarl-sv.xml.bz2"):
        if line.startswith("<sentence"):
            cur = []
        elif line.startswith("<w ") and cur is not None:
            cur.append(line[line.index(">") + 1 : line.rindex("</w>")])
        elif line.startswith("</sentence>") and cur is not None:
            yield " ".join(cur)
            cur = None
            k += 1
            if k >= n:
                return


def build(n: int = N_SENTENCES, path: Path = NGRAM_PATH) -> Trigram:
    import html

    model = Trigram.from_sentences(html.unescape(s) for s in europarl_sentences(n))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model.c, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model


def load(path: Path = NGRAM_PATH) -> Trigram:
    if not path.exists():
        return build(path=path)
    with open(path, "rb") as f:
        return Trigram(pickle.load(f))  # noqa: S301 - our own cache file


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--n", type=int, default=N_SENTENCES)
    args = parser.parse_args(argv)
    m = build(args.n)
    print(f"trigram model: {m.total:,} tokens, {m.vocab:,} types, {len(m.c[3]):,} trigrams")


if __name__ == "__main__":
    main()
