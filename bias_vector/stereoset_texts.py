"""Load StereoSet intrasentence examples as full sentences using only the stereotype option."""

from __future__ import annotations

from typing import Literal

from datasets import Dataset, load_dataset


BiasType = Literal["race", "profession", "gender", "religion", "all"]


def _label_is_stereotype(lab) -> bool:
    if isinstance(lab, str):
        return lab == "stereotype"
    if isinstance(lab, int):
        # ClassLabel order: anti-stereotype=0, stereotype=1, unrelated=2
        return lab == 1
    name = getattr(lab, "name", None)
    if name is not None:
        return str(name) == "stereotype"
    return str(lab) == "stereotype"


def _stereotype_sentence_from_example(ex: dict) -> str | None:
    sent_block = ex["sentences"]
    if isinstance(sent_block, list):
        for s in sent_block:
            lab = s["gold_label"]
            if _label_is_stereotype(lab):
                return s["sentence"].strip()
        return None
    labels = sent_block["gold_label"]
    sentences = sent_block["sentence"]
    if isinstance(labels, str):
        labels = [labels]
        sentences = [sentences]
    for lab, sent in zip(labels, sentences):
        if _label_is_stereotype(lab):
            return sent.strip()
    return None


def load_stereoset_stereotype_texts(
    bias_filter: BiasType = "all",
    split: str | None = None,
) -> Dataset:
    """
    Returns a HuggingFace Dataset with column ``text`` (one stereotype sentence per row).

    Uses config ``intrasentence``. If ``split`` is None, uses ``train`` if present else ``validation``.
    """
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence")
    if split is None:
        split = "train" if "train" in ds else "validation"
    d = ds[split]

    if bias_filter != "all":
        d = d.filter(lambda x: x["bias_type"] == bias_filter)

    texts = []
    for ex in d:
        s = _stereotype_sentence_from_example(ex)
        if s:
            texts.append({"text": s})

    return Dataset.from_list(texts)
