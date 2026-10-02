import re
from pathlib import Path

import pandas as pd
import pytest

from movie_rag.evaluation import ground_truth, load_queries, precision_at_k, recall_at_n, reciprocal_rank

GENOME_DIR = Path("data/raw/ml-25m")


def test_metrics():
    ranked = [5, 1, 7, 2]
    relevant = {1, 2, 9}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_n(ranked, relevant, 4) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank([5, 7], relevant) == 0.0


def test_query_ids_are_unique():
    ids = [q.id for q in load_queries()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("query", load_queries(), ids=lambda q: q.id)
def test_profile_does_not_contain_its_anchor_words(query):
    """Otherwise the eval rewards keyword overlap with the embedded tags, not understanding."""
    text = query.profile.lower()
    for tag in query.anchor_tags:
        for word in re.findall(r"[a-z]+", tag):
            if len(word) >= 4:
                assert word not in text, f"{query.id}: profile contains anchor word {word!r}"


def test_ground_truth_requires_every_anchor(tmp_path):
    pd.DataFrame({"tagId": [1, 2], "tag": ["road trip", "buddy movie"]}).to_csv(tmp_path / "genome-tags.csv", index=False)
    pd.DataFrame(
        {"movieId": [10, 10, 20, 20, 30, 30], "tagId": [1, 2, 1, 2, 1, 2], "relevance": [0.9, 0.8, 0.9, 0.2, 0.6, 0.7]}
    ).to_csv(tmp_path / "genome-scores.csv", index=False)
    queries = [q for q in load_queries() if q.id == "buddy_road"]
    assert ground_truth(queries, tmp_path, {10, 20, 30}) == {"buddy_road": {10, 30}}
    assert ground_truth(queries, tmp_path, {10, 20}) == {"buddy_road": {10}}  # only indexed movies count


@pytest.mark.skipif(not (GENOME_DIR / "genome-tags.csv").exists(), reason="ml-25m not downloaded")
def test_all_anchor_tags_exist_in_genome():
    genome = set(pd.read_csv(GENOME_DIR / "genome-tags.csv")["tag"])
    missing = {t for q in load_queries() for t in q.anchor_tags} - genome
    assert not missing
