# Retrieval evaluation

`movie-rag eval` scores how well a target profile retrieves the right real movies. Use it
before and after every change to the catalog text, tag filtering, embedding model or enrichment,
so improvements are measured instead of eyeballed.

## How it works

- **Queries** (`movie_rag/data/eval_queries.json`): 30 target profiles written the way the SLM
  writes them, covering different vibes (heists, courtroom dramas, stoner comedies, ...).
- **Ground truth from the MovieLens tag genome**, not from anyone's taste. Each query names 1-2
  anchor genome tags; a movie is relevant if the genome scores **every** anchor ≥ 0.5.
  For example `buddy_road` = "buddy movie" + "road trip" → 21 of the 3,794 indexed movies.
- **No keyword leakage.** A query's text never contains its anchor words (enforced by
  `tests/test_evaluation.py`): `buddy_road` says "two mismatched companions forced to cross the
  country together", not "buddy road trip". A good score means the retriever understood the
  description, not that it found the tag word in the movie's text.
- **Retrieval only.** Profiles are embedded and searched directly; the SLM is bypassed.

Requires an index built from MongoDB with ml-25m data, and the genome files in `data/raw/ml-25m`.

## Metrics

| Metric | Meaning |
|---|---|
| **P@10** | Fraction of the top 10 results that are relevant. The headline number: what a user sees. |
| **R@100** | Fraction of all relevant movies found in the top 100. |
| **RR** | 1 / rank of the first relevant result (1.0 = the top hit is relevant). MRR is the mean. |
| **random** | P@10 a random ranking would get (relevant / indexed). Shows how hard a query is. |

## Usage

```bash
movie-rag eval                                    # table of per-query scores and means
movie-rag eval --details                          # also each query's top 10, hits marked ✓
movie-rag eval --label "bge-small" --out data/eval/bge-small.json   # save a report to compare
```

## Results

| Run | P@10 | R@100 | MRR | Report |
|---|---|---|---|---|
| all-MiniLM-L6-v2, ml-25m genome + TMDb, generic tags dropped | 0.567 | 0.455 | 0.796 | [baseline-minilm.json](../data/eval/baseline-minilm.json) |
| bge-small-en-v1.5, same catalog text, with query prefix | 0.560 | 0.494 | 0.810 | [bge-small.json](../data/eval/bge-small.json) |
| bge-small-en-v1.5, same catalog text, no query prefix | 0.577 | 0.506 | 0.828 | [bge-small-noprefix.json](../data/eval/bge-small-noprefix.json) |
| **MiniLM + genome blend, α = 0.7 (default)** | **0.763** | **0.658** | **0.929** | [genome-blend.json](../data/eval/genome-blend.json) |
| MiniLM + genome blend, α = 0.7, anchors hidden (leak-free) | 0.650 | 0.565 | 0.845 | [genome-blend-hidden.json](../data/eval/genome-blend-hidden.json) |

Random P@10 is 0.016, so the baseline is 35× better than chance.

**bge-small vs MiniLM:** a wash on P@10 (13 queries better, 12 worse, 5 the same) and a
modest gain in recall@100 (+4-5 points), at twice the build time (21 vs 10 min on an Intel CPU).
The model isn't the bottleneck; the catalog text is. MiniLM stays the default. To use BGE:

```bash
MOVIE_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 MOVIE_RAG_DB_PATH=chroma_db_bge movie-rag build --source mongo --min-ratings 1000
MOVIE_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 MOVIE_RAG_DB_PATH=chroma_db_bge movie-rag eval
```

The index records which model built it, and querying it with a different model is refused.

## Genome-blended ranking

`rank()` in `movie_rag/pipeline.py` blends two signals, each z-scored across the candidates:

```
score = α · text_similarity + (1 − α) · genome_score
```

The genome score embeds all 1,128 tag-genome tag names, takes the 20 closest non-generic tags to
the target profile (its *implied genome*), and scores each movie by its genome relevance on
those tags. For `buddy_road` the implied tags include "road trip", "buddy movie", "road movie"
and "unlikely friendships", from a description containing none of those words.

**This test favours it.** The ground truth is defined by genome tags, and genome scoring can
find those exact tags. So `--hide-anchors` removes each query's anchor tags from genome scoring:
whatever gain survives comes from *related* tags, not from rediscovering the label.

| α | P@10 | P@10, anchors hidden |
|---|---|---|
| 1.0 (text only) | 0.563 | 0.563 |
| **0.7 (default)** | 0.763 | **0.650** |
| 0.5 | 0.780 | 0.593 |
| 0.3 | 0.763 | 0.567 |
| 0.0 (genome only) | 0.753 | 0.527 |

α = 0.7 is the best leak-free setting and close to the best overall. Pure genome scoring is
mostly label leakage (it drops below text-only when anchors are hidden). Even leak-free, the
blend beats text-only on P@10 by 9 points, recall@100 by 11 and MRR by 5; 18 queries better,
5 worse. With anchors visible: 24 better, 1 worse (`buddy_road`, 0.1 → 0.0).

The real catalog always has the exact tags available, so actual quality likely sits between the
two columns. α was chosen on the same 30 queries it's reported on; it wins in both conditions,
but treat the numbers as optimistic. Run `movie-rag eval --alpha 1` for text-only.

## What the baseline gets wrong

MiniLM matches surface words, including words in titles. `buddy_road` scores P@10 = 0.10:
"by car, bus and train … friendship" retrieves *Strangers on a Train*, *The Station Agent*,
*Friends with Benefits* and *Driving Miss Daisy*. `black_satire` ("stupidity", "funniest") gets
spoofs like *Scary Movie* instead of dark satire, and `stoner` ("all-night") gets *Night Shift*.

## Caveats

- The genome labels are strict: *The Odd Couple* is a fair answer to `buddy_road`'s bickering
  pair but has no road trip, so it counts as a miss. Compare runs against each other, not
  against 1.0.
- 30 queries is small. Differences of a few points between runs can be noise; look for
  consistent gains across many queries (`--details`).
