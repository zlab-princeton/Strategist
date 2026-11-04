"""
Workflow search and inspiration engine (paper A.4).

Each query is scored by the hybrid function

    s = alpha * s_tfidf + (1 - alpha) * s_emb        with alpha = 0.4

where `s_tfidf` is computed over the blueprint name, one-line description,
when-to-use field, and trait tags, and `s_emb` is the cosine similarity
between the query and a sentence-transformer (all-MiniLM-L6-v2) embedding of
the blueprint's measurement card and experience notes.

Retrieval issues two textual queries per iteration: a relevance query
summarising the task and an inspiration query phrased to deliberately drift
from it. Each returns its top-5, and the inspiration slate is deduplicated
against the relevance hits (see `retrieve`).

Document embeddings are cached on disk keyed by text hash, so the first call
pays the encoding cost once and later calls only encode new/changed cards.
"""

import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

STRATEGIST_DIR = Path(__file__).parent
INDEX_PATH = STRATEGIST_DIR / "index.json"
EXPERIENCE_PATH = STRATEGIST_DIR / "workflow_experience.json"
TESTED_PATH = STRATEGIST_DIR / "tested_workflows.json"
CARDS_DIR = STRATEGIST_DIR / "cards"
WORKFLOWS_DIR = STRATEGIST_DIR / "workflows"
# Per-blueprint sidecar written by the search when it adds a blueprint:
# which task it was designed for, its task kind, run and iteration.
LIBRARY_META_DIR = STRATEGIST_DIR / "library_meta"
EMB_CACHE_PATH = STRATEGIST_DIR / ".emb_cache.npz"

HYBRID_ALPHA = 0.4  # s = 0.4 * tfidf + 0.6 * embedding (paper A.4)

# ---------- Embedding model (lazy-loaded, runs on CPU) ----------

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    return model.encode(texts, normalize_embeddings=True)


# ---------- Index / experience loading ----------

def _load_index() -> dict:
    try:
        return json.loads(INDEX_PATH.read_text())
    except Exception:
        return {"workflows": {}, "blocks": {}}


def _load_experience() -> dict:
    try:
        return json.loads(EXPERIENCE_PATH.read_text())
    except Exception:
        return {}


def _load_tested() -> dict:
    """Blueprints with recorded test-set results (optional tested_workflows.json)."""
    try:
        return json.loads(TESTED_PATH.read_text()).get("workflows", {})
    except Exception:
        return {}


# ---------- TF-IDF (keyword matching) ----------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z]+', text.lower())


def _build_tfidf(documents: dict[str, str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    df = Counter()
    tf_per_doc = {}
    n = len(documents)
    for name, text in documents.items():
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        tf_per_doc[name] = {t: c / total for t, c in tf.items()}
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log(n / (1 + c)) for t, c in df.items()}
    tfidf = {}
    for name, tf in tf_per_doc.items():
        tfidf[name] = {t: tf_val * idf.get(t, 0) for t, tf_val in tf.items()}
    return tfidf, idf


def _cosine_sim_sparse(v1: dict[str, float], v2: dict[str, float]) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ---------- Corpora ----------
#
# Two views of the library (paper A.4):
#   tfidf corpus — name, one-line description, when_to_use, trait tags
#   embedding corpus — measurement card text + experience notes
#     (falls back to the tfidf text when a blueprint has no card yet)


_FILE_META_CACHE: dict[str, tuple[float, dict]] = {}


def _file_meta(path: Path) -> dict:
    """META of a blueprint file, parsed without importing it (cached by mtime)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    hit = _FILE_META_CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        from strategist.cards import _load_workflow_meta
        meta = _load_workflow_meta(path) or {}
    except Exception:
        meta = {}
    _FILE_META_CACHE[str(path)] = (mtime, meta)
    return meta


def library_meta(name: str) -> dict:
    """The sidecar of a blueprint (designed_for, task_kind, run, iteration), or {}."""
    try:
        return json.loads((LIBRARY_META_DIR / f"{name}.json").read_text())
    except Exception:
        return {}


def _with_library_meta(name: str, meta: dict) -> dict:
    side = library_meta(name)
    if not side:
        return meta
    out = dict(meta)
    tags = list(out.get("tags") or [])
    for key in ("designed_for", "task_kind"):
        if side.get(key):
            out[key] = side[key]
            if side[key] not in tags:
                tags.append(side[key])
    out["tags"] = tags
    return out


def _workflow_meta(include_tested: bool = True) -> dict[str, dict]:
    """Every blueprint in the library: the pre-built index, the tested set, and
    the files the running searches have added since the index was built (the
    index is not rebuilt during a sweep: concurrent writers would corrupt it).
    Sidecar task metadata is merged into each entry."""
    index = _load_index()
    workflow_meta = dict(index.get("workflows", {}))
    if include_tested:
        for name, info in _load_tested().items():
            if name in workflow_meta:
                continue
            meta = info.get("meta")
            if isinstance(meta, dict):
                workflow_meta[name] = meta
    try:
        for path in WORKFLOWS_DIR.glob("*.py"):
            name = path.stem
            if name.startswith("_") or name in workflow_meta:
                continue
            meta = _file_meta(path)
            if meta.get("description") or meta.get("name"):
                workflow_meta[name] = meta
    except OSError:
        pass
    return {name: _with_library_meta(name, meta) for name, meta in workflow_meta.items()}


def _tfidf_text(name: str, meta: dict) -> str:
    parts = [
        name.replace("_", " "),
        meta.get("description", ""),
        meta.get("when_to_use", ""),
        " ".join(meta.get("tags", []) or []),
        (f"designed for {meta['designed_for']}" if meta.get("designed_for") else ""),
    ]
    traits = meta.get("traits", {})
    if isinstance(traits, list):
        parts.append(" ".join(traits))
    elif isinstance(traits, dict):
        parts.append(" ".join(k for k, v in traits.items() if v is True))
        foundation = traits.get("foundation", "")
        if foundation:
            parts.append(f"foundation: {foundation}")
    return " ".join(p for p in parts if p)


def _emb_text(name: str, meta: dict, experience: dict) -> str:
    parts: list[str] = []
    card_path = CARDS_DIR / f"{name}.md"
    if card_path.exists():
        try:
            parts.append(card_path.read_text()[:4000])
        except OSError:
            pass
    for exp in experience.get(name, []):
        parts.append(exp.get("observation", ""))
        parts.append(exp.get("recommendation", ""))
    if not any(p.strip() for p in parts):
        parts = [_tfidf_text(name, meta)]
    return " ".join(p for p in parts if p)


def _build_corpora(include_tested: bool = True) -> tuple[dict[str, str], dict[str, str]]:
    workflow_meta = _workflow_meta(include_tested)
    experience = _load_experience()
    tfidf_corpus: dict[str, str] = {}
    emb_corpus: dict[str, str] = {}
    for name, meta in workflow_meta.items():
        tfidf_corpus[name] = _tfidf_text(name, meta)
        emb_corpus[name] = _emb_text(name, meta, experience)
    return tfidf_corpus, emb_corpus


# Back-compat: the combined corpus some callers (find_adjacent) use.
def _build_corpus(include_tested: bool = True) -> dict[str, str]:
    tfidf_corpus, emb_corpus = _build_corpora(include_tested)
    return {
        name: f"{tfidf_corpus.get(name, '')} {emb_corpus.get(name, '')}".strip()
        for name in tfidf_corpus
    }


# ---------- Document-embedding cache ----------

def _doc_embeddings(corpus: dict[str, str]) -> dict[str, np.ndarray]:
    """Return {name: unit vector} for every corpus doc, using the on-disk
    cache for unchanged texts and encoding only new/changed ones."""
    cached_names: list[str] = []
    cached_hashes: list[str] = []
    cached_vecs: np.ndarray | None = None
    if EMB_CACHE_PATH.exists():
        try:
            data = np.load(EMB_CACHE_PATH, allow_pickle=False)
            cached_names = [str(x) for x in data["names"]]
            cached_hashes = [str(x) for x in data["hashes"]]
            cached_vecs = data["vectors"]
        except Exception:
            cached_names, cached_hashes, cached_vecs = [], [], None

    cache = {}
    if cached_vecs is not None:
        for i, name in enumerate(cached_names):
            cache[name] = (cached_hashes[i], cached_vecs[i])

    out: dict[str, np.ndarray] = {}
    to_encode: list[tuple[str, str, str]] = []
    for name, text in corpus.items():
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        hit = cache.get(name)
        if hit is not None and hit[0] == h:
            out[name] = hit[1]
        else:
            to_encode.append((name, h, text))

    if to_encode:
        vectors = _embed([t for _, _, t in to_encode])
        for (name, h, _), vec in zip(to_encode, vectors):
            out[name] = vec
            cache[name] = (h, vec)
        try:
            # Atomic replace: 30 searches share this cache and a half-written
            # file would otherwise be read by another process.
            names = list(cache.keys())
            tmp = EMB_CACHE_PATH.with_name(f".emb_cache.{os.getpid()}.tmp.npz")
            np.savez(
                tmp,
                names=np.array(names),
                hashes=np.array([cache[n][0] for n in names]),
                vectors=np.stack([cache[n][1] for n in names]),
            )
            os.replace(tmp, EMB_CACHE_PATH)
        except Exception:
            pass
    return out


# ---------- Hybrid scoring ----------

def _hybrid_scores(
    query: str,
    candidates: set[str],
    tfidf_corpus: dict[str, str],
    emb_corpus: dict[str, str],
    alpha: float = HYBRID_ALPHA,
) -> list[tuple[str, float]]:
    """Score every candidate with s = alpha*tfidf + (1-alpha)*emb."""
    names = [n for n in tfidf_corpus if n in candidates]
    if not names:
        return []

    docs = {n: tfidf_corpus[n] for n in names}
    docs["__query__"] = query
    tfidf, _ = _build_tfidf(docs)
    query_vec = tfidf.get("__query__", {})
    tfidf_scores = {n: _cosine_sim_sparse(query_vec, tfidf.get(n, {})) for n in names}

    emb_scores: dict[str, float] = {n: 0.0 for n in names}
    try:
        doc_vecs = _doc_embeddings({n: emb_corpus.get(n, "") for n in names})
        q_vec = _embed([query])[0]
        for n in names:
            v = doc_vecs.get(n)
            if v is not None:
                emb_scores[n] = float(np.dot(v, q_vec))
    except Exception as exc:
        # Embedding backend unavailable. The paper's retrieval is hybrid
        # (TF-IDF + embeddings), so do not degrade silently: fail when
        # STRATEGIST_REQUIRE_EMBEDDINGS=1, otherwise warn and use TF-IDF.
        if os.environ.get("STRATEGIST_REQUIRE_EMBEDDINGS") == "1":
            raise
        print(f"[workflow_search] WARNING: embeddings unavailable ({exc!r}); "
              "retrieval is TF-IDF only, not the hybrid scoring the paper describes",
              flush=True)
        alpha = 1.0

    scored = [
        (n, alpha * tfidf_scores.get(n, 0.0) + (1.0 - alpha) * emb_scores.get(n, 0.0))
        for n in names
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------- Search API ----------

def search(query: str, tags: list[str] = None, traits: dict = None, top_k: int = 10,
           tier: str = "raw", bench: str | None = None) -> list[tuple[str, float]]:
    """
    Hybrid search over the library: every candidate scored by
    s = 0.4*tfidf + 0.6*embedding, top_k returned.

    tier: admission filter
        "raw"          — no filter (default)
        "tested"       — only workflows with ≥1 held-out test-set record
        "bench_locked" — only workflows with ≥1 test record on `bench`

    bench: required when tier == "bench_locked".
    """
    index = _load_index()
    tfidf_corpus, emb_corpus = _build_corpora()

    candidates = set(tfidf_corpus.keys())

    if tier == "tested":
        tested = _load_tested()
        candidates = {n for n in candidates if n in tested}
    elif tier == "bench_locked":
        if not bench:
            raise ValueError("tier='bench_locked' requires a `bench` argument")
        tested = _load_tested()
        candidates = {
            n for n in candidates
            if n in tested and bench in (tested[n].get("tested_benchmarks") or [])
        }
    if tags:
        for tag in tags:
            tag_lower = tag.lower()
            candidates = {
                name for name in candidates
                if tag_lower in [t.lower() for t in index["workflows"].get(name, {}).get("tags", [])]
            }
    if traits:
        def _wf_trait(wf_name, tname):
            t = index["workflows"].get(wf_name, {}).get("traits", {})
            if isinstance(t, list):
                return tname in t
            if isinstance(t, dict):
                return t.get(tname)
            return None
        for trait_name, trait_val in traits.items():
            candidates = {
                name for name in candidates
                if _wf_trait(name, trait_name) == trait_val
            }
    # If empty after filters, fall back to full corpus — UNLESS a hard tier
    # filter was requested, in which case empty means "no workflow qualifies".
    if not candidates and tier == "raw":
        candidates = set(tfidf_corpus.keys())
    if not candidates:
        return []

    return _hybrid_scores(query, candidates, tfidf_corpus, emb_corpus)[:top_k]


def retrieve(
    relevance_query: str,
    inspiration_query: str,
    k: int = 5,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Two-query retrieval (paper A.4).

    The relevance query returns the top-k blueprints for the task at hand;
    the inspiration query — phrased to deliberately drift from the task —
    returns the top-k after removing any blueprint already surfaced by the
    relevance query, so the inspiration slate cannot overlap.
    """
    tfidf_corpus, emb_corpus = _build_corpora()
    candidates = set(tfidf_corpus.keys())
    if not candidates:
        return [], []

    relevant = _hybrid_scores(relevance_query, candidates, tfidf_corpus, emb_corpus)[:k]
    taken = {name for name, _ in relevant}
    inspiration_all = _hybrid_scores(
        inspiration_query, candidates, tfidf_corpus, emb_corpus
    )
    inspiration = [(n, s) for n, s in inspiration_all if n not in taken][:k]
    return relevant, inspiration


# ---------- Adjacent Inspiration ----------

def find_adjacent(current_workflows: list[str], current_task: str, n: int = 3) -> list[tuple[str, str]]:
    """
    Find workflows ADJACENT to what the strategist currently uses.
    Similar in approach but from different domains / with different traits.
    """
    index = _load_index()
    corpus = _build_corpus()
    experience = _load_experience()

    if not corpus or not current_workflows:
        all_names = list(corpus.keys())
        random.shuffle(all_names)
        return [(name, "random discovery") for name in all_names[:n]]

    # Semantic similarity to the combined description of current workflows
    combined_text = " ".join(corpus.get(w, "") for w in current_workflows if w in corpus)
    if not combined_text.strip():
        all_names = list(corpus.keys())
        random.shuffle(all_names)
        return [(name, "random discovery") for name in all_names[:n]]

    other_names = [name for name in corpus if name not in current_workflows]
    if not other_names:
        return []

    try:
        texts = [combined_text] + [corpus[name] for name in other_names]
        embeddings = _embed(texts)
        current_emb = embeddings[0]
        other_embs = embeddings[1:]
        scores = other_embs @ current_emb
        scored = list(zip(other_names, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        # Fallback to TF-IDF
        all_docs = dict(corpus)
        all_docs["__current__"] = combined_text
        tfidf, _ = _build_tfidf(all_docs)
        current_vec = tfidf.get("__current__", {})
        scored = []
        for name in other_names:
            sim = _cosine_sim_sparse(current_vec, tfidf.get(name, {}))
            scored.append((name, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

    # Adjacent zone: ranks 3-15 (not the most similar, not the least)
    adjacent_zone = scored[2:15] if len(scored) > 15 else scored
    random.shuffle(adjacent_zone)
    selected = adjacent_zone[:n]

    results = []
    for name, sim in selected:
        meta = index["workflows"].get(name, {})
        desc = meta.get("description", "")[:80]
        exps = experience.get(name, [])
        cross_task = [e for e in exps if current_task.lower() not in e.get("task_type", "").lower()]
        if cross_task:
            exp = random.choice(cross_task)
            reason = f"Used on {exp['task_type']}: {exp.get('observation', '')[:60]}"
        else:
            reason = f"Similar approach: {desc}"
        results.append((name, reason))

    return results


# ---------- Selective Experience ----------

def get_relevant_experience(current_task: str, current_workflows: list[str], max_relevant: int = 5, max_cross: int = 3) -> str:
    experience = _load_experience()
    if not experience:
        return "(no experience recorded yet)"

    lines = []
    task_lower = current_task.lower()
    for wf in current_workflows:
        for exp in experience.get(wf, []):
            if task_lower in exp.get("task_type", "").lower():
                lines.append(
                    f"- `{wf}` on {exp['task_type']}: {exp.get('observation', '')[:100]}"
                    f"{' → ' + exp.get('recommendation', '')[:60] if exp.get('recommendation') else ''}"
                )
    relevant = lines[:max_relevant]

    cross = []
    all_exps = [(wf, exp) for wf, exps in experience.items() for exp in exps
                if task_lower not in exp.get("task_type", "").lower()]
    if all_exps:
        random.shuffle(all_exps)
        for wf, exp in all_exps[:max_cross]:
            cross.append(f"- `{wf}` on {exp['task_type']}: {exp.get('observation', '')[:100]}")

    result = ""
    if relevant:
        result += "Relevant to your task:\n" + "\n".join(relevant)
    if cross:
        result += "\n\nFrom other tasks (cross-pollination):\n" + "\n".join(cross)
    return result or "(no experience recorded yet)"


def format_search_prompt(query: str, results: list[tuple[str, float]], index: dict = None) -> str:
    if index is None:
        index = _load_index()
    lines = []
    for name, score in results:
        meta = index.get("workflows", {}).get(name, {})
        desc = meta.get("description", "")[:80]
        lines.append(f"- `{name}` (relevance: {score:.2f}): {desc}")
    return "\n".join(lines) or "(no results)"
