"""TF-IDF cosine similarity of documents; Levenshtein and Jaro-Winkler string distances.

`import qf`. Pure Python and numpy: no tokenizer package is loaded. A scalar result is a float; a function with several
results returns a dict with named keys.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ._common import Result, choice, integer

__all__ = ["string_similarity", "tfidf_cosine"]

TOKEN_RULES = ("sklearn", "alphanumeric", "letters")
CASES = ("lower", "keep")
TERM_WEIGHTS = ("raw", "sublinear", "frequency", "binary")
IDFS = ("smooth", "plain", "standard", "none")
NORMS = ("l2", "l1", "none")
METRICS = ("levenshtein", "levenshtein_similarity", "jaro", "jaro_winkler")
MATCH_CASES = ("sensitive", "insensitive")
LOG_BASES = ("e",)
PATTERNS = {"sklearn": re.compile(r"(?u)\b\w\w+\b"), "alphanumeric": re.compile(r"(?u)\b\w+\b"),
            "letters": re.compile(r"[^\W\d_]+")}


def tfidf_cosine(documents, tokens, case, stop_words, tf, idf, log_base, norm):
    """TF-IDF cosine similarity of documents (scikit-learn's TfidfVectorizer settings), for novelty and overlap.

    Terms by the token rule, the vocabulary every term, sorted. Term j in document i, count c, df_j of the n documents
    holding it: tf 'raw' c, 'sublinear' 1 + log c, 'frequency' c over the document's terms, 'binary' 1; idf 'smooth'
    log((1 + n)/(1 + df)) + 1, 'plain' log(n/df) + 1, 'standard' log(n/df), 'none' 1. Cosine v_i . v_j / (|v_i| |v_j|),
    0 for a document with no term; the norm leaves it unchanged.

    Novelty against the statement before: 1 - similarity[i, i - 1]; against all earlier, 1 - max similarity[i, :i]. The
    idf counts every document given: pass those up to a date for a point-in-time score.

    conventions:
    - documents: a list of strings
    - tokens: 'sklearn' (runs of two or more letters, digits or underscores), 'alphanumeric' (any run) or 'letters'
    - case: 'lower' (scikit-learn's) or 'keep'; stop_words: None or a list of words
    - tf: 'raw', 'sublinear' (scikit-learn's sublinear_tf), 'frequency' or 'binary'
    - idf: 'smooth' (scikit-learn's), 'plain' (its smooth_idf=False), 'standard' (the textbook's) or 'none'
    - log_base: 'e' (scikit-learn's) or a number (10 in the textbook); norm: 'l2' (scikit-learn's), 'l1' or 'none'
    returns: a dict: similarity (n x n), weights (n x terms, normed), vocabulary (the columns' terms)
    source: Manning, Raghavan and Schutze, Introduction to Information Retrieval (2008), ch. 6; scikit-learn TfidfVectorizer
    aliases: [TF-IDF], [tf-idf], [TFIDF], [term frequency inverse document frequency], TF-IDF cosine similarity, [novelty score], [document similarity], [cosine similarity], [bag of words]
    example:
        # similarity[0, 2] = 0.732359; similarity[0, 1] = 0.0.
        qf.tfidf_cosine(documents=["rates rise", "inflation eases", "rates rise again"], tokens="sklearn", case="lower",
                        stop_words=None, tf="raw", idf="smooth", log_base="e", norm="l2")
    """
    choice("tokens", tokens, TOKEN_RULES)
    choice("case", case, CASES)
    choice("tf", tf, TERM_WEIGHTS)
    choice("idf", idf, IDFS)
    choice("norm", norm, NORMS)
    if isinstance(documents, str) or not all(isinstance(d, str) for d in documents) or len(documents) < 1:
        raise ValueError("documents must be a list of strings")
    if isinstance(log_base, str):
        choice("log_base", log_base, LOG_BASES)
        base = math.e
    else:
        base = float(log_base)
        if not (base > 0 and base != 1.0 and math.isfinite(base)):
            raise ValueError("log_base must be 'e' or a positive number other than 1 (10 for the common log)")
    dropped = set() if stop_words is None else {w.lower() if case == "lower" else w for w in stop_words}
    parsed = []
    for text in documents:
        found = PATTERNS[tokens].findall(text.lower() if case == "lower" else text)
        parsed.append([word for word in found if word not in dropped])
    vocabulary = sorted({word for words in parsed for word in words})
    column = {word: j for j, word in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)))
    for i, words in enumerate(parsed):
        for word in words:
            counts[i, column[word]] += 1.0
    n, held = counts.shape[0], counts > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if tf == "raw":
            weight = counts
        elif tf == "sublinear":
            weight = np.where(held, 1.0 + np.log(np.where(held, counts, 1.0)) / math.log(base), 0.0)
        elif tf == "frequency":
            weight = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        else:
            weight = held.astype(float)
    df = held.sum(axis=0)
    if idf == "smooth":
        factor = np.log((1.0 + n) / (1.0 + df)) / math.log(base) + 1.0
    elif idf == "plain":
        factor = np.log(n / df) / math.log(base) + 1.0
    elif idf == "standard":
        factor = np.log(n / df) / math.log(base)
    else:
        factor = np.ones(len(vocabulary))
    weights = weight * factor
    if norm != "none":
        size = np.abs(weights).sum(axis=1) if norm == "l1" else np.sqrt((weights ** 2).sum(axis=1))
        weights = weights / np.where(size > 0, size, 1.0)[:, None]
    length = np.sqrt((weights ** 2).sum(axis=1))
    unit = weights / np.where(length > 0, length, 1.0)[:, None]
    return Result({"similarity": unit @ unit.T, "weights": weights, "vocabulary": vocabulary})


def _levenshtein(a: str, b: str) -> int:
    """Wagner and Fischer's table, one row at a time: insertions, deletions and substitutions cost one each."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(max(len(a), len(b)) // 2 - 1, 0)
    used_a, used_b = [False] * len(a), [False] * len(b)
    matches = 0
    for i, char in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not used_b[j] and b[j] == char:
                used_a[i] = used_b[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    out_of_order, j = 0, 0
    for i, char in enumerate(a):
        if used_a[i]:
            while not used_b[j]:
                j += 1
            out_of_order += char != b[j]
            j += 1
    return (matches / len(a) + matches / len(b) + (matches - out_of_order / 2.0) / matches) / 3.0


def string_similarity(a, b, metric, case, prefix_scale, prefix_cap, boost_threshold):
    """String similarity: Levenshtein edit distance, or the Jaro and Jaro-Winkler similarities of two strings.

    'levenshtein' the fewest single-letter insertions, deletions and substitutions turning a into b (Wagner-Fischer);
    'levenshtein_similarity' 1 - distance / the longer length. 'jaro' (m/|a| + m/|b| + (m - t)/m) / 3, m the letters
    matching within max(|a|, |b|) // 2 - 1 places, t half the matched letters out of order; 'jaro_winkler' jaro + l p
    (1 - jaro), l the common prefix's length up to prefix_cap, added only where jaro exceeds boost_threshold.

    conventions:
    - a, b: the two strings
    - metric: 'levenshtein' (a distance: 0 for equal strings), 'levenshtein_similarity', 'jaro' or 'jaro_winkler'
      (similarities: 1 for equal strings)
    - case: 'sensitive' or 'insensitive' (both lowercased first)
    - prefix_scale: p, from 0 to 1/max(prefix_cap, 1), Winkler's 0.1; prefix_cap: Winkler's 4; boost_threshold: 0.7, or 0.0
      for the bonus always; all three None unless 'jaro_winkler'
    returns: a float
    source: Wagner and Fischer (1974), J. ACM 21(1); Jaro (1989), JASA 84(406); Winkler (1994), Census RR94-05, Table 2.1
    aliases: Levenshtein distance, edit distance, Jaro-Winkler, Jaro similarity, fuzzy matching, string distance, [string similarity]
    example:
        # Returns 0.961111.
        qf.string_similarity(a="MARTHA", b="MARHTA", metric="jaro_winkler", case="sensitive", prefix_scale=0.1,
                             prefix_cap=4, boost_threshold=0.7)
    """
    choice("metric", metric, METRICS)
    choice("case", case, MATCH_CASES)
    if not (isinstance(a, str) and isinstance(b, str)):
        raise ValueError("a and b must be strings")
    if case == "insensitive":
        a, b = a.lower(), b.lower()
    winkler = metric == "jaro_winkler"
    if not winkler and (prefix_scale, prefix_cap, boost_threshold) != (None, None, None):
        raise ValueError("prefix_scale, prefix_cap and boost_threshold are None unless metric is 'jaro_winkler'")
    if metric in ("levenshtein", "levenshtein_similarity"):
        distance = _levenshtein(a, b)
        if metric == "levenshtein":
            return float(distance)
        return 1.0 if not (a or b) else 1.0 - distance / max(len(a), len(b))
    similarity = _jaro(a, b)
    if not winkler:
        return float(similarity)
    scale, cap = float(prefix_scale), integer("prefix_cap", prefix_cap, 0)
    if not 0.0 <= scale <= 1.0 / max(cap, 1):
        raise ValueError("prefix_scale must lie from 0 to 1 / prefix_cap (Winkler's 0.1 with 4)")
    prefix = 0
    for ca, cb in zip(a[:cap], b[:cap]):
        if ca != cb:
            break
        prefix += 1
    if similarity > float(boost_threshold):
        similarity += prefix * scale * (1.0 - similarity)
    return float(similarity)
