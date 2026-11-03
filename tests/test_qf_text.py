"""agent/qf/text.py (lane lib-signals1; docs/research/library-syllabus-1011.md rows 75 and 123): TF-IDF weights against
scikit-learn's TfidfVectorizer under three settings, frozen (scikit-learn 1.9.1, this file's three documents); the
cosine's identities; the textbook idf in base 10 written out (Manning, Raghavan and Schutze ch. 6); Levenshtein's
classic pairs (kitten and sitting, three edits; Wagner and Fischer) and Jaro and Jaro-Winkler on Winkler's pairs
(MARTHA and MARHTA 0.944 and 0.961, DWAYNE and DUANE 0.822 and 0.840, DIXON and DICKSONX 0.767 and 0.813). No unit's
    text or values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf

pytestmark = pytest.mark.fast

DOCS = ["The Fed raised rates; rates rose again.", "Inflation eased and the Fed held rates.",
        "Growth slowed, inflation eased, inflation fell."]
VOCABULARY = ["again", "and", "eased", "fed", "fell", "growth", "held", "inflation", "raised", "rates", "rose", "slowed",
              "the"]
SKLEARN = {   # TfidfVectorizer(**settings).fit_transform(DOCS).toarray(), scikit-learn 1.9.1, rounded to 15 places
    "default": [[0.393128514142393, 0.0, 0.0, 0.298984368070914, 0.0, 0.0, 0.0, 0.0, 0.393128514142393,
                 0.597968736141828, 0.393128514142393, 0.0, 0.298984368070914],
                [0.0, 0.452123308263386, 0.343851429617485, 0.343851429617485, 0.0, 0.0, 0.452123308263386,
                 0.343851429617485, 0.0, 0.343851429617485, 0.0, 0.0, 0.343851429617485],
                [0.0, 0.0, 0.313316068889206, 0.0, 0.41197297843389, 0.41197297843389, 0.0, 0.626632137778412, 0.0,
                 0.0, 0.0, 0.41197297843389, 0.0]],
    "sublinear_tf=True, smooth_idf=False, norm='l1'": [
        [0.182703673148177, 0.0, 0.0, 0.122358779236888, 0.0, 0.0, 0.0, 0.0, 0.182703673148177, 0.207171422081693,
         0.182703673148177, 0.0, 0.122358779236888],
        [0.0, 0.186966271844223, 0.125213491262311, 0.125213491262311, 0.0, 0.0, 0.186966271844223,
         0.125213491262311, 0.0, 0.125213491262311, 0.0, 0.0, 0.125213491262311],
        [0.0, 0.0, 0.139417767012466, 0.0, 0.208175811283471, 0.208175811283471, 0.0, 0.23605479913712, 0.0, 0.0,
         0.0, 0.208175811283471, 0.0]],
}
SKLEARN_COSINE = [[1.0, 0.411224809577857, 0.0], [0.411224809577857, 1.0, 0.323202534629052],
                  [0.0, 0.323202534629052, 1.0]]   # sklearn.metrics.pairwise.cosine_similarity of the default


def tfidf(documents=DOCS, **changes):
    args = dict(tokens="sklearn", case="lower", stop_words=None, tf="raw", idf="smooth", log_base="e", norm="l2")
    return qf.tfidf_cosine(documents=documents, **{**args, **changes})


def test_the_weights_and_cosines_are_scikit_learns_under_its_default_and_sublinear_settings():
    found = tfidf()
    assert found["vocabulary"] == VOCABULARY
    assert found["weights"] == pytest.approx(np.array(SKLEARN["default"]), abs=1e-14)
    assert found["similarity"] == pytest.approx(np.array(SKLEARN_COSINE), abs=1e-14)
    other = tfidf(tf="sublinear", idf="plain", norm="l1")
    assert other["weights"] == pytest.approx(np.array(SKLEARN["sublinear_tf=True, smooth_idf=False, norm='l1'"]),
                                             abs=1e-14)


def test_identical_texts_give_1_disjoint_ones_0_and_the_norm_never_moves_the_cosine():
    found = tfidf(["rates rise", "rates rise", "growth slows"])
    assert found["similarity"] == pytest.approx(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), abs=1e-15)
    for norm in ("l1", "none"):
        assert tfidf(norm=norm)["similarity"] == pytest.approx(tfidf()["similarity"], abs=1e-14)
    assert tfidf(["", "rates"])["similarity"][0, 1] == 0.0


def test_the_textbook_idf_in_base_10_and_the_term_weights_written_out():
    """Manning, Raghavan and Schutze (2008), 6.2: idf = log10(N / df); a term in every document weighs 0. 'frequency'
    divides each count by the document's terms; 'binary' marks presence."""
    found = tfidf(tf="raw", idf="standard", log_base=10, norm="none")
    counts = {"rates": [2, 1, 0], "inflation": [0, 1, 2], "the": [1, 1, 0]}
    for word, row in counts.items():
        df = sum(c > 0 for c in row)
        assert found["weights"][:, VOCABULARY.index(word)] == pytest.approx([c * math.log10(3 / df) for c in row])
    shares = tfidf(tf="frequency", idf="none", norm="none")["weights"]
    assert shares[0, VOCABULARY.index("rates")] == pytest.approx(2 / 7) and shares.sum(axis=1) == pytest.approx(1.0)
    assert set(tfidf(tf="binary", idf="none", norm="none")["weights"].ravel()) == {0.0, 1.0}


def test_stop_words_case_and_the_token_rules():
    assert "the" not in tfidf(stop_words=["The", "and"])["vocabulary"]
    assert "Fed" in tfidf(case="keep")["vocabulary"]
    assert "a" in tfidf(["a b rate"], tokens="alphanumeric", idf="none")["vocabulary"]
    assert tfidf(["a b rate"], idf="none")["vocabulary"] == ["rate"]          # scikit-learn's: two letters or more
    assert tfidf(["q3 rate 2024"], tokens="letters", idf="none")["vocabulary"] == ["q", "rate"]
    with pytest.raises(ValueError, match="list of strings"):
        tfidf("one text")


# ------------------------------------------------------------------------------------------------ strings


def similar(a, b, metric, **changes):
    winkler = dict(prefix_scale=0.1, prefix_cap=4, boost_threshold=0.7) if metric == "jaro_winkler" else \
        dict(prefix_scale=None, prefix_cap=None, boost_threshold=None)
    return qf.string_similarity(a=a, b=b, metric=metric, **{"case": "sensitive", **winkler, **changes})


def test_levenshtein_on_the_classic_pairs_and_its_symmetry():
    assert similar("kitten", "sitting", "levenshtein") == 3.0
    assert similar("flaw", "lawn", "levenshtein") == 2.0
    assert similar("", "abc", "levenshtein") == 3.0 and similar("abc", "abc", "levenshtein") == 0.0
    assert similar("sitting", "kitten", "levenshtein") == 3.0
    assert similar("kitten", "sitting", "levenshtein_similarity") == pytest.approx(1 - 3 / 7)


@pytest.mark.parametrize("a, b, jaro, winkler", [("MARTHA", "MARHTA", 0.944, 0.961), ("DWAYNE", "DUANE", 0.822, 0.840),
                                                 ("DIXON", "DICKSONX", 0.767, 0.813)])
def test_jaro_and_jaro_winkler_on_winklers_pairs(a, b, jaro, winkler):
    """Winkler (1994), Census RR94-05, Table 2.1 for MARTHA and DWAYNE; DIXON from the matching definition."""
    assert similar(a, b, "jaro") == pytest.approx(jaro, abs=5e-4)
    assert similar(a, b, "jaro_winkler") == pytest.approx(winkler, abs=5e-4)
    assert similar(b, a, "jaro") == pytest.approx(similar(a, b, "jaro"), abs=1e-15)


def test_the_prefix_bonus_its_cap_its_threshold_and_case():
    """Winkler (1994), Census RR94-05, Table 2.1: MARTHA/MARHTA and DWAYNE/DUANE; DIXON/DICKSONX from
    the Jaro definition. MARTHA/MARHTA has Jaro 17/18 and a common prefix MAR of three letters."""
    assert similar("MARTHA", "MARHTA", "jaro") == pytest.approx(17 / 18, abs=1e-15)
    assert similar("MARTHA", "MARHTA", "jaro_winkler") == pytest.approx(17 / 18 + 3 * 0.1 / 18, abs=1e-15)
    assert similar("MARTHA", "MARHTA", "jaro_winkler", prefix_cap=2) == pytest.approx(17 / 18 + 2 * 0.1 / 18)
    low = similar("ABCDEFGH", "ABZZZZZZ", "jaro")
    assert low < 0.7 and similar("ABCDEFGH", "ABZZZZZZ", "jaro_winkler") == low
    assert similar("ABCDEFGH", "ABZZZZZZ", "jaro_winkler", boost_threshold=0.0) > low
    assert similar("martha", "MARHTA", "jaro", case="insensitive") == pytest.approx(17 / 18)
    assert similar("AB", "BA", "jaro") == 0.0          # the matching window, max(2, 2) // 2 - 1, is 0 places
    assert similar("ABC", "BCA", "jaro") == 0.0
    with pytest.raises(ValueError, match="None unless"):
        similar("a", "b", "jaro", prefix_scale=0.1)
