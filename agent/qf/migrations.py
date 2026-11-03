"""Rating migrations: cohort and duration (Lando-Skodeberg) estimates, generators, CreditMetrics.

`import qf`. A migration matrix is row-stochastic over the states the caller names, best first, an absorbing default
among them (last by convention); a generator's rows sum to 0. Times are in years. A function with several results
returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.special import ndtri

from ._common import Result, choice, positive
from .conventions import _finite
from .copulas import _draws

__all__ = ["cohort_transition_matrix", "creditmetrics", "generator_from_transition", "generator_mle"]

FIXES = ("none", "diagonal", "weighted")
GENERATOR_METHODS = ("duration", "aalen_johansen")


def cohort_transition_matrix(start, end, *, states, absorbing):
    """Cohort rating transition matrix P_ij = N_ij / N_i from start and end ratings.

    Each pair (start[k], end[k]) is one obligor over one period; pooled over periods the counts add. N_ij counts the
    pairs from i to j and N_i = sum_j N_ij. A state in `absorbing` (the default state, e.g. 'D') has its row set to
    staying put, whatever its counts; any other row with N_i = 0 is NaN. Every label must be in `states` (withdrawn
    ratings removed or listed first). Cumulative default probabilities over h periods are the default column of
    numpy.linalg.matrix_power(P, h).

    Whether two periods (or groups) migrate alike from a rating, the chi-squared test of homogeneity: their count
    rows for that rating stacked (one row per period, a column per end state), all-zero rows and columns dropped,
    into scipy.stats.chi2_contingency(table, correction=False), which returns the statistic, p-value and degrees of
    freedom. The duration estimate from dated histories is qf.generator_mle.
    conventions:
        states: the ordered labels, e.g. ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
        absorbing: the labels whose rows are set to the unit row, e.g. ["D"], or [] for none
        start, end: one label pair per obligor and period; from counts already tallied (a k x k N) the
            same estimate is N / N.sum(axis=1, keepdims=True), a row with no obligors set as the task says
    returns: (matrix, counts), k x k arrays in the order of `states`
    source: Jafry and Schuermann (2004), "Measurement, estimation and comparison of credit migration matrices",
        Journal of Banking and Finance 28(11), the cohort method; Hull, Risk Management and Financial Institutions,
        "Credit ratings migration"
    aliases: [transition matrix], migration matrix, cohort method, rating migration, credit migration
    example:
        import qf
        start, end = ["A", "A", "B", "B", "B", "A"], ["A", "B", "B", "D", "A", "A"]
        qf.cohort_transition_matrix(start, end, states=["A", "B", "D"], absorbing=["D"])
    """
    labels = list(states)
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels) or len(labels) < 2:
        raise ValueError("states must list at least two distinct labels")
    unknown = sorted({str(x) for x in absorbing if x not in index})
    if unknown:
        raise ValueError(f"absorbing names labels not in states: {unknown}")
    a, b = list(np.asarray(start).ravel()), list(np.asarray(end).ravel())
    if len(a) != len(b) or not a:
        raise ValueError("start and end must be equal-length, non-empty sequences of labels")
    unknown = sorted({str(x) for x in a + b if x not in index})
    if unknown:
        raise ValueError(f"start or end holds labels not in states: {unknown}")
    k = len(labels)
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, ([index[x] for x in a], [index[x] for x in b]), 1)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(totals > 0, counts / np.maximum(totals, 1), np.nan)
    for label in absorbing:
        matrix[index[label]] = np.eye(k)[index[label]]
    return matrix, counts


def generator_from_transition(matrix, *, dt, fix):
    """Continuous-time generator matrix Q = logm(P) / dt, negative rates fixed by a named rule.

    Q = logm(P) / dt, so expm(Q dt) = P. An empirical log often has negative off-diagonal entries; 'diagonal' sets
    them to 0 and adds them to the diagonal (Israel, Rosenthal and Wei's DA); 'weighted' sets them to 0 and removes
    their total B_i from the row's other non-zero entries in proportion to their size, q_ij - B_i |q_ij| / G_i, G_i =
    |q_ii| + sum_{j != i} max(q_ij, 0) (their WA). Rows then sum to 0 with non-negative off-diagonals; 'none' returns
    the log itself. A P with no real logarithm is refused.
    conventions:
        dt: the period of P in years (1 for an annual matrix)
        fix: 'none', 'diagonal' or 'weighted'
    returns: a k x k array
    source: Israel, Rosenthal and Wei (2001), "Finding generators for Markov chains via empirical transition
        matrices, with applications to credit ratings", Mathematical Finance 11(2); Kreinin and Sidelnikova (2001),
        Algo Research Quarterly 4
    aliases: generator matrix, transition intensity, matrix logarithm, logm, continuous-time migration, rate matrix
    example:
        import numpy as np, qf
        p = np.array([[0.90, 0.08, 0.02], [0.05, 0.85, 0.10], [0.0, 0.0, 1.0]])
        qf.generator_from_transition(p, dt=1.0, fix="diagonal")
    """
    choice("fix", fix, FIXES)
    p = _finite("matrix", matrix)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("matrix must be square and stochastic: non-negative rows summing to 1")
    step = float(positive("dt", dt))
    log = linalg.logm(p)
    if np.iscomplexobj(log):
        if np.max(np.abs(log.imag)) > 1e-10:
            raise ValueError("matrix has no real logarithm (an eigenvalue on the negative real axis)")
        log = log.real
    q = np.asarray(log, dtype=float) / step
    if fix == "none":
        return q
    off = ~np.eye(q.shape[0], dtype=bool)
    kept = np.where(off, np.maximum(q, 0.0), q)
    removed = np.sum(np.where(off, np.maximum(-q, 0.0), 0.0), axis=1)
    if fix == "diagonal":
        return kept - np.diag(removed)
    size = np.abs(np.diag(q)) + np.sum(np.where(off, np.maximum(q, 0.0), 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(size[:, None] > 0, np.abs(kept) / size[:, None], 0.0)
    return kept - removed[:, None] * share

def creditmetrics(*, transition, initial, values, rho, n, rng):
    """CreditMetrics: rating migrations simulated by asset-return thresholds, the bonds revalued in each state.

    Obligor i's asset return Z_i = sqrt(rho_i) M + sqrt(1 - rho_i) e_i (M the common factor) maps to an end state by
    thresholds from the worst: with its rating's transition row p_1 (best) ... p_D (default, last), it defaults where
    Z_i < Phi^-1(p_D), ends one state better where Z_i < Phi^-1(p_D + p_{D-1}), and so on up; each state keeps its
    probability (they sum to 1), and two obligors default together with Phi_2(z_i, z_j; sqrt(rho_i rho_j)). A
    scenario's portfolio value sums each obligor's value in its end state (its bond on that rating's forward curve;
    in default its recovery). Losses against a reference value (at the initial ratings, or the mean) go to
    qf.historical_var_es as -losses; qf.factor_copula_losses simulates defaults alone.
    conventions:
        transition: the migration matrix, a row per initial rating, columns best to worst, default last
            (probabilities, rows summing to 1)
        initial: each obligor's initial rating, a row index of transition
        values: obligors x end states, each obligor's value in each state (default: the recovery; face times (1 -
            LGD) where only defaults matter)
        rho: the asset correlation with the factor, one number or one per obligor, inside [0, 1)
        n: the number of scenarios
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); the factor M (n) first,
            then the e_i (n x obligors). A task's bare "seed s" may mean either stream: they draw different samples
    returns: a dict: thresholds (a row per rating, cut-offs from the worst state up, the last inf); states (n x
        obligors end-state indices); portfolio_values (n); mean and std of the portfolio value
    source: Gupton, Finger and Bhatia, CreditMetrics Technical Document (1997), ch. 8 and 10; Crouhy, Galai and Mark
        (2000), J. Banking and Finance 24
    aliases: CreditMetrics, CreditMetrics VaR, rating migration simulation, simulate rating migrations, migration thresholds, rating transition simulation
    example:
        import numpy as np, qf
        P, v = [[0.90, 0.08, 0.02], [0.05, 0.85, 0.10]], [[101.0, 99.0, 40.0], [101.0, 99.0, 40.0], [102.0, 98.0, 45.0]]
        qf.creditmetrics(transition=P, initial=[0, 1, 1], values=v, rho=0.2, n=10000, rng=np.random.default_rng(4))
    """
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("transition must be a matrix of probabilities (not percent), each row summing to 1")
    start = np.asarray(initial).ravel()
    if start.dtype.kind not in "iu" or np.any(start < 0) or np.any(start >= matrix.shape[0]):
        raise ValueError("initial must hold each obligor's row index of transition")
    worth = np.asarray(values, dtype=float)
    if worth.shape != (start.size, matrix.shape[1]):
        raise ValueError("values must be obligors x end states (the columns of transition)")
    corr = np.broadcast_to(np.asarray(rho, dtype=float), start.shape)
    if np.any(~((corr >= 0) & (corr < 1))):
        raise ValueError("rho must lie inside [0, 1)")
    scenarios, rng = _draws(n, rng)
    with np.errstate(divide="ignore"):
        thresholds = ndtri(np.clip(np.cumsum(matrix[:, ::-1], axis=1), 0.0, 1.0))
    thresholds[:, -1] = np.inf
    factor = rng.standard_normal(scenarios)
    noise = rng.standard_normal((scenarios, start.size))
    z = np.sqrt(corr) * factor[:, None] + np.sqrt(1.0 - corr) * noise
    worst_first = np.empty((scenarios, start.size), dtype=np.int16)
    for i in range(start.size):
        worst_first[:, i] = np.searchsorted(thresholds[start[i]], z[:, i], side="right")
    states = (matrix.shape[1] - 1 - worst_first).astype(np.int16)
    portfolio = np.zeros(scenarios)
    for i in range(start.size):
        portfolio += worth[i, states[:, i]]
    return Result({"thresholds": thresholds, "states": states, "portfolio_values": portfolio,
                   "mean": float(portfolio.mean()), "std": float(portfolio.std(ddof=1))})


def _histories(histories, index, absorbing, censored, end):
    """Each obligor's spells (state index or None for censored, start, stop) and its jumps (from, to, time)."""
    spells, jumps = [], []
    for history in histories:
        events = [(float(time), label) for time, label in history]
        if not events or any(b[0] <= a[0] for a, b in zip(events, events[1:])) or events[-1][0] > end:
            raise ValueError("each history must be (time, rating) pairs at increasing times, none after end")
        unknown = sorted({str(label) for _, label in events if label not in index and label not in censored})
        if unknown:
            raise ValueError(f"ratings in neither states nor censored: {unknown}")
        for (start, label), stop in zip(events, [time for time, _ in events[1:]] + [end]):
            if label in censored or label in absorbing:
                break
            spells.append((index[label], start, stop))
        for (_, a), (time, b) in zip(events, events[1:]):
            if a in censored or a in absorbing:
                break
            if b not in censored:
                jumps.append((index[a], index[b], time))
    return spells, jumps


def generator_mle(histories, *, states, absorbing, censored, end, method, horizon):
    """Rating generator by duration (Lando-Skodeberg): each rating's moves over its years observed; Aalen-Johansen too.

    Each obligor is watched from its first rating until `end`, a censoring label (a withdrawn rating) or an absorbing
    state. q_ij = N_ij / T_i for i != j, N_ij the moves from i to j and T_i the years all obligors spent in i, and
    q_ii = -sum_{j != i} q_ij: a rating seen to move only to a neighbour still gets a default rate where the
    neighbour defaults. The matrix over `horizon`: 'duration' expm(Q horizon); 'aalen_johansen' the product over the
    move times s in (start, start + horizon] of (I + dA(s)), dA_ij = the moves i -> j at s over the obligors in i
    just before s (dA_ii = -their sum), start the first observation (Kaplan-Meier's where default is the only move). The mean time to default from each other state is -Q_T^-1 1, Q_T the generator
    among them: numpy.linalg.solve(-Q[:-1, :-1], numpy.ones(k - 1)) with default last.
    conventions:
        histories: one per obligor, its (time, rating) pairs in time order, the first when it is first rated
        states: the ordered ratings, best first, default last, e.g. ["A", "B", "D"]
        absorbing: the ratings whose rows are 0 and where watching stops, e.g. ["D"], or [] for none
        censored: ratings that end the watch with no move counted (withdrawn, e.g. ["NR"]), or [] for none
        end: the time the observation window closes
        method: 'duration' or 'aalen_johansen' (how the matrix is made)
        horizon: the matrix's span in years (1 for an annual matrix)
    returns: a dict: generator (k x k); matrix (k x k over the horizon); exposure (T_i, years per state); counts
        (N_ij, k x k)
    source: Lando and Skodeberg (2002), "Analyzing rating transitions and rating drift with continuous
        observations", Journal of Banking and Finance 26; Lando, Credit Risk Modeling (2004), 4.3-4.4; Aalen and
        Johansen (1978), Scandinavian Journal of Statistics 5
    aliases: Lando-Skodeberg, duration-based generator, Aalen-Johansen, rating generator, continuous-time migration
    example:
        import qf
        h = [[(0.0, "A"), (1.5, "B")], [(0.0, "B"), (0.7, "D")], [(0.0, "A")], [(0.0, "B"), (1.2, "A")]]
        qf.generator_mle(histories=h, states=["A", "B", "D"], absorbing=["D"], censored=[], end=2.0, method="duration", horizon=1.0)
        # generator = [[-1/4.3, 1/4.3, 0], [1/2.4, -2/2.4, 1/2.4], [0, 0, 0]]
    """
    choice("method", method, GENERATOR_METHODS)
    labels = list(states)
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels) or len(labels) < 2:
        raise ValueError("states must list at least two distinct labels")
    if any(x not in index for x in absorbing) or any(x in index for x in censored):
        raise ValueError("absorbing labels must be states; censored labels must not be")
    close, span = float(end), float(positive("horizon", horizon))
    spells, jumps = _histories(histories, index, set(absorbing), set(censored), close)
    k = len(labels)
    exposure, counts = np.zeros(k), np.zeros((k, k), dtype=int)
    for state, start, stop in spells:
        exposure[state] += stop - start
    for a, b, _ in jumps:
        counts[a, b] += 1
    np.fill_diagonal(counts, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(exposure[:, None] > 0, counts / np.maximum(exposure, 1e-300)[:, None], np.nan)
    np.fill_diagonal(q, 0.0)
    q[np.isnan(q).any(axis=1)] = np.nan
    np.fill_diagonal(q, -np.nansum(q, axis=1))
    for label in absorbing:
        q[index[label]] = 0.0
    if method == "duration":
        matrix = linalg.expm(np.nan_to_num(q) * span)
        matrix[np.isnan(q).any(axis=1)] = np.nan
    else:
        start = min(begin for _, begin, _ in spells)
        matrix = np.eye(k)
        for when in sorted({time for _, _, time in jumps if start < time <= start + span}):
            at_risk = np.zeros(k)
            for state, begin, stop in spells:
                if begin < when <= stop:
                    at_risk[state] += 1
            step = np.eye(k)
            for a, b, time in jumps:
                if time == when and at_risk[a] > 0:
                    step[a, b] += 1.0 / at_risk[a]
                    step[a, a] -= 1.0 / at_risk[a]
            matrix = matrix @ step
    return Result({"generator": q, "matrix": matrix, "exposure": exposure, "counts": counts})
