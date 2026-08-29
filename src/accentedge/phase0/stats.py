"""
Statistical analysis for Gate 1B listening study.

Provides:
- Inter-rater reliability (Cohen's kappa for categorical, ICC for continuous)
- Signal detection metrics (d-prime)
- Bootstrap confidence intervals
- Result summarization by condition
- Per-speaker reference distributions
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional scipy import
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    rater1_labels: list[int],
    rater2_labels: list[int],
    weights: Optional[str] = None,
) -> float:
    """
    Compute Cohen's kappa for two raters on categorical labels.

    Args:
        rater1_labels: Integer labels from rater 1.
        rater2_labels: Integer labels from rater 2.
        weights: None for unweighted, 'linear' for linearly weighted kappa.

    Returns:
        Cohen's kappa value.
    """
    if len(rater1_labels) != len(rater2_labels):
        raise ValueError("Rater label lists must have the same length.")
    if len(rater1_labels) == 0:
        return 0.0

    labels = np.array(rater1_labels)
    labels2 = np.array(rater2_labels)

    if _HAS_SCIPY:
        table = _build_contingency_table(labels, labels2)
        po = np.trace(table) / len(labels)
        n = len(labels)
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        pe = np.dot(row_sums, col_sums) / (n * n)
        if pe >= 1.0:
            return 1.0
        return float((po - pe) / (1.0 - pe))

    # Fallback: manual implementation
    all_labels = np.unique(np.concatenate([labels, labels2]))
    n_categories = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    confusion = np.zeros((n_categories, n_categories), dtype=float)
    for l1, l2 in zip(labels, labels2):
        confusion[label_to_idx[l1], label_to_idx[l2]] += 1

    n = len(labels)
    po = np.trace(confusion) / n
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = np.dot(row_sums, col_sums) / (n * n)

    if pe >= 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return float(np.clip(kappa, -1.0, 1.0))


def _build_contingency_table(labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
    """Build a contingency table from two label arrays."""
    all_labels = np.unique(np.concatenate([labels1, labels2]))
    n_categories = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    table = np.zeros((n_categories, n_categories), dtype=float)
    for l1, l2 in zip(labels1, labels2):
        table[label_to_idx[l1], label_to_idx[l2]] += 1
    return table


# ---------------------------------------------------------------------------
# Inter-rater reliability
# ---------------------------------------------------------------------------

def compute_inter_rater_reliability(
    ratings: dict[int, list[float]],
    method: str = "icc",
) -> dict[str, float]:
    """
    Compute inter-rater reliability across multiple raters.

    Args:
        ratings: Mapping of rater_id -> list of ratings (one per stimulus).
        method: 'icc' for intraclass correlation, 'cohen' for pairwise Cohen's kappa.

    Returns:
        Dictionary with reliability metrics.
    """
    if not ratings:
        return {"icc": 0.0, "cohens_kappa_mean": 0.0}

    rater_ids = sorted(ratings.keys())
    if len(rater_ids) < 2:
        return {"icc": 0.0, "cohens_kappa_mean": 0.0}

    # Build rating matrix: raters x stimuli
    matrix = np.array([ratings[rid] for rid in rater_ids])

    if method == "icc":
        n_r = matrix.shape[0]
        n_s = matrix.shape[1]

        if n_s < 2:
            return {"icc": 1.0}

        grand_mean = matrix.mean()
        mean_raters = matrix.mean(axis=1, keepdims=True)
        mean_stimuli = matrix.mean(axis=0, keepdims=True)

        ss_between_stimuli = n_r * np.sum((mean_stimuli - grand_mean) ** 2)
        ss_residual = np.sum(
            (matrix - mean_raters - mean_stimuli + grand_mean) ** 2
        )

        ms_stimuli = ss_between_stimuli / (n_s - 1)
        ms_residual = ss_residual / max((n_r - 1) * (n_s - 1), 1)

        if ms_residual == 0:
            icc = 1.0
        else:
            icc = (ms_stimuli - ms_residual) / (
                ms_stimuli + (n_r - 1) * ms_residual
            )

        return {"icc": float(np.clip(icc, -1.0, 1.0))}

    # Fallback: mean pairwise correlation
    if matrix.shape[1] >= 2:
        correlations = []
        for i in range(len(rater_ids)):
            for j in range(i + 1, len(rater_ids)):
                if np.std(matrix[i]) > 0 and np.std(matrix[j]) > 0:
                    corr = np.corrcoef(matrix[i], matrix[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        if correlations:
            return {"icc": float(np.mean(correlations))}

    return {"icc": 0.0}


# ---------------------------------------------------------------------------
# Signal Detection Theory
# ---------------------------------------------------------------------------

def _approx_norm_ppf(p: float) -> float:
    """Rational approximation for the standard normal CDF inverse."""
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3, d4 = 1.432788, 0.189269, 0.001308, 0.000027
    return t - (c0 + c1 * t + c2 * t ** 2) / (
        1 + d1 * t + d2 * t ** 2 + d3 * t ** 3 + d4 * t ** 4
    )


def compute_dprime(
    hit_rate: float,
    false_alarm_rate: float,
    half_hit: float = 0.01,
    half_fa: float = 0.01,
) -> float:
    """
    Compute d-prime (d') from hit rate and false alarm rate.

    d' = Z(hit_rate) - Z(false_alarm_rate)

    Args:
        hit_rate: Proportion of "same person" trials correctly identified.
        false_alarm_rate: Proportion of "different person" trials
            incorrectly identified as same.
        half_hit: Correction value when hit_rate is 0 or 1.
        half_fa: Correction value when false_alarm_rate is 0 or 1.

    Returns:
        d-prime value.
    """
    hr = max(half_hit, min(1.0 - half_hit, hit_rate))
    fa = max(half_fa, min(1.0 - half_fa, false_alarm_rate))

    if _HAS_SCIPY:
        d_prime = scipy_stats.norm.ppf(hr) - scipy_stats.norm.ppf(fa)
    else:
        d_prime = _approx_norm_ppf(hr) - _approx_norm_ppf(fa)

    return float(d_prime)


def compute_dprime_from_trials(
    same_responses: list[int],
    different_responses: list[int],
) -> dict[str, Any]:
    """
    Compute d-prime from raw trial responses.

    Args:
        same_responses: List of 1-5 ratings for SAME-person trials.
        different_responses: List of 1-5 ratings for DIFFERENT-person trials.

    Returns:
        Dict with 'd_prime', 'hit_rate', 'false_alarm_rate', and 'criterion'.
    """
    # Threshold: rating >= 4 means "yes, same person"
    threshold = 4

    if same_responses:
        hit_rate = sum(1 for r in same_responses if r >= threshold) / len(same_responses)
    else:
        hit_rate = 0.5

    if different_responses:
        fa_rate = sum(1 for r in different_responses if r >= threshold) / len(different_responses)
    else:
        fa_rate = 0.5

    d_prime = compute_dprime(hit_rate, fa_rate)

    # Criterion C = -0.5 * (Z(HR) + Z(FA))
    if _HAS_SCIPY:
        criterion = -0.5 * (
            scipy_stats.norm.ppf(max(0.01, min(0.99, hit_rate)))
            + scipy_stats.norm.ppf(max(0.01, min(0.99, fa_rate)))
        )
    else:
        criterion = -0.5 * (
            _approx_norm_ppf(max(0.01, min(0.99, hit_rate)))
            + _approx_norm_ppf(max(0.01, min(0.99, fa_rate)))
        )

    return {
        "d_prime": d_prime,
        "hit_rate": hit_rate,
        "false_alarm_rate": fa_rate,
        "criterion": float(criterion),
        "n_same_trials": len(same_responses),
        "n_different_trials": len(different_responses),
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list[float],
    statistic: str | Callable = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Sample data points.
        statistic: 'mean', 'median', or callable that takes a 1-D array.
        n_bootstrap: Number of bootstrap samples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: Optional seed for reproducibility.

    Returns:
        Dict with 'statistic', 'ci_lower', 'ci_upper', 'ci_level', 'n_bootstrap'.
    """
    if not data:
        raise ValueError("Cannot compute bootstrap CI on empty data.")

    arr = np.array(data, dtype=float)
    n = len(arr)

    if statistic == "mean":
        func = np.mean
    elif statistic == "median":
        func = np.median
    elif callable(statistic):
        func = statistic
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    rng = np.random.RandomState(random_seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_stats[i] = func(sample)

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_stats, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    point_estimate = float(func(arr))

    return {
        "statistic": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Result Summarization
# ---------------------------------------------------------------------------

@dataclass
class ConditionSummary:
    """Aggregated ratings for one condition."""
    condition: str
    n_trials: int
    same_person_mean: float
    same_person_ci_lower: float
    same_person_ci_upper: float
    accent_shift_mean: float
    accent_shift_ci_lower: float
    accent_shift_ci_upper: float
    naturalness_mean: float
    naturalness_ci_lower: float
    naturalness_ci_upper: float
    content_preserved_rate: float
    same_person_std: float = 0.0
    accent_shift_std: float = 0.0
    naturalness_std: float = 0.0


def summarize_results(
    trials: list[Any],  # list of ListeningTrial objects
    conditions: Optional[list[str]] = None,
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> list[ConditionSummary]:
    """
    Aggregate listening trial ratings by condition.

    Args:
        trials: List of completed ListeningTrial objects.
        conditions: List of conditions to include. If None, include all found.
        ci_level: Confidence level for bootstrap CIs.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        List of ConditionSummary, one per condition.
    """
    if not trials:
        return []

    # Group trials by condition
    by_condition: dict[str, list[dict]] = {}
    for trial in trials:
        cond = trial.stimulus.condition
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append({
            "same_person": trial.same_person,
            "accent_shift": trial.accent_shift,
            "naturalness": trial.naturalness,
            "content_preserved": trial.content_preserved,
        })

    if conditions is None:
        conditions = sorted(by_condition.keys())

    summaries = []
    for cond in conditions:
        if cond not in by_condition or not by_condition[cond]:
            continue

        records = by_condition[cond]
        sp_vals = [r["same_person"] for r in records if r["same_person"] is not None]
        as_vals = [r["accent_shift"] for r in records if r["accent_shift"] is not None]
        n_vals = [r["naturalness"] for r in records if r["naturalness"] is not None]
        cp_vals = [r["content_preserved"] for r in records if r["content_preserved"] is not None]

        def _summarize(vals):
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            m = float(np.mean(vals))
            s = float(np.std(vals))
            ci = bootstrap_ci(vals, ci_level=ci_level, n_bootstrap=min(n_bootstrap, 5000))
            return m, s, ci["ci_lower"], ci["ci_upper"]

        sp_m, sp_s, sp_lo, sp_hi = _summarize(sp_vals)
        as_m, as_s, as_lo, as_hi = _summarize(as_vals)
        n_m, n_s, n_lo, n_hi = _summarize(n_vals)

        cp_rate = 0.0
        if cp_vals:
            cp_rate = sum(1 for v in cp_vals if v == "yes") / len(cp_vals)

        summaries.append(ConditionSummary(
            condition=cond,
            n_trials=len(records),
            same_person_mean=sp_m,
            same_person_ci_lower=sp_lo,
            same_person_ci_upper=sp_hi,
            accent_shift_mean=as_m,
            accent_shift_ci_lower=as_lo,
            accent_shift_ci_upper=as_hi,
            naturalness_mean=n_m,
            naturalness_ci_lower=n_lo,
            naturalness_ci_upper=n_hi,
            content_preserved_rate=cp_rate,
            same_person_std=sp_s,
            accent_shift_std=as_s,
            naturalness_std=n_s,
        ))

    return summaries


def compute_effect_size(
    treatment_values: list[float],
    control_values: list[float],
) -> dict[str, Any]:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        treatment_values: Ratings for the treatment group.
        control_values: Ratings for the control group.

    Returns:
        Dict with 'cohens_d', 'variance', and interpretation.
    """
    if not treatment_values or not control_values:
        return {"cohens_d": 0.0, "interpretation": "insufficient_data"}

    t_mean = float(np.mean(treatment_values))
    c_mean = float(np.mean(control_values))
    t_std = float(np.std(treatment_values, ddof=1))
    c_std = float(np.std(control_values, ddof=1))

    # Pooled standard deviation
    n1, n2 = len(treatment_values), len(control_values)
    pooled_std = np.sqrt(
        ((n1 - 1) * t_std ** 2 + (n2 - 1) * c_std ** 2) / max(n1 + n2 - 2, 1)
    )

    if pooled_std == 0:
        d = 0.0 if t_mean == c_mean else (999.0 if t_mean > c_mean else -999.0)
    else:
        d = (t_mean - c_mean) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {
        "cohens_d": float(d),
        "pooled_std": float(pooled_std),
        "treatment_mean": t_mean,
        "control_mean": c_mean,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Per-Speaker Reference Distributions
# ---------------------------------------------------------------------------

@dataclass
class SpeakerReference:
    """Reference distribution for one speaker's identity/timing metrics."""
    speaker_id: str
    n_samples: int
    identity_mean: float
    identity_std: float
    timing_mean: float
    timing_std: float
    naturalness_mean: float
    naturalness_std: float
    raw_identity_scores: list[float] = field(default_factory=list)
    raw_timing_scores: list[float] = field(default_factory=list)
    raw_naturalness_scores: list[float] = field(default_factory=list)


def compute_reference_distribution_stats(
    speaker_trials: list[dict[str, Any]],
) -> SpeakerReference:
    """
    Compute per-speaker reference distribution from a list of trial records.

    Each trial record should have:
    - 'speaker_id': str
    - 'identity_score': float (1-5)
    - 'timing_score': float (1-5, optional)
    - 'naturalness_score': float (1-5)

    Args:
        speaker_trials: List of trial dicts for one speaker.

    Returns:
        SpeakerReference with distribution statistics.
    """
    if not speaker_trials:
        raise ValueError("Cannot compute reference for empty speaker trials.")

    speaker_id = speaker_trials[0].get("speaker_id", "unknown")
    identity_scores = [t["identity_score"] for t in speaker_trials if t.get("identity_score") is not None]
    naturalness_scores = [t["naturalness_score"] for t in speaker_trials if t.get("naturalness_score") is not None]
    timing_scores = [t.get("timing_score") for t in speaker_trials if t.get("timing_score") is not None]
    timing_scores = [s for s in timing_scores if s is not None]

    return SpeakerReference(
        speaker_id=speaker_id,
        n_samples=len(speaker_trials),
        identity_mean=float(np.mean(identity_scores)) if identity_scores else 0.0,
        identity_std=float(np.std(identity_scores)) if identity_scores else 0.0,
        timing_mean=float(np.mean(timing_scores)) if timing_scores else 0.0,
        timing_std=float(np.std(timing_scores)) if timing_scores else 0.0,
        naturalness_mean=float(np.mean(naturalness_scores)) if naturalness_scores else 0.0,
        naturalness_std=float(np.std(naturalness_scores)) if naturalness_scores else 0.0,
        raw_identity_scores=identity_scores,
        raw_timing_scores=timing_scores,
        raw_naturalness_scores=naturalness_scores,
    )
