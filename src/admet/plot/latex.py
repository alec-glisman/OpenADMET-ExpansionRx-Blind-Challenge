from __future__ import annotations

import logging
import re

from numpy import ndarray

from admet.data.stats import correlation, distribution

logger = logging.getLogger(__name__)


def latex_sanitize(text: str, use_math_mode: bool = False) -> str:
    """Sanitize text for LaTeX rendering in plots.

    Handles both Unicode and ASCII special characters that cause LaTeX errors.
    This function is idempotent - calling it multiple times on the same text
    produces the same result.

    Parameters
    ----------
    text : str
        Input text.
    use_math_mode : bool, default=False
        If True, use math mode for symbols.
        If False, convert Unicode to LaTeX-compatible format.

    Returns
    -------
    str
        Sanitized text safe for LaTeX rendering.
    """
    # Check if already sanitized (contains LaTeX math mode markers or escaped percent)
    # This makes the function idempotent and prevents double-sanitization
    if "$^{" in text or "$\\rightarrow$" in text or "\\%" in text or " pct " in text:
        return text

    # Replace percent sign - use simple text replacement to avoid LaTeX issues
    # Don't use backslash escaping as it causes line-breaking problems
    text = text.replace("% unbound", " pct unbound")
    text = text.replace("%", " pct")

    # Handle arrows
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace(">", r"$>$")
    text = text.replace("<", r"$<$")

    # Handle Unicode superscripts
    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }

    # Replace sequences of Unicode superscripts
    i = 0
    result = []
    while i < len(text):
        if text[i] in superscript_map:
            # Start of superscript sequence
            superscript_text = ""
            j = i
            while j < len(text) and text[j] in superscript_map:
                superscript_text += superscript_map[text[j]]
                j += 1
            result.append(f"$^{{{superscript_text}}}$")
            i = j
        else:
            result.append(text[i])
            i += 1

    text = "".join(result)

    # Handle ASCII caret notation (e.g., "10^-6" -> "10$^{-6}$")
    text = re.sub(r"(\d+)\^(-?\d+)", r"\1$^{\2}$", text)

    return text


def text_distribution(array: ndarray) -> str:
    """Generate a LaTeX-formatted string summarizing distribution statistics.

    Parameters
    ----------
    array : numpy.ndarray
        Numeric array.

    Returns
    -------
    str
        LaTeX-formatted summary string.
    """
    stats = distribution(array)
    summary = (
        f"Min: {stats['min']:.2f}\n"
        f"Max: {stats['max']:.2f}\n"
        f"Mean: {stats['mean']:.2f}\n"
        f"Median: {stats['median']:.2f}\n"
        f"Std: {stats['std']:.2f}\n"
        f"Skew: {stats['skew']:.2f}\n"
        f"Kurtosis: {stats['kurtosis']:.2f}\n"
        f"$N$: {stats['count']}"
    )
    return summary


def text_correlation(y_true: ndarray, y_pred: ndarray) -> str:
    """Generate a LaTeX-formatted string summarizing correlation metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        True values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    str
        LaTeX-formatted summary string.
    """
    metrics = correlation(y_true, y_pred)
    summary = (
        f"MAE: {metrics['mae']:.2f}\n"
        f"RAE: {metrics['rae']:.2f}\n"
        f"MAPE: {metrics['mape']:.2f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"$R^2$: {metrics['R2']:.2f}\n"
        f"Pearson $r$: {metrics['pearson_r']:.2f}\n"
        f"Spearman $\\rho$: {metrics['spearman_rho']:.2f}"
    )
    return summary
