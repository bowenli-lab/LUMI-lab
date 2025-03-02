import numpy as np
from scipy.stats import rankdata


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def relaxed_spearman_correlation(preds, targets, relax_ratio=0.05):
    """
    Compute the relaxed spearman correlation. The relax ratio tells the amount of difference allowed. A delta threshold will be computed by the relax ratio times the dynamica range of the target values. For each pair of values, if the difference is smaller than the delta threshold, will make the difference to be zero.

    Args:
        preds (np.ndarray): The predicted values.
        target (np.ndarray): The target values.
        relax_ratio (float): The relax ratio.
    """

    assert len(preds) == len(
        targets
    ), "The length of preds and target should be the same."
    n = len(preds)

    # Use rankdata to correctly handle ties
    x_rank = rankdata(preds)
    y_rank = rankdata(targets)
    delta = relax_ratio * n

    # Calculate the difference in ranks
    d = x_rank - y_rank
    d = np.where(np.abs(d) <= delta, 0, d)

    # Calculate the sum of the squared differences
    d_squared_sum = np.sum(d**2)

    # Calculate the Spearman correlation coefficient
    correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return correlation
