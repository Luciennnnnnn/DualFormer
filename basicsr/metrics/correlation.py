import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def plcc(pred, mos, **kwargs):
    """
    Pearson linear correlation coefficient (PLCC).
    :param pred:
        Vector 1, a list or an array, with n values.
    :param mos:
        Vector 2, a list or an array, with n values.
    :return:
        The PLCC of 2 input vectors (pred and y).
    """
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(mos, list):
        mos = np.array(mos)

    assert len(pred.shape) == 1 and len(mos.shape) == 1, "Please input a vector with only one dimension."
    assert pred.shape == mos.shape, "The lengths of 2 input vectors are not equal."

    pred = pred - np.average(pred)
    mos = mos - np.average(mos)
    numerator = np.dot(pred, mos)
    denominator = np.sqrt(np.sum(pred ** 2)) * np.sqrt(np.sum(mos ** 2))
    ours = numerator / denominator
    scis = pearsonr(pred, mos)[0]  # Use scipy to calculate PLCC again.

    if abs(ours - scis) > 1e-8:
        print(f"Warning: Our PLCC = {ours:.15f}, scipy PLCC = {scis:.15f}, please check the results!")

    return scis


@METRIC_REGISTRY.register()
def fitted_plcc(pred, mos, **kwargs):
    """
    Calculate PLCC with a nonlinear regression using third-order poly fitting.
    It follows the setting of PIPAL:
        Gu et al. PIPAL: A large-scale image quality assessment dataset for perceptual image restoration. In ECCV 2020.
        https://www.jasongt.com/projectpages/pipal.html.
        NTIRE 2021 Perceptual Image Quality Assessment Challenge.
    """
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(mos, list):
        mos = np.array(mos)

    f = np.polyfit(pred, mos, deg=3)
    fitted_mos = np.polyval(f, pred)

    return plcc(fitted_mos, mos)


@METRIC_REGISTRY.register()
def srcc(pred, mos, **kwargs):
    """
    Spearman rank order correlation coefficient (SRCC).
    :param pred:
        Vector 1, a list or an array, with n values.
    :param mos:
        Vector 2, a list or an array, with n values.
    :return:
        The SRCC of 2 input vectors (pred and y).
    """
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(mos, list):
        mos = np.array(mos)

    assert len(pred.shape) == 1 and len(mos.shape) == 1, "Please input a vector with only one dimension."
    assert pred.shape == mos.shape, "The lengths of 2 input vectors are not equal."

    rank_pred = pred.argsort().argsort()
    rank_mos = mos.argsort().argsort()

    ours = plcc(rank_pred, rank_mos)
    scis = spearmanr(pred, mos)[0]  # Use scipy to calculate PLCC again.

    if abs(ours - scis) > 1e-5:
        print(f"Warning: Our SRCC = {ours:.15f}, scipy SRCC = {scis:.15f}, please check the results!")

    return scis


@METRIC_REGISTRY.register()
def krcc(pred, mos, **kwargs):
    """
    Kendall rank order correlation coefficient (KRCC).
    :param pred:
        Vector 1, a list or an array, with n values.
    :param mos:
        Vector 2, a list or an array, with n values.
    :return:
        The KRCC of 2 input vectors (pred and y).
    """
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(mos, list):
        mos = np.array(mos)

    assert len(pred.shape) == 1 and len(mos.shape) == 1, "Please input a vector with only one dimension."
    assert pred.shape == mos.shape, "The lengths of 2 input vectors are not equal."

    return kendalltau(pred, mos)[0]


if __name__ == "__main__":
    pass