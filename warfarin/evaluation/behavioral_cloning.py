import torch

import numpy as np

import scipy as sp
import scipy.stats

import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, classification_report

from warfarin.evaluation.plotting import plot_policy_heatmap


def coverage(prob, y, thresh=0.3):
    """
    Computes the fraction of instances where the observed option is included in
    the feasible set at a given BCQ threshold.

    Args:
        prob: The probability outputs of the model.
        y: The observed option.
        thresh: The BCQ threshold.

    Returns:
        coverage: The fraction of instances where the observed option is
                  included in the feasible set.
    """
    covered = (prob.T / prob.max(axis=1)).T > thresh
    transition_idx = np.arange(prob.shape[0]).astype(int)
    coverage = covered[transition_idx, y].sum() / prob.shape[0]

    return coverage


def multi_output_f1_score(prob, y, thresh=0.3):
    """
    Computes the multi-output F1 score at a given BCQ threshold.

    Args:
        prob: The probability outputs of the model.
        y: The observed option.
        thresh: The BCQ threshold.

    Returns:
        f1: The multi-output F1 score.
    """
    covered = (prob.T / prob.max(axis=1)).T > thresh
    f1 = classification_report(
        np.array(pd.get_dummies(pd.Categorical(y))),
        (prob.T / prob.max(axis=1)).T > thresh,
        output_dict=True
    )["macro avg"]["f1-score"]

    return f1


def consistency(prob, thresh=0.3):
    """
    Computes the fraction of instances where the feasible set is consistent.

    Args:
        prob: The probability outputs of the model.
        thresh: The BCQ threshold.

    Returns:
        consistency: The fraction of instances where the feasible options are
                     contiguous in ordinal option space.
    """
    covered = ((prob.T / prob.max(axis=1)).T > thresh).astype(float)
    D = np.diff(covered, axis=-1)

    # Impute leading zeros
    idxs = (D != 0.).argmax(axis=-1)
    for n, idx in enumerate(idxs):
        D[n, :idx] = 2.
    # Impute trailing zeros
    idxs = D.shape[1] - (np.flip(D, axis=-1) != 0.).argmax(axis=-1)
    for n, idx in enumerate(idxs):
        D[n, idx:] = -2.

    c = np.sign(-D)
    c_all = np.all(np.sort(c, axis=-1) == c, axis=-1)
    c_single = np.sum(c != 0, axis=-1) == 1

    consistency = (c_all | c_single).sum() / prob.shape[0]

    return consistency


def calibration_error(yprob, y, binwidth=0.1):
    """
    Returns the expected calibration error of the model, as defined in [1].

    [1] Guo et al., On Calibration of Modern Neural Networks.
        https://arxiv.org/pdf/1706.04599.pdf

    Args:
        yprob: The probability outputs of the model.
        y: The observed option.
        binwidth: The width of the bins to cosnider. Empirically, in our case it
                  is not particularly sensitive to this parameter.

    Returns:
        ece: The expected calibration error.
    """
    _df = pd.DataFrame(yprob)
    _df["y"] = y
    _df = _df.melt(id_vars=["y"])
    _df.columns = ["y", "ypred", "yprob"]

    _df["correct"] = (_df["y"] == _df["ypred"])

    _df["yprob_bin"] = pd.cut(_df["yprob"],
                              np.arange(0., 1. + binwidth, binwidth).round(decimals=2),
                              labels=np.arange(0., 1., binwidth) + binwidth / 2.)

    comp_df = _df.groupby(["ypred", "yprob_bin"])["correct"].mean().to_frame()
    comp_df["n"] = _df.groupby(["ypred", "yprob_bin"])["correct"].count()
    comp_df = comp_df.reset_index()
    comp_df["ypred"] = pd.Categorical(comp_df["ypred"], ordered=True)
    comp_df["yprob_bin"] = np.array(comp_df["yprob_bin"])

    ece = (
        np.abs(comp_df["yprob_bin"] -
               comp_df["correct"]) * comp_df["n"] / comp_df["n"].sum()
    ).sum()

    return ece


def evaluate_behavioral_cloning(model, data):
    """
    Evaluate the behavioral cloning network.

    Args:
        model: The BehaviorCloner model.
        data: The dataloader to get batches from.

    Returns:
        metrics: A dictionary of metrics.
    """
    # Get model predictions and labels
    obs_state = np.array(data.observed_state).astype(np.float32)
    obs_option = np.array(data.observed_option).astype(np.float32)
    sel = ~np.isnan(obs_option)
    obs_state_sub = obs_state[sel, :]
    obs_option_sub = obs_option[sel]
    state = torch.from_numpy(obs_state_sub).to(model.device)
    option = torch.from_numpy(obs_option_sub).to(model.device)
    yprob = model(state)
    y = option.squeeze()
    ypred = yprob.argmax(dim=1)

    ypred_ary = ypred.cpu().detach().numpy()
    y_ary = y.cpu().detach().numpy().astype(int)
    yprob_ary = yprob.cpu().detach().numpy()

    # Accuracy, AUROC, F1 score, and Spearman rank-correlation coefficient
    acc = (ypred == y).sum() / len(y)
    auroc = roc_auc_score(y_ary, yprob_ary, multi_class="ovr")
    f1 = f1_score(y_ary, ypred_ary, average="macro")
    corr = sp.stats.spearmanr(y_ary, ypred_ary).correlation

    metrics = {"accuracy": acc.item(),
               "auroc": auroc,
               "f1": f1,
               "rank_corr": corr,
               "bcq_coverage_0.2": coverage(yprob_ary, y_ary, 0.2),
               "bcq_coverage_0.3": coverage(yprob_ary, y_ary, 0.3),
               "multi_f1_0.2": multi_output_f1_score(yprob_ary, y_ary, 0.2),
               "multi_f1_0.3": multi_output_f1_score(yprob_ary, y_ary, 0.3),
               "consistency_0.2": consistency(yprob_ary, 0.2),
               "consistency_0.3": consistency(yprob_ary, 0.3),
               "calibration_error": calibration_error(yprob_ary, y_ary),
               "calibration_error": calibration_error(yprob_ary, y_ary)}

    # Plotting
    plot_df = pd.DataFrame(
        {"INR_VALUE": data.df["INR_VALUE"][sel],
         "ACTION": ypred_ary},
        index=data.df.index[sel]
    )
    plots = {"heatmap": plot_policy_heatmap(plot_df)}

    return metrics, plots
