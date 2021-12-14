import torch

import numpy as np

import scipy as sp
import scipy.stats

import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score

from warfarin.evaluation.plotting import plot_policy_heatmap


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

    ypred_ary = ypred.cpu().detach()
    y_ary = y.cpu().detach()
    yprob_ary = yprob.cpu().detach()

    # Accuracy, AUROC, F1 score, and Spearman rank-correlation coefficient
    acc = (ypred == y).sum() / len(y)
    auroc = roc_auc_score(y_ary, yprob_ary, multi_class="ovr")
    f1 = f1_score(y_ary, ypred_ary, average="macro")
    corr = sp.stats.spearmanr(y_ary, ypred_ary).correlation

    metrics = {"accuracy": acc.item(),
               "auroc": auroc,
               "f1": f1,
               "rank_corr": corr}

    # Plotting
    plot_df = pd.DataFrame(
        {"INR_VALUE": data.df["INR_VALUE"][sel],
         "ACTION": ypred_ary},
        index=data.df.index[sel]
    )
    plots = {"heatmap": plot_policy_heatmap(plot_df)}

    return metrics, plots
