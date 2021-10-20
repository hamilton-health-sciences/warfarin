import torch

from sklearn.metrics import roc_auc_score


def evaluate_behavioral_cloning(model, data):
    yprob, y = [], []
    for batch in data:
        _, state, option, _, _, _ = batch
        yprob.append(model(state))
        y.append(option.squeeze())
    yprob = torch.cat(yprob, dim=0)
    y = torch.cat(y, dim=0)
    ypred = yprob.argmax(dim=1)

    # Accuracy
    acc = (ypred == y).sum() / len(y)

    # AUROC
    auroc = roc_auc_score(y.cpu().detach(), yprob.cpu().detach(),
                          multi_class="ovr")

    metrics = {"accuracy": acc, "auroc": auroc}

    return metrics
