import torch
import torch.nn.functional as F


def ce_loss(logits, targets, reduction="none"):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


###### Loss Fixed Threshold ######
def consistency_loss(logits_u_str, logits_u_w, threshold=0.6):
    """
    Consistency regularization for fixed threshold loss in semi-supervised learning.
    Args:
        logits_u_str: logits of strong augmented unlabeled samples
        logits_u_w: logits of weak augmented unlabeled samples
        threshold: fixed threshold
    Returns:
        loss: consistency regularization loss
    """
    pseudo_label = torch.softmax(logits_u_w, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(threshold).float()
    loss = (ce_loss(logits_u_str, targets_u, reduction="none") * mask).mean()
    return loss
##################################
