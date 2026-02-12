import torch

from goripy.tensor.info import sprint_tensor_info, sprint_tensor_stats


class CategoricalCrossEntropyLoss:
    """
    Categorical Cross-Entropy Loss function for category prediction.

    Tensor dimensions:

    - B: Batch size
    - C: Number of classes

    Args:
    
        cls_weights (torch.Tensor, optional):
            (C) dimensional tensor with class weights.
            If not provided, uniform weights will be used.

        label_smoothing (float):
            Label smoothing coefficient. Must lie in the (0, 1) interval.
            If not provided, no label smoothing will be applied.
    """

    def __init__(
        self,
        cls_weights=None,
        label_smoothing=None
    ):
        
        self._cls_weights = cls_weights
        self._label_smoothing = label_smoothing


    def __call__(
        self,
        pred_logits,
        target_probs,
        inst_weights=None
    ):
        """
        Computes the Categorical Cross-Entropy Loss function between predictions and ground truth.

        Args:

            pred_logits (torch.Tensor):
                (B x C) dimensional tensor with predicted category logits.
                Logits are features that are unnormalized with no activation function.

            target_probs (torch.Tensor):
                (B x C) dimensional tensor with target probabilities (one-hot encoded categories).

            inst_weights (torch.Tensor, optional):
                (B x C) dimensional tensor with instance weights.
                If not provided, uniform weights will be used.

        Returns:

            torch.Tensor:
                (B) dimensional tensor with the unreduced loss values.
        """

        if self._label_smoothing is not None:
            self._apply_label_smoothing(target_probs)

        pred_probs = torch.softmax(pred_logits, dim=1)

        loss_ten = target_probs * torch.log(pred_probs)

        if self._cls_weights is not None:
            loss_ten *= self._cls_weights[None, :]

        if inst_weights is not None:
            loss_ten *= inst_weights

        loss_ten = - torch.sum(loss_ten, axis=1)

        return loss_ten


    def _apply_label_smoothing(
        self,
        target_probs
    ):
        """
        Applies generalized label smoothing (inplace).

        Args:

            target_probs (torch.Tensor):
                (B x C) dimensional tensor with target probabilities (one-hot encoded categories).
        """

        cen_pull = (2 * target_probs) - 1

        pos_pull = torch.maximum(cen_pull, 0)
        neg_pull = torch.maximum(-cen_pull, 0)

        pos_pull /= numpy.sum(pos_pull)
        neg_pull /= numpy.sum(neg_pull)

        target_probs -= (pos_pull - neg_pull) * self._label_smoothing



class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss function for multi-attribute prediction.

    Tensor dimensions:

    - B: Batch size
    - A: Number of attributes

    Args:

        pos_attr_weights (torch.Tensor):
            (A) dimensional tensor with attribute positive weights.
            If not provided, uniform weights will be used.

        neg_attr_weights (torch.Tensor):
            (A) dimensional tensor with attribute negative weights.
            If not provided, uniform weights will be used.

        label_smoothing (float):
            Label smoothing coefficient. Must lie in the (0, 1) interval.
            If not provided, no label smoothing will be applied.
    """

    def __init__(
        self,
        pos_attr_weights=None,
        neg_attr_weights=None,
        label_smoothing=None
    ):
        
        self._pos_attr_weights = pos_attr_weights
        self._neg_attr_weights = neg_attr_weights
        self._label_smoothing = label_smoothing


    def __call__(
        self,
        pred_logits,
        target_probs,
        inst_weights=None
    ):
        """
        Computes the Binary Cross-Entropy Loss function between predictions and ground truth.

        Args:

            pred_logits (torch.Tensor):
                (B x A) dimensional tensor with predicted attribute logits.
                Logits are features that are unnormalized with no activation function.

            target_probs (torch.Tensor):
                (B x A) dimensional tensor with target probabilities (one-hot encoded attributes).

            inst_weights (torch.Tensor, optional):
                (B x A) dimensional tensor with instance weights.
                If not provided, uniform weights will be used.

        Returns:

            torch.Tensor:
                (B) dimensional tensor with the unreduced loss values.
        """

        if self._label_smoothing is not None:
            self._apply_label_smoothing(target_probs)

        pred_probs = torch.sigmoid(pred_logits)        

        pos_loss_ten = target_probs * torch.log(pred_probs)
        if self._pos_attr_weights is not None:
            pos_loss_ten *= self._pos_attr_weights[None, :]

        neg_loss_ten = (1 - target_probs) * torch.log(1 - pred_probs)
        if self._neg_attr_weights is not None:
            neg_loss_ten *= self._neg_attr_weights[None, :]

        loss_ten = pos_loss_ten + neg_loss_ten

        if inst_weights is not None:
            loss_ten *= inst_weights

        loss_ten = - torch.sum(loss_ten, axis=1)

        return loss_ten


    def _apply_label_smoothing(
        self,
        target_probs
    ):
        """
        Applies generalized label smoothing (inplace).

        Args:

            target_probs (torch.Tensor):
                (B x A) dimensional tensor with target probabilities (one-hot encoded attributes).
        """

        cen_pull = (2 * target_probs) - 1

        target_probs -= cen_pull * self._label_smoothing
