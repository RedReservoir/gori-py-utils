import torch



class CategoricalCrossEntropyLoss():
    """
    Categorical Cross-Entroy Loss function for category prediction.

    Tensor dimensions:
      - B: Batch size
      - C: Number of classes

    :param cls_weights: torch.Tensor, optional
        (C) dimensional tensor with class weights.
        If not provided, uniform weights will be used.
    """

    def __init__(
        self,
        cls_weights=None
    ):
        
        self._cls_weights = cls_weights


    def __call__(
        self,
        pred_logits,
        target_probs,
        inst_weights=None
    ):
        """
        Computes the Categorical Cross-Entroy Loss function between predictions and ground truth.

        :param pred_logits: torch.Tensor
            (B x C) dimensional tensor with predicted category logits.
            Logits are features that are unnormalized with no activation function.
        :param target_probs: torch.Tensor
            (B x C) dimensional tensor with target one-hot encoded categories.
        :param inst_weights: torch.Tensor, optional
            (B x C) dimensional tensor with instance weights.
            If not provided, uniform weights will be used.

        :return: torch.Tensor
            (B)-dimensional tensor with the unreduced loss values.
        """

        pred_probs = torch.softmax(pred_logits, dim=1)
        
        loss_ten = target_probs * torch.log(pred_probs)

        if self._cls_weights is not None:
            loss_ten *= self._cls_weights[None, :]

        if inst_weights is not None:
            loss_ten *= inst_weights

        loss_ten = - torch.sum(loss_ten, axis=1)

        return loss_ten



def categorical_cross_entropy_loss(
    pred_logits,
    target_probs,
    cls_weights=None,
    inst_weights=None
):
    """
    Categorical Cross-Entroy Loss function for category prediction.
    Functional version.
    
    Tensor dimensions:
      - B: Batch size
      - C: Number of classes

    :param pred_logits: torch.Tensor
        (B x C) dimensional tensor with predicted category logits.
        Logits are features that are unnormalized with no activation function.
    :param target_probs: torch.Tensor
        (B x C) dimensional tensor with target one-hot encoded categories.
    :param cls_weights: torch.Tensor, optional
        (C) dimensional tensor with class weights.
        If not provided, uniform weights will be used.
    :param inst_weights: torch.Tensor, optional
        (B x C) dimensional tensor with instance weights.
        If not provided, uniform weights will be used.

    :return: torch.Tensor
        (B)-dimensional tensor with the unreduced loss values.
    """

    pred_probs = torch.softmax(pred_logits, dim=1)
    
    loss_ten = target_probs * torch.log(pred_probs)

    if cls_weights is not None:
        loss_ten *= cls_weights[None, :]

    if inst_weights is not None:
        loss_ten *= inst_weights

    loss_ten = - torch.sum(loss_ten, axis=1)

    return loss_ten



class BinaryCrossEntropyLoss():
    """
    Binary Cross-Entroy Loss function for multi-attribute prediction.

    Tensor dimensions:
      - B: Batch size
      - A: Number of attributes

    :param pos_attr_weights: torch.Tensor
        (A) dimensional tensor with attribute positive weights.
        If not provided, uniform weights will be used.
    :param neg_attr_weights: torch.Tensor
        (A) dimensional tensor with attribute positive weights.
        If not provided, uniform weights will be used.
    """

    def __init__(
        self,
        pos_attr_weights=None,
        neg_attr_weights=None
    ):
        
        self._pos_attr_weights = pos_attr_weights
        self._neg_attr_weights = neg_attr_weights
        

    def __call__(
        self,
        pred_logits,
        target_probs,
        inst_weights=None
    ):
        """
        Computes the Binary Cross-Entroy Loss function between predictions and ground truth.

        :param pred_logits: torch.Tensor
            (B x A) dimensional tensor with predicted attribute logits.
            Logits are features that are unnormalized with no activation function.
        :param target_probs: torch.Tensor
            (B x A) dimensional tensor with target one-hot encoded attributes.
        :param inst_weights: torch.Tensor, optional
            (B x A) dimensional tensor with instance weights.
            If not provided, uniform weights will be used.

        :return: torch.Tensor
            (B) dimensional tensor with the unreduced loss values.
        """

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
        


def binary_cross_entropy_loss(
    pred_logits,
    target_probs,
    pos_attr_weights=None,
    neg_attr_weights=None,
    inst_weights=None
):
    """
    Binary Cross-Entroy Loss function for multi-attribute prediction.
    Functional version.

    Tensor dimensions:
      - B: Batch size
      - A: Number of attributes

    :param pred_logits: torch.Tensor
        (B x A) dimensional tensor with predicted attribute logits.
        Logits are features that are unnormalized with no activation function.
    :param target_probs: torch.Tensor
        (B x A) dimensional tensor with target one-hot encoded attributes.
    :param pos_attr_weights: torch.Tensor
        (A) dimensional tensor with attribute positive weights.
        If not provided, uniform weights will be used.
    :param neg_attr_weights: torch.Tensor
        (A) dimensional tensor with attribute positive weights.
        If not provided, uniform weights will be used.
    :param inst_weights: torch.Tensor, optional
        (B x A) dimensional tensor with instance weights.
        If not provided, uniform weights will be used.

    :return: torch.Tensor
        (B) dimensional tensor with the unreduced loss values.
    """

    pred_probs = torch.sigmoid(pred_logits)        

    pos_loss_ten = target_probs * torch.log(pred_probs)
    if pos_attr_weights is not None:
        pos_loss_ten *= pos_attr_weights[None, :]

    neg_loss_ten = (1 - target_probs) * torch.log(1 - pred_probs)
    if neg_attr_weights is not None:
        neg_loss_ten *= neg_attr_weights[None, :]

    loss_ten = pos_loss_ten + neg_loss_ten

    if inst_weights is not None:
        loss_ten *= inst_weights

    loss_ten = - torch.sum(loss_ten, axis=1)

    return loss_ten
