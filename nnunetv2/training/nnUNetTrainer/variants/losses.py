import numpy as np

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2,
        smooth: float = 0.0,
        weight: torch.Tensor | None = None,
        nonlinearity: Callable[[torch.Tensor],
                               torch.Tensor] = nn.Softmax(dim=1),
        ignore_index: int = -100
    ) -> None:
        """
        Focal loss that supports label smoothing and multiclass tasks. Inspired by the
        implementation in
        https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_focalLoss.py

        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t). Larger values
                increase the weighting of misclassified samples.
            smooth (float, optional): A float in [0.0, 1.0]. Specifies the amount of
                label smoothing to use when computiing the loss. 0.0 corresponds to no
                smoothing.
            weight (torch.Tensor | None, optional): A manual rescaling weight given to
                each class. If provided, must be a Tensor of size C and floating point
                type where C is the number of classes. Meant to replace the alpha term
                in focal loss in a more explicit manner.
            nonlinearity (nn.Module): Nonlinearity to apply to output layer of nnUNet.
                Must constrain logits to the range [0, 1]
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        assert smooth >= 0 and smooth <= 1, "Smooth must be in range [0, 1]"
        self.smooth = smooth
        self.nonlinearity = nonlinearity
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes Focal Loss on logits and targets with initialized gamma and alpha.

        Args:
            logits (torch.Tensor): A float tensor (logits) of arbitrary shape. The
                predictions for each example. Must have be one hot encoded whith shape
                (N, C, ...) where N is the number of samples and C is the number of
                classes.
            targets (torch.Tensor): A float tensor containing correct class labels. If
                the same shape as logits, pytorch assumes it has already been one hot
                encoded. Otherwise, pytorch assumes it contains class indices.

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        # Nnunet weirdness, get rid of an empty dim
        if targets.ndim == logits.ndim:
            assert targets.shape[1] == 1
            targets = targets[:, 0]

        # Ensure target is an int. In nnunet they are floats
        targets = targets.long()

        # logits shape (N, C, ...), targets shape either (N, ...) or (N, C, ...)
        num_classes = logits.size(1)  # C
        ce_loss = F.cross_entropy(
            logits, targets, self.weight, reduction="none", label_smoothing=self.smooth, ignore_index=self.ignore_index
        )  # Returns shape (N, ...)

        # Ensure targets are one hot encoded
        if logits.shape != targets.shape:
            targets = F.one_hot(
                targets, num_classes=num_classes).movedim(-1, 1)
            assert logits.shape == targets.shape, f"Logits must have Shape (N, C, ...). Logits shape: {logits.shape}, Targets Shape after OHE: {targets.shape}"

        # Get probs and smooth labels
        probs = self.nonlinearity(logits)  # Shape (N, C, ...)
        targets_smoothed = (1 - self.smooth) * targets + \
            self.smooth / num_classes

        # p_t contains the predicted probability of the correct class
        p_t = (probs * targets_smoothed).sum(1)  # Shape (N, ...)

        loss = ((1 - p_t) ** self.gamma) * ce_loss

        return loss.mean()


class FocalLossAndCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2,
        smooth: float = 0.0,
        weight: torch.Tensor | None = None,
        nonlinearity: Callable[[torch.Tensor],
                               torch.Tensor] = nn.Softmax(dim=1),
        focal_loss_weight: float = 0.5,
        ignore_index: int = -100
    ):
        """
        Combination of FocalLoss and CrossEntropy Loss. Inputs must be logits and not probabilities.

        Implementation of FocalLoss adapted from:
        https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_focalLoss.py

        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t). Larger values
                increase the weighting of misclassified samples inf Focal Loss.
            smooth (float, optional): A float in [0.0, 1.0]. Specifies the amount of
                label smoothing to use when computiing the Focal loss. 0.0 corresponds
                to no smoothing.
            weight (torch.Tensor | None, optional): A manual rescaling weight given to
                each class. If provided, must be a Tensor of size C and floating point
                type where C is the number of classes. Meant to replace the alpha term
                in focal loss in a more explicit manner. Does not apply to the CE loss
                component.
            nonlinearity (nn.Module): Nonlinearity to apply to output layer of nnUNet.
                Must constrain logits to the range [0, 1]
            focal_loss_weight (float): Weight between 0 and 1 to apply to weight focal 
                loss. (1 - focal_loss_weight) applied to cross entropy loss.
            ignore_index (int): Class index to ignore
        """
        super().__init__()

        assert focal_loss_weight >= 0 and focal_loss_weight <= 1.0
        self.focal_loss_weight = focal_loss_weight

        self.focal_loss = FocalLoss(
            gamma, smooth, weight, nonlinearity, ignore_index)
        self.cross_entropy_loss = RobustCrossEntropyLoss(
            ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Computes combination Cross Entropy Loss and Focal Loss on logits
            and targets with initialized gamma and alpha.

        Args:
            logits (torch.Tensor): A float tensor (logits) of arbitrary shape.
                    The predictions for each example.
            targets (torch.Tensor): A float tensor with the same shape as inputs. Stores
                the binary classification label for each element in inputs (0 for the
                negative class and 1 for the positive class).

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        floss = self.focal_loss(logits, targets)
        celoss = self.cross_entropy_loss(logits, targets)
        result = self.focal_loss_weight * floss + \
            (1 - self.focal_loss_weight) * celoss
        return result


class FocalLossV2(nn.Module):
    """
    copied from: https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py

    orginal implementation from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        """
        Computes Focal Loss on logits and targets with initialized gamma and alpha.

        Args:
            logit (torch.Tensor): A float tensor (logits) of arbitrary shape. The
                predictions for each example. Must have be one hot encoded whith shape
                (N, C, ...) where N is the number of samples and C is the number of
                classes.
            target (torch.Tensor): A float tensor containing correct class labels. If
                the same shape as logits, pytorch assumes it has already been one hot
                encoded. Otherwise, pytorch assumes it contains class indices.

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]

        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLossAndCrossEntropyLossV2(nn.Module):
    def __init__(self, focal_loss_kwargs=None, cross_entropy_loss_kwargs=None, focal_loss_weight: float = 0.5):
        """
        Combination of FocalLoss and CrossEntropy Loss. Inputs must be logits and not probabilities.

        Implementation of FocalLoss COPIED from: 
        https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py

        Args:
            focal_loss_weight (float): Weight between 0 and 1 to apply to weight focal 
                loss. (1 - focal_loss_weight) applied to cross entropy loss.
        """
        super(FocalLossAndCrossEntropyLossV2, self).__init__()
        if focal_loss_kwargs is None:
            focal_loss_kwargs = {}
        if cross_entropy_loss_kwargs is None:
            cross_entropy_loss_kwargs = {}

        self.focal_loss = FocalLossV2(
            apply_nonlin=nn.Softmax(), **focal_loss_kwargs)
        self.cross_entropy_loss = RobustCrossEntropyLoss(
            **cross_entropy_loss_kwargs)
        self.focal_loss_weight = focal_loss_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Computes combination Cross Entropy Loss and Focal Loss on logits
            and targets with initialized gamma and alpha.

        Args:
            logits (torch.Tensor): A float tensor (logits) of arbitrary shape.
                    The predictions for each example.
            targets (torch.Tensor): A float tensor with the same shape as inputs. Stores
                the binary classification label for each element in inputs (0 for the
                negative class and 1 for the positive class).

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        floss = self.focal_loss(logits, targets)
        celoss = self.cross_entropy_loss(logits, targets)
        result = self.focal_loss_weight * floss + \
            (1 - self.focal_loss_weight) * celoss
        return result
