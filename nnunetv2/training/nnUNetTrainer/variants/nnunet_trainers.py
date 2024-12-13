from nnunetv2.training.nnUNetTrainer.variants.losses import FocalLoss, FocalLossV2, FocalLossAndCrossEntropyLoss, FocalLossAndCrossEntropyLossV2

import numpy as np

from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerFocalLoss(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have Focal Loss with ADAPTED FocalLoss implementation from:
    https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py
    """
    def __init__(self, *args, **kwargs):
        super(nnUNetTrainerFocalLoss, self).__init__(*args, **kwargs)

    def _build_loss(self):
        loss = FocalLoss(
            gamma=2, smooth=1e-5, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i)
                               for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFLAndCELoss(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have a combination of Focal Loss
    and CrossEntropyLoss with ADAPTED FocalLoss Implementation from:
    https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py
    """
    def __init__(self, *args, **kwargs):
        super(nnUNetTrainerFocalLoss, self).__init__(*args, **kwargs)

    def _build_loss(self):
        loss = FocalLossAndCrossEntropyLoss(
            gamma=2,
            smooth=1e-5,
            ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i)
                               for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalLossV2(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have a Focal Loss
    COPIED from implementation:
    https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py

    """
    def __init__(self, *args, **kwargs):
        super(nnUNetTrainerFocalLoss, self).__init__(*args, **kwargs)

    def _build_loss(self):
        loss = FocalLossV2()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i)
                               for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFLAndCELossV2(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have a combination of Focal Loss
    and CrossEntropyLoss with copied FocalLoss implementation from:
    https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/focal_loss.py

    """

    def _build_loss(self):
        loss = FocalLossAndCrossEntropyLossV2()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i)
                               for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
