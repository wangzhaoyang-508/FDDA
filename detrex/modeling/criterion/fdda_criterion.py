import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from detrex.layers import box_cxcywh_to_xyxy
from detrex.modeling.criterion import BaseCriterion


class FDDACriterion(BaseCriterion):
    """
    FDDA Criterion that extends BaseCriterion with disentanglement losses
    Implements: L_total = L_Det + β * L_Dis
    where L_Det = L_cls + L_reg (from BaseCriterion)
    and L_Dis = (L_En_a + L_De_a) + α * (L_En_d + L_De_d)
    """

    def __init__(self,
                 num_classes: int,
                 matcher: nn.Module,
                 alpha: float = 1.0,  # Balance factor for disentanglement loss components
                 beta: float = 1.0,  # Balance factor between detection and disentanglement loss
                 loss_class_weights: Dict[str, float] = None,
                 **kwargs):
        super().__init__(num_classes=num_classes, matcher=matcher, **kwargs)

        self.alpha = alpha
        self.beta = beta

        # Default loss weights if not provided
        if loss_class_weights is None:
            loss_class_weights = {
                "loss_class": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_encoder_alignment": 1.0,
                "loss_encoder_discriminant": 1.0,
                "loss_decoder_alignment": 1.0,
                "loss_decoder_discriminant": 1.0,
            }
        self.loss_class_weights = loss_class_weights

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss: L_total = L_Det + β * L_Dis

        Args:
            outputs: Model outputs containing:
                - pred_logits: classification logits
                - pred_boxes: bounding box predictions
                - disentanglement_losses: dictionary with encoder/decoder disentanglement losses
            targets: Ground truth targets
        """
        # Calculate standard detection loss using BaseCriterion (L_Det = L_cls + L_reg)
        detection_losses = super().forward(outputs, targets)

        # Apply weights to detection losses
        weighted_detection_loss = 0.0
        for loss_name, loss_value in detection_losses.items():
            weight = self.loss_class_weights.get(loss_name, 1.0)
            weighted_detection_loss += loss_value * weight

        # Calculate disentanglement loss L_Dis if available
        disentanglement_loss = 0.0
        disentanglement_loss_dict = {}

        if "disentanglement_losses" in outputs:
            dis_losses = outputs["disentanglement_losses"]

            # L_Dis = (L_En_a + L_De_a) + α * (L_En_d + L_De_d)
            alignment_loss = (dis_losses.get("loss_encoder_alignment", 0) +
                              dis_losses.get("loss_decoder_alignment", 0))
            discriminant_loss = (dis_losses.get("loss_encoder_discriminant", 0) +
                                 dis_losses.get("loss_decoder_discriminant", 0))

            # Apply weights to disentanglement losses
            alignment_loss_weighted = alignment_loss * self.loss_class_weights.get("loss_encoder_alignment", 1.0)
            discriminant_loss_weighted = discriminant_loss * self.loss_class_weights.get("loss_encoder_discriminant",
                                                                                         1.0)

            disentanglement_loss = alignment_loss_weighted + self.alpha * discriminant_loss_weighted

            # Store individual losses for logging
            disentanglement_loss_dict = {
                "loss_disentanglement_total": disentanglement_loss.detach(),
                "loss_disentanglement_alignment": alignment_loss.detach(),
                "loss_disentanglement_discriminant": discriminant_loss.detach(),
                "loss_encoder_alignment": dis_losses.get("loss_encoder_alignment", torch.tensor(0.0)).detach(),
                "loss_encoder_discriminant": dis_losses.get("loss_encoder_discriminant", torch.tensor(0.0)).detach(),
                "loss_decoder_alignment": dis_losses.get("loss_decoder_alignment", torch.tensor(0.0)).detach(),
                "loss_decoder_discriminant": dis_losses.get("loss_decoder_discriminant", torch.tensor(0.0)).detach(),
            }

        # Total loss: L_total = L_Det + β * L_Dis
        total_loss = weighted_detection_loss + self.beta * disentanglement_loss

        # Combine all losses for logging
        loss_dict = {
            **detection_losses,
            **disentanglement_loss_dict,
            "loss_total": total_loss.detach(),
            "loss_detection": weighted_detection_loss.detach(),
        }

        return loss_dict


class FDDASetCriterion(nn.Module):
    """
    FDDA Criterion using SetCriterion for detection loss computation
    Alternative implementation using SetCriterion instead of BaseCriterion
    """

    def __init__(self,
                 num_classes: int,
                 matcher: nn.Module,
                 weight_dict: Dict[str, float],
                 losses: List[str],
                 alpha: float = 0.5,                            # hyper-parameter
                 beta: float = 2.0,                             # hyper-parameter
                 **kwargs):
        super().__init__()

        # Use SetCriterion for detection losses
        self.detection_criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            **kwargs
        )

        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss using SetCriterion for detection losses
        """
        # Calculate detection losses using SetCriterion
        detection_loss_dict = self.detection_criterion(outputs, targets)
        detection_loss = sum(detection_loss_dict.values())

        # Calculate disentanglement loss
        disentanglement_loss = 0.0
        disentanglement_loss_dict = {}

        if "disentanglement_losses" in outputs:
            dis_losses = outputs["disentanglement_losses"]

            # L_Dis = (L_En_a + L_De_a) + α * (L_En_d + L_De_d)
            alignment_loss = (dis_losses.get("loss_encoder_alignment", 0) +
                              dis_losses.get("loss_decoder_alignment", 0))
            discriminant_loss = (dis_losses.get("loss_encoder_discriminant", 0) +
                                 dis_losses.get("loss_decoder_discriminant", 0))

            disentanglement_loss = alignment_loss + self.alpha * discriminant_loss

            disentanglement_loss_dict = {
                "loss_disentanglement_total": disentanglement_loss.detach(),
                "loss_disentanglement_alignment": alignment_loss.detach(),
                "loss_disentanglement_discriminant": discriminant_loss.detach(),
            }

        # Total loss: L_total = L_Det + β * L_Dis
        total_loss = detection_loss + self.beta * disentanglement_loss

        # Combine loss dictionaries
        loss_dict = {
            **detection_loss_dict,
            **disentanglement_loss_dict,
            "loss_total": total_loss.detach(),
        }

        return loss_dict