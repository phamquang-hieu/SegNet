from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha: float, beta: float, weight: list=None, ignore_index=None, reduction='none') -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6
        self.weight = None
        self.ignore_index = ignore_index
        self.reduction = reduction
        if weight is not None:
            self.weight = torch.Tensor(weight)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).type(input.dtype).to(input.device)

        # compute the actual dice score
        dims = (2, 3) # left the 1 dimension for later FocalLoss
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)

        if self.weight is not None:
            if self.ignore_index is not None:
                self.weight[self.ignore_index] = 0.00
            tversky_loss = tversky_loss * self.weight.type(tversky_loss.dtype).to(tversky_loss.device)
        # print(tversky_loss)
        # print(tversky_loss.shape) [1, #classes]
        if self.reduction == 'none':
            return 1. - tversky_loss
        if self.reduction == 'mean':
            return torch.mean(1. - tversky_loss)

    ######################
    # functional interface
    ######################

    def tversky_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        beta: float) -> torch.Tensor:
        
        return TverskyLoss(alpha, beta)(input, target)