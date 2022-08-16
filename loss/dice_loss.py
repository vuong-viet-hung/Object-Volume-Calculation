import torch


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predicted_masks: torch.Tensor, true_masks: torch.Tensor, smooth: float = 1.0) -> float:
        predicted_masks = torch.nn.Sigmoid()(predicted_masks)

        # Flatten the true and predicted masks
        predicted_masks = predicted_masks.contiguous().view(-1)
        true_masks = true_masks.contiguous().view(-1)

        intersection = (predicted_masks * true_masks).sum()

        a_sum = torch.sum(predicted_masks * predicted_masks)
        b_sum = torch.sum(true_masks * true_masks)

        dice = (2 * intersection + smooth) / (a_sum + b_sum + smooth)

        return 1 - dice
