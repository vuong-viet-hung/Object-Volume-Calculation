def compute_dice_coefficient(pred, true):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()
    m2 = true.view(num, -1).float()
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
