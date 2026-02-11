import segmentation_models_pytorch as smp
import torch

DEVICE = 'mps' if torch.mps.is_available() else 'cpu'

DiceLoss = smp.losses.DiceLoss
FocalLoss = smp.losses.FocalLoss

dice_loss = smp.losses.DiceLoss(
    mode='multiclass',
    from_logits=True
)

weights = torch.tensor([
    0.05,  # background
    4.0,   # MA (tiny lesions)
    3.0,   # HE
    2.5,   # EX
    2.0,   # SE
    1.0    # OD
], dtype=torch.float32).to(DEVICE)

ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

