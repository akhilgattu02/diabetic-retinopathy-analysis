import segmentation_models_pytorch as smp
import torch

DEVICE = 'mps' if torch.mps.is_available() else 'cpu'

DiceLoss = smp.losses.DiceLoss
FocalLoss = smp.losses.FocalLoss

dice_loss = smp.losses.DiceLoss(
    mode='multiclass',
    from_logits=True
)

def multiclass_iou(preds, targets, num_classes, eps=1e-6):
    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).float().sum()
        union = pred_inds.float().sum() + target_inds.float().sum() - intersection

        if union == 0:
            continue

        ious.append((intersection + eps) / (union + eps))

    return torch.mean(torch.stack(ious))

weights = torch.tensor([
    0.05,  # background
    4.0,   # MA (tiny lesions)
    3.0,   # HE
    2.5,   # EX
    2.0,   # SE
    1.0    # OD
], dtype=torch.float32).to(DEVICE)

ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

