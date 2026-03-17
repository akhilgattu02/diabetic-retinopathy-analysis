import segmentation_models_pytorch as smp
import torch

#Use UNET++ model with timm-efficientnet-b3 encoder and imagenet encoder weights
model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6,
    decoder_attention_type="scse",
)