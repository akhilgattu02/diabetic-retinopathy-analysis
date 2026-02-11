import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import matplotlib.patches as mpatches


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/data/processed/test/images/IDRiD_55.jpg"
MODEL_PATH = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/checkpoints/unet_plus_plus_idridd.pth"
DEVICE = "mps" if torch.mps.is_available() else "cpu"

# -----------------------------
# MODEL
# -----------------------------
model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6,
    activation=None,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
orig = cv2.imread(IMAGE_PATH)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
orig = cv2.resize(orig, (512, 512))

# Keep a copy for display
display_img = orig.copy()

# -----------------------------
# PREPROCESS (ImageNet norm)
# -----------------------------
img = orig.astype(np.float32) / 255.0


img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

# -----------------------------
# INFERENCE
# -----------------------------
with torch.no_grad():
    output = model(img)

pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
print("Predicted unique classes:", np.unique(pred_mask))
print("Lesion pixels:", np.sum(pred_mask > 0))

# -----------------------------
# COLOR MAP
# -----------------------------
class_colors = {
    0: (0,0,0),        # background
    1: (255,0,0),      # MA
    2: (0,255,0),      # HE
    3: (0,0,255),      # EX
    4: (255,255,0),    # SE
    5: (255,0,255),    # OD
}

class_names = {
    0: "Background",        # background
    1: "Microaneurysms",      # MA
    2: "Haemorrhages",      # HE
    3: "Hard Exudates",      # EX
    4: "Soft Exudates",    # SE
    5: "Optic Disc",    # OD
}

color_mask = np.zeros((512,512,3), dtype=np.uint8)

for cid, color in class_colors.items():
    color_mask[pred_mask == cid] = color

# -----------------------------
# OVERLAY
# -----------------------------
alpha = 0.5
overlay = display_img.copy()
print(display_img.shape, color_mask.shape, overlay.shape)

lesion_pixels = pred_mask > 0
overlay[lesion_pixels] = (
    (1-alpha)*display_img[lesion_pixels] +
    alpha*color_mask[lesion_pixels]
).astype(np.uint8)

contours, _ = cv2.findContours(
    lesion_pixels.astype(np.uint8),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(overlay, contours, -1, (255,255,255), 1)
# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(12,6))


plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(display_img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Segmentation Overlay")
plt.imshow(overlay)
plt.axis("off")

# Create legend
patches = []
for cid, name in class_names.items():
    if np.any(pred_mask == cid):
        color = np.array(class_colors[cid]) / 255.0
        patches.append(mpatches.Patch(color=color, label=name))

plt.legend(
    handles=patches,
    bbox_to_anchor=(1.05,1),
    loc="upper left",
    borderaxespad=0.
)

plt.tight_layout()
plt.show()

