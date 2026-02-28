import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# ------------------------
# Configuration
# ------------------------

model_id = "Qwen/Qwen2-VL-2B-Instruct"
image_path = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/overlay.png"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------
# Load Processor + Model
# ------------------------

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map="auto"
)

model.eval()

# ------------------------
# Load Image
# ------------------------

image = Image.open(image_path).convert("RGB")

# ------------------------
# Structured Medical Prompt
# ------------------------

prompt = """
<image>

You are an ophthalmology AI assistant.

Analyze the RIGHT retinal fundus image and generate a detailed structured report.

Follow this format strictly:

IMAGE QUALITY:
- Clarity:
- Illumination:
- Field of view:

ANATOMICAL FINDINGS:
- Optic disc:
- Macula:
- Retinal vessels:

LESION ANALYSIS:
- Microaneurysms:
- Hemorrhages:
- Hard exudates:
- Cotton wool spots:
- Neovascularization:

DIABETIC RETINOPATHY STAGE:
- Stage classification:
- Severity level:
- Confidence level:

PATIENT SUMMARY:
Explain findings in simple language.
"""

# ------------------------
# Prepare Inputs
# ------------------------

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(device)

# ------------------------
# Generate Output
# ------------------------

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        temperature=0.2,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=processor.tokenizer.eos_token_id
    )

report = processor.decode(output[0], skip_special_tokens=True)

print("\n" + "="*80)
print("RETINAL ANALYSIS REPORT")
print("="*80 + "\n")
print(report)