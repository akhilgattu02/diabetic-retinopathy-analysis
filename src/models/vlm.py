from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "Qwen/Qwen2-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map="auto"
)
image_path = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/overlay.png"
image = Image.open(image_path).convert("RGB")
image.thumbnail((768, 768))  # prevent memory explosion

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Provide a detailed structured report of this retinal image and assess diabetic retinopathy severity based on segmented image on right and the labels."
            }
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        temperature=0.2,
        repetition_penalty=1.2
    )

print(processor.decode(output[0], skip_special_tokens=True))