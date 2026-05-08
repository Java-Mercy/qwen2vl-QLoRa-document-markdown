import torch
import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

adapter_id = "tahir-next/qwen2vl-nougat-qlora"
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

processor = AutoProcessor.from_pretrained(base_model_id, min_pixels=256*256, max_pixels=512*512)

def generate(image):
    if image is None:
        return "Please upload an image first."

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Convert this document image into Markdown. Preserve all headings, paragraphs, equations, and formatting."},
            ],
        }
    ]

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text.strip()

demo = gr.Interface(
    fn=generate,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=gr.Textbox(label="Markdown Output", lines=20),
    title="Document-to-Markdown (Qwen2-VL + QLoRA)",
    description="Fine-tuned on the Nougat dataset for scientific document conversion.",
)

if __name__ == "__main__":
    demo.launch()