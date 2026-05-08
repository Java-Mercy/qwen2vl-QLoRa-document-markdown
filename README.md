# qwen2vl-QLoRa-document-markdown
It is good at academic papers. 

Live running on Gradio: https://huggingface.co/spaces/tahir-next/qwen2vl-document-markdown

It would take time to output so i already did a sample input so here is link of that: https://tahir-next-qwen2vl-document-markdown.hf.space/?__theme=system&deep_link=akqqYRoWAfY

So watch how it responded me on a piece of picture of research paper

# Fine-Tuning a Vision Language Model for Document Understanding with QLoRA

How we trained a 2B parameter model to convert research papers into Markdown 
Most research papers exist as PDFs or images. If you've ever tried extracting structured text from a paper - with equations, tables, headers, and formatting intact, you know regular OCR falls short. It strips away all the structure that makes a document readable.
We took a different approach: fine-tune a Vision Language Model (VLM) to look at a document image and output clean, structured Markdown.
The Setup
Model: Qwen2-VL-2B-Instruct - a 2 billion parameter model that can process both images and text simultaneously. It has a vision encoder that "sees" the image and a language decoder that generates text based on what it sees.
Dataset: The Nougat training dataset, which contains pages from arXiv papers paired with their Markdown representations. Each sample is a PNG image of a document page and a corresponding .mmd file with the structured text.
Hardware: Two NVIDIA T4 GPUs on Kaggle
Each T4 has 16GB of VRAM which is not enough to fine-tune a 2B model conventionally.
Technique: QLoRA (Quantized Low-Rank Adaptation)
What is QLoRA and Why Does It Matter?
A 2B parameter model in standard float16 precision takes about 4GB just for weights. Add optimizer states and activations during training, and you need 12–16GB right at the limit of a T4. One bad batch and you're out of memory.
QLoRA solves this with two ideas:
4-bit Quantization: Instead of storing each weight as a 16-bit number, we compress them to 4 bits using NF4 (NormalFloat4). This format is designed specifically for neural network weights, which follow a bell curve distribution. NF4 places its 16 quantization levels optimally for this distribution, preserving more information than naive 4-bit rounding. The 2B model now fits in about 1GB.
Low-Rank Adaptation: Instead of updating all 2 billion parameters, we freeze everything and inject tiny trainable matrices (called adapters) into the attention and MLP layers. These adapters have a "rank" of 16, meaning each adapter matrix is very small. In total, we only train about 1.5% of the model's parameters roughly 30 million out of 2 billion.
The result: we can fine-tune a model that would normally need an A100 GPU on a free Kaggle T4.
Data Preparation
The Nougat dataset contains 2,230 paper folders, each with multiple page images and corresponding Markdown files. Our first step was exploring the data:
Total document pages: several thousand
Each page has a paired PNG image and .mmd (Markdown) file
Markdown lengths range from a few hundred to several thousand characters
Content includes equations (LaTeX), tables, headers, lists, and plain paragraphs

We formatted each sample as a ChatML conversation:
System message: "You are a document understanding assistant…"
User message: [image] + "Convert this to Markdown"
Assistant message: the actual Markdown text

This matches the conversation format Qwen2-VL was pretrained with, so the model knows exactly what's expected.
We used 800 samples with an 80/20 train/validation split.
Challenges We Hit
Corrupted images in the dataset. About 30 minutes into our first training run, it crashed on a corrupted PNG file. PIL couldn't read it, and the entire run was lost.
The fix: before training, we validate every single image using PIL's verify() method, which checks file integrity without fully loading pixel data. This adds 2-3 minutes upfront but catches all corrupted files. We also added try/except in the data collator as a safety net.
BFloat16 compatibility on T4 GPUs. Qwen2-VL internally stores some layers in bfloat16 format. When we enabled fp16 mixed precision training, the gradient scaler crashed because its CUDA kernel for gradient unscaling doesn't support bfloat16 on T4 hardware. T4 GPUs use Turing architecture (2018), which predates native bfloat16 support that came with Ampere (A100, 2020).
The fix: we removed fp16 mixed precision entirely. Since the model is already in 4-bit quantization with compute dtype set to float16, the forward pass is already memory-efficient. The gradient scaler wasn't providing additional benefit in this setup.
Checkpoint strategy. We changed from saving per-epoch to saving every 20 training steps, with automatic resume from the latest checkpoint. If training crashes for any reason, re-running the cell picks up where it left off instead of starting over.
Training Configuration
Batch size: 1 (with gradient accumulation over 8 steps for effective batch size of 8)
Learning rate: 2e-4 with cosine scheduler and warmup
LoRA rank: 16, alpha: 32
Target modules: all attention projections (q, k, v, o) and MLP projections (gate, up, down)
Image resolution: 512x512 maximum
Epochs: 3

Results
The fine-tuned model successfully converts unseen document images into structured Markdown. Comparing zero-shot (base model without fine-tuning) against our fine-tuned version shows clear improvement - the base model often produces generic descriptions of the image instead of extracting the actual text content, while the fine-tuned model generates proper headers, preserves equation formatting, and maintains paragraph structure.
We also compared different prompt styles (minimal vs. detailed vs. structured instructions) and found that the detailed prompt produced the most consistent results, though the differences were smaller than expected suggesting the model internalized the task well during fine-tuning.
Deployment
We deployed the model as a Gradio app on HuggingFace Spaces. Only the LoRA adapter weights (about 50MB) need to be hosted, the base model is loaded from HuggingFace's model hub at runtime. Users can upload any document image and get Markdown output in seconds.
Key Takeaways

QLoRA makes it possible to fine-tune large models on consumer/free GPUs. The adapter weights are just 50MB compared to the full model's 2GB+.
Always validate your dataset before training. A single corrupted file can waste hours of GPU time.
Hardware generation matters. T4s don't support all the same operations as newer A100s. Understanding your hardware's capabilities saves debugging time.
Checkpoint frequently. Training crashes happen - make them recoverable instead of catastrophic.

Try It Yourself
Live demo: 
Code: 
Model adapter: 

Built by Tahir and Ehtisham at FAST-NUCES
