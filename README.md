# TinyLlama-VLM-LoRA

A vision-language model (VLM) built by fine-tuning TinyLlama (1.1B) with LoRA adapters on Flickr30k image-caption pairs, grafted onto a frozen CLIP vision encoder. This repository provides:

- **LoRA adapter weights** for TinyLlama’s `q_proj` and `v_proj` layers (trained on 30K images, 3 epochs).
- **Projector** (`768 → 512`) and **Gate** (`512 → 4096`) state-dicts (FP16) to convert CLIP’s `[CLS]` embedding into a “vision token.”
- **Tokenizer files** matching TinyLlama’s BPE vocabulary, with `<pad>` set to `<eos>`.
- **Inference and evaluation scripts** (with BLEU, ROUGE-1, F1, and perplexity).

---

## Model Details

### Model Description

TinyLlama-VLM-LoRA is a multi-modal extension of TinyLlama (1.1B-parameter Chat model). We freeze CLIP’s ViT-Base vision encoder and sandwich its `[CLS]` embedding into TinyLlama by:

1. **CLIP Vision Encoder (frozen)**
   - **Input:** RGB image (`224×224`)
   - **Output:** 768-dim CLS embedding (FP16)

2. **Projector** (`nn.Linear(768 → 512, dtype=torch.float16)`)
   - Reduces CLIP’s `[CLS]` from 768 → 512.

3. **Gate** (`nn.Linear(512 → 4096, dtype=torch.float16)`)
   - Expands the 512 → TinyLlama’s hidden size (4096) to form a single “vision token” vector.

4. **TinyLlama + LoRA**
   - We attach LoRA adapters (`r=8, α=16, drop=0.1`) onto TinyLlama’s `q_proj` and `v_proj` layers. During training, only LoRA, Projector, and Gate parameters are updated (TinyLlama’s base weights remain frozen).
   - The “vision token” is prepended to the usual token embeddings. The combined sequence `(vision_token + text_tokens[0…L−2])` is fed to TinyLlama to predict the image caption `(text_tokens[1…L−1]` masked appropriately).

---

### File/Version Breakdown

- **adapter_config.json**, **adapter_model.safetensors**  
  LoRA configuration + learned delta weights for TinyLlama’s `q_proj`/`v_proj` (≈ 4 MB).

- **proj_final.pt** (≈ 1.6 MB)  
  `nn.Linear(768 → 512)` state_dict after final epoch (FP16).

- **gate_final.pt** (≈ 4.2 MB)  
  `nn.Linear(512 → 4096)` state_dict after final epoch (FP16).

- **Tokenizer files**  
  - `tokenizer.json`  
  - `tokenizer.model`  
  - `tokenizer_config.json`  
  - `special_tokens_map.json`  
  Exact BPE vocabulary, merges, and special token mappings used during fine-tuning.

- **finetuned_qvlam_flickr30k_final/**  
  A directory containing:  
  - LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`)  
  - `tokenizer.json` + related tokenizer artifacts  
  > This folder can be passed directly to `PeftModel.from_pretrained` and `AutoTokenizer.from_pretrained`.

---

## Uses

1. **Caption Generation**  
   Given an arbitrary RGB image, produce a descriptive caption in plain English via beam search.

2. **Downstream Fine-Tuning**  
   Further adapt the LoRA adapters on a new, smaller image-caption dataset, or graft additional modules on top of the vision token.

3. **Research / Educational**  
   Demonstrates a lightweight VLM pipeline (CLIP → TinyLlama) using LoRA, projector, and gate. Useful as a starting point for more advanced multi-modal research (e.g., adding a quantum layer).

---

## Quick Start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/TinyLlama-VLM-LoRA.git
   cd TinyLlama-VLM-LoRA
