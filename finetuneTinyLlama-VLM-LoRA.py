import os
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION (define EPOCHS before using it)
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS               = 3            # number of training epochs
LEARNING_RATE        = 2e-7         # reduced LR to prevent FP16 overflow/NaN
GRADIENT_CLIP_NORM   = 1.0          # max norm for gradient clipping
MAX_LENGTH           = 50           # max token length (prompt + caption)
BATCH_SIZE           = 1            # increase to 2 if your VRAM allows
NUM_WORKERS          = 0            # 0 on Windows; increase if Linux
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device = {DEVICE}")
print(f"EPOCHS = {EPOCHS}, Batch size = {BATCH_SIZE}, Num workers = {NUM_WORKERS}")

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD FLICKR30K AND SPLIT INTO TRAIN/VALID
# ──────────────────────────────────────────────────────────────────────────────
ds = load_dataset("nlphuji/flickr30k")
print("Available splits:", ds.keys())  # should show ['test']
full_hf = ds["test"]  # 31,014 image‐records, each has 5 captions

split = full_hf.shuffle(seed=42).train_test_split(test_size=0.10, seed=42)
train_hf = split["train"]  # ~27,912 images
valid_hf = split["test"]   # ~3,102 images
print("Train images:", len(train_hf), "Valid images:", len(valid_hf))

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD CLIP AND TINYLLAMA + LoRA + TOKENIZER (ALL IN FP16 FOR MODEL)
# ──────────────────────────────────────────────────────────────────────────────
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=torch.float16
).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

llama = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
llama = get_peft_model(llama, lora_cfg)
llama.gradient_checkpointing_enable()

# ──────────────────────────────────────────────────────────────────────────────
# 3. DEFINE ITERABLE DATASET (EXPAND EACH IMAGE INTO 5 CAPTIONS)
# ──────────────────────────────────────────────────────────────────────────────
class HFImageCaptionIterableDataset(IterableDataset):
    def __init__(self, hf_split, clip_processor, tokenizer, max_length=MAX_LENGTH):
        super().__init__()
        self.hf_split = hf_split
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_text = "Describe the image: "

    def __iter__(self):
        for example in self.hf_split:
            # 3.1) Preprocess image → pixel_values [3,224,224], fp16
            pixel_vals = self.clip_processor(
                images=example["image"],
                return_tensors="pt"
            ).pixel_values.squeeze(0).to(DEVICE)

            # 3.2) Tokenize the prompt to get prompt_len
            prompt_tok = self.tokenizer(
                self.prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            prompt_ids = prompt_tok.input_ids.squeeze(0).to(DEVICE)
            prompt_len = prompt_ids.size(0)

            # 3.3) For each of the 5 captions:
            for caption_str in example["caption"]:
                full_text = self.prompt_text + caption_str
                full_tok = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True
                )
                full_input_ids = full_tok.input_ids.squeeze(0).to(DEVICE)         # [max_length]
                full_attention_mask = full_tok.attention_mask.squeeze(0).to(DEVICE)  # [max_length]

                yield {
                    "pixel_values":    pixel_vals,            # [3,224,224], fp16
                    "full_input_ids":  full_input_ids,        # [50], long
                    "attention_mask":  full_attention_mask,   # [50], long
                    "prompt_len":      prompt_len             # int
                }

train_ds = HFImageCaptionIterableDataset(
    hf_split=train_hf,
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)
valid_ds = HFImageCaptionIterableDataset(
    hf_split=valid_hf,
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. COLLATE FUNCTION (NO FURTHER PADDING)
# ──────────────────────────────────────────────────────────────────────────────
def custom_collate_fn(batch):
    pixel_values = torch.stack([ex["pixel_values"] for ex in batch])         # [B,3,224,224]
    full_input_ids = torch.stack([ex["full_input_ids"] for ex in batch])     # [B,50]
    attention_masks = torch.stack([ex["attention_mask"] for ex in batch])    # [B,50]
    prompt_lens = torch.tensor([ex["prompt_len"] for ex in batch], device=DEVICE)  # [B]
    return {
        "pixel_values":    pixel_values,
        "full_input_ids":  full_input_ids,
        "attention_mask":  attention_masks,
        "prompt_len":      prompt_lens
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5. BUILD DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,      # IterableDataset is already “shuffled” at split time
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS
)

print(f"Train images: {len(train_hf)}  → {len(train_hf)*5} examples/epoch")
print(f"Valid images: {len(valid_hf)}  → {len(valid_hf)*5} examples/epoch")

# ──────────────────────────────────────────────────────────────────────────────
# 6. DEFINE VISION-LANGUAGE MODEL (frozen CLIP + LoRA TinyLlama + projector/gate)
# ──────────────────────────────────────────────────────────────────────────────
class SimpleVisionLanguageModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        # Freeze CLIP vision encoder
        self.vision_model = vision_model.vision_model
        for p in self.vision_model.parameters():
            p.requires_grad = False

        # LoRA-wrapped TinyLlama text model (fp16)
        self.text_model = text_model

        # Projector: 768 → projection_dim, IN FP32 to avoid FP16 overflow
        vision_hidden = self.vision_model.config.hidden_size  # 768
        text_hidden = self.text_model.config.hidden_size      # 4096
        self.projector = nn.Linear(
            vision_hidden,
            projection_dim,
            dtype=torch.float32,   # use FP32 here
            device=DEVICE
        )
        # Gate: projection_dim → text_hidden, IN FP32 to avoid FP16 overflow
        self.gate = nn.Linear(
            projection_dim,
            text_hidden,
            dtype=torch.float32,   # use FP32 here
            device=DEVICE
        )

    def forward(self, pixel_values, full_input_ids, attention_mask, prompt_len, labels=None):
        B, L_text_max = full_input_ids.size()  # should be [B, 50]

        # 6.1) Run CLIP (frozen) → get [CLS] embedding [B,768], fp16
        with torch.no_grad():
            vision_out = self.vision_model(pixel_values=pixel_values)
            cls_embed = vision_out.last_hidden_state[:, 0, :]  # [B,768], fp16

        # 6.2) Cast cls_embed to FP32 before projecting
        cls_embed_fp32 = cls_embed.to(torch.float32)

        # 6.3) Project (FP32) + Gate (FP32) → [B,4096], then cast to FP16
        vision_proj_fp32 = self.projector(cls_embed_fp32)  # [B,512], fp32
        gated_embed_fp32 = self.gate(vision_proj_fp32)     # [B,4096], fp32
        gated_embed = gated_embed_fp32.to(torch.float16)   # cast back to FP16

        # 6.4) Embed text tokens except last → [B,49,4096], fp16
        input_ids_trunc = full_input_ids[:, :-1]  # [B,49]
        text_embeds = self.text_model.get_input_embeddings()(input_ids_trunc)  # [B,49,4096], fp16

        # 6.5) Prepend vision token → [B,50,4096], fp16
        vision_token = gated_embed.unsqueeze(1)  # [B,1,4096], fp16
        combined_embeds = torch.cat([vision_token, text_embeds], dim=1)  # [B,50,4096], fp16

        # 6.6) Build combined attention mask → [B,50], long
        vision_attn = torch.ones((B, 1), device=DEVICE, dtype=attention_mask.dtype)
        combined_attn = torch.cat([vision_attn, attention_mask[:, :-1]], dim=1)

        # 6.7) Build labels if not provided (mask prompt, keep caption)
        if labels is None:
            labels = torch.full((B, L_text_max), fill_value=-100, dtype=torch.long, device=DEVICE)
            for i in range(B):
                P = prompt_len[i].item()
                total_nonpad = attention_mask[i].sum().item()
                raw_C = total_nonpad - P
                max_C = (L_text_max - 1 - P)
                C = min(raw_C, max(0, max_C))
                if C > 0:
                    start = 1 + P
                    end = 1 + P + C
                    labels[i, start:end] = full_input_ids[i, P : P + C]

        # 6.8) TinyLlama forward with combined_embeds (fp16) + combined_attn + labels
        outputs = self.text_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=labels
        )
        return outputs

model = SimpleVisionLanguageModel(clip_model, llama, projection_dim=512).to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# 7. OPTIMIZER, SCHEDULER, AND GRADIENT CLIPPING
# ──────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(model.projector.parameters()) +
    list(model.gate.parameters()) +
    list(model.text_model.parameters()),
    lr=LEARNING_RATE
)

num_train_examples = len(train_hf) * 5  # ~27,912 images × 5 captions
total_steps = num_train_examples * EPOCHS
warmup_steps = int(0.01 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAINING & VALIDATION LOOP WITH TQDM PROGRESS BARS + CHECKPOINT SAVING
# ──────────────────────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    # ─── (A) TRAINING PASS ─────────────────────────────────────────────────────
    model.train()
    total_train_loss = 0.0
    train_steps = 0

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} ‒ Training", leave=True) as pbar:
        for batch in pbar:
            pixel_vals = batch["pixel_values"]            # [1,3,224,224], fp16
            full_ids = batch["full_input_ids"]             # [1,50], long
            attn = batch["attention_mask"]                 # [1,50], long
            p_len = batch["prompt_len"]                    # [1], long

            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_vals,
                full_input_ids=full_ids,
                attention_mask=attn,
                prompt_len=p_len,
                labels=None
            )
            loss = outputs.loss

            # Check for NaN before backward; if found, skip this batch
            if torch.isnan(loss):
                print("Warning: NaN loss detected; skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

    avg_train_loss = total_train_loss / max(train_steps, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} → Avg Train Loss = {avg_train_loss:.4f}")

    # ─── (B) VALIDATION PASS ───────────────────────────────────────────────────
    model.eval()
    total_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        with tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} ‒ Validating", leave=True) as pbar_val:
            for batch in pbar_val:
                pixel_vals = batch["pixel_values"]
                full_ids = batch["full_input_ids"]
                attn = batch["attention_mask"]
                p_len = batch["prompt_len"]

                outputs = model(
                    pixel_values=pixel_vals,
                    full_input_ids=full_ids,
                    attention_mask=attn,
                    prompt_len=p_len,
                    labels=None
                )
                val_loss = outputs.loss

                if torch.isnan(val_loss):
                    print("Warning: NaN validation loss detected; skipping this example.")
                    continue

                total_val_loss += val_loss.item()
                val_steps += 1
                pbar_val.set_postfix_str(f"val_loss={val_loss.item():.4f}")

    avg_val_loss = total_val_loss / max(val_steps, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} → Avg Validation Loss = {avg_val_loss:.4f}\n")

    # ─── (C) SAVE CHECKPOINTS ───────────────────────────────────────────────────
    ckpt_dir = f"finetuned_qvlam_flickr30k_epoch{epoch+1}"
    os.makedirs(ckpt_dir, exist_ok=True)

    llama.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(model.projector.state_dict(), f"proj_epoch{epoch+1}.pt")
    torch.save(model.gate.state_dict(),      f"gate_epoch{epoch+1}.pt")

    print(f"Checkpoint saved:")
    print(f"  • {ckpt_dir}/  (LoRA TinyLlama + tokenizer)")
    print(f"  • proj_epoch{epoch+1}.pt")
    print(f"  • gate_epoch{epoch+1}.pt\n")

# ──────────────────────────────────────────────────────────────────────────────
# 9. FINAL SAVE
# ──────────────────────────────────────────────────────────────────────────────
final_dir = "finetuned_qvlam_flickr30k_final"
os.makedirs(final_dir, exist_ok=True)
llama.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
torch.save(model.projector.state_dict(), os.path.join(final_dir, "proj_final.pt"))
torch.save(model.gate.state_dict(),      os.path.join(final_dir, "gate_final.pt"))

print("✅ Training completed. Final checkpoints saved under:", final_dir)
