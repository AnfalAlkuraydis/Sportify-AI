# 🥗 Sportify AI — GPT‑2 Recipe Generator

> **Model:** [`anf1lll/gpt2-model`](https://huggingface.co/anf1lll/gpt2-model)  
> **Data:** `darkraipro/recipe-instructions` (Hugging Face Datasets)

---

## Abstract
Sportify AI is a lightweight recipe-generation project that fine‑tunes GPT‑2 on public cooking instructions and pairs it with a simple meal‑planning flow. A calorie estimate is used to propose a daily plan, and the fine‑tuned GPT‑2 produces detailed, step‑by‑step recipes from the selected meal. The goal is to demonstrate an end‑to‑end pipeline—data → fine‑tuning → evaluation → interactive use—while staying minimal and easy to reproduce.

---

## Overview
- **LLM:** GPT‑2 fine‑tuned on cooking instructions → [`anf1lll/gpt2-model`](https://huggingface.co/anf1lll/gpt2-model)  
- **Dataset:** `darkraipro/recipe-instructions` (short, imperative recipe steps)  
- **Flow:** (1) estimate calories, (2) propose a day plan, (3) generate chosen recipe with the fine‑tuned model  
- **Evaluation:** quick perplexity comparison (base vs. fine‑tuned)  
- **Use cases:** teaching, demos, and fast experimentation with recipe generation

---

## Pipeline
1. **Prepare data** → load `darkraipro/recipe-instructions` with 🤗 Datasets; filter/clean short imperative steps.  
2. **Tokenize** → GPT‑2 tokenizer with `pad_token = eos_token`.  
3. **Fine‑tune** → Causal LM training (`mlm=False`) using 🤗 `Trainer`.  
4. **Evaluate** → compute perplexity on a held‑out split; compare to base GPT‑2.  
5. **Serve/Use** → load [`anf1lll/gpt2-model`](https://huggingface.co/anf1lll/gpt2-model) for recipe generation in a tiny UI or script.

**ASCII sketch**  
```
User inputs → TDEE estimate → Day plan (LLM) → Pick a meal → anf1lll/gpt2-model → Step-by-step recipe
```

---

## Results
- **Quality signal:** Perplexity drops on recipe text after fine‑tuning (lower is better).  
- **Observed behavior:** More coherent, imperative, and on‑topic instructions vs. base GPT‑2.  

> Reproduce your own numbers with the snippet in **Reproduce Results** below.

---

## Installation
Python 3.9+ is recommended. For GPU training, install a CUDA build of PyTorch.

```bash
pip install -U torch transformers datasets accelerate huggingface_hub ipywidgets matplotlib
```

If using Jupyter:
```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## Quick Start (Inference)
Use the hosted model: **`anf1lll/gpt2-model`**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("anf1lll/gpt2-model", use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("anf1lll/gpt2-model").to(device).eval()

prompt = "Generate concise, step-by-step instructions to make chicken pasta"
enc = tok(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=180, do_sample=True, top_p=0.92, temperature=0.8,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Reproduce Results (Fine‑Tuning & PPL)
Minimal recipe (pseudo‑hyperparams; tune for your GPU):
```python
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2LMHeadModel,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments)
import math, torch

ds = load_dataset("darkraipro/recipe-instructions")
tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tok.eos_token_id

def tok_fn(ex):
    return tok(ex["text"], truncation=True, max_length=256)
tds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

collator = DataCollatorForLanguageModeling(tok, mlm=False)
args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=4, gradient_accumulation_steps=8,
    num_train_epochs=3, fp16=torch.cuda.is_available(), logging_steps=50, save_total_limit=1
)
trainer = Trainer(model=model, args=args, train_dataset=tds["train"],
                  eval_dataset=tds.get("validation", tds["train"].select(range(1000))),
                  data_collator=collator)

trainer.train()
metrics = trainer.evaluate()
ppl = math.exp(metrics["eval_loss"])
print("Perplexity:", round(ppl, 2))
```

> For a base comparison, run the `evaluate()` step with an **untrained** GPT‑2 checkpoint, then with the **fine‑tuned** checkpoint.

---

## Data
- **Dataset:** `darkraipro/recipe-instructions` on Hugging Face Datasets.  
  Short instructions aligned with recipe tasks; ideal for causal LM fine‑tuning.  
- **Preprocessing tips:** remove empty lines, keep imperative/step‑like text, limit max length (e.g., 256).

---

## Notes & Tips
- Set `tok.pad_token = tok.eos_token` and `model.config.pad_token_id` for GPT‑2.  
- If you hit OOM, lower `per_device_train_batch_size`, shorten `max_length`, or use `distilgpt2`.  
- Prefer GPU for training; CPU is fine for inference but slower.

---

## License
Add your repository license (e.g., MIT). Respect the dataset/model licenses and API terms.
