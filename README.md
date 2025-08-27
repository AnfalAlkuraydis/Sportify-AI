# 🥗 Sportify AI — Recipe Planner & GPT‑2 Fine‑Tuning Demo

Sportify AI is a hands‑on notebook/script that **fine‑tunes GPT‑2 on cooking instructions** and combines it with a small **meal‑planning UI**. The UI estimates daily calories from basic inputs (age, weight, height, activity, goal), asks an LLM to suggest a menu, and then uses your **fine‑tuned GPT‑2** to generate step‑by‑step recipes for the selected meal.

> Built with: PyTorch • 🤗 Transformers/Datasets • ipywidgets • Matplotlib • OpenAI API

---

## ✨ Features

- **Quick inference with base GPT‑2** to generate recipes from prompts.
- **Fine‑tuning GPT‑2** on the public `darkraipro/recipe-instructions` dataset.
- **Perplexity check & bar chart** comparing base vs. fine‑tuned model.
- **Lightweight UI (ipywidgets)** to calculate calories and generate a **daily meal plan**.
- **Two‑stage recipe flow**: plan with Chat Completions → detailed recipe with your **fine‑tuned GPT‑2**.
- **Hugging Face Hub integration** to push and later reload the fine‑tuned model.

---

## 📦 Requirements

> Python 3.9+ and a CUDA‑enabled GPU are recommended for training.

Install the core dependencies:

```bash
pip install -U torch transformers datasets accelerate huggingface_hub openai ipywidgets matplotlib
```

If you run in Jupyter, enable widgets (once per environment):

```bash
jupyter nbextension enable --py widgetsnbextension
```

> The project is notebook‑style; running it in **Jupyter/Colab** is the smoothest experience.

---

## 🔐 API Keys (OpenAI)

The UI’s meal plan step calls the OpenAI Chat Completions API. Set your key via an environment variable and initialize the client from that variable instead of hard‑coding it:

```python
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

**Never commit your key** to version control.

---

## 🗂️ Project Layout (minimal)

```
.
├── sportify_ai.py           # The notebook-style script (originally from Colab)
└── README.md
```

> Your uploaded file may be named slightly differently (e.g., `Sportify AI.py`).

---

## 🚀 Quick Start (Inference Only)

Generate a simple recipe with base GPT‑2:

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

prompt = "Give me recipe instructions to make a cake"
enc = tok(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=180, do_sample=True, top_p=0.92, temperature=0.8,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)

gen_only = out[0, enc["input_ids"].shape[-1]:]
print(tok.decode(gen_only, skip_special_tokens=True).strip())
```

---

## 🧪 Fine‑Tuning Guide

This project demonstrates fine‑tuning GPT‑2 on **recipe instructions**:

1. **Dataset**
   - `darkraipro/recipe-instructions` via 🤗 Datasets.

2. **Tokenizer & Model**
   - GPT‑2 with `pad_token = eos_token` and aligned `pad_token_id` on the model config.

3. **Trainer Setup**
   - `TrainingArguments` with gradient accumulation and optional fp16 on CUDA.
   - `DataCollatorForLanguageModeling(mlm=False)` for causal LM.

4. **Train**
   - `num_train_epochs=5`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` (tune as needed).

5. **Save & (Optionally) Push**
   - Save to `gpt2-model/` and optionally `push_to_hub()` after `huggingface_hub` login.

> **Tip:** If you see OOM (out‑of‑memory), reduce `per_device_train_batch_size`, shorten `max_length`, or increase `gradient_accumulation_steps`.

---

## 📊 Perplexity Check

The script computes and plots perplexity **before** vs **after** fine‑tuning to give a quick quality signal. Lower is better.

---

## 🧩 The Meal‑Plan → Recipe Flow

1. **Enter inputs**: age, weight (kg), height (cm), activity (Low/Moderate/High), goal (Gain/Loss/Maintain).
2. **Click “Generate Meal Plan”**: uses the OpenAI Chat Completions API to propose a daily plan.
3. **Pick a meal** from the dropdown and **Generate Recipe**: your **fine‑tuned GPT‑2** expands it into detailed instructions.

> If Chat Completions returns an unexpected format, the script falls back to a couple of default meal names.

---

## ☁️ Push & Reload from Hugging Face

```python
from huggingface_hub import notebook_login
notebook_login()  # one-time
model.push_to_hub("YOUR_USERNAME/gpt2-model")
```

Then load it anywhere:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("YOUR_USERNAME/gpt2-model", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/gpt2-model")
```

---

## 🛠️ Troubleshooting

- **`pad_token` issues**: Always set `tokenizer.pad_token = tokenizer.eos_token` for GPT‑2 and align `model.config.pad_token_id`.
- **Long training time**: Use a smaller subset, fewer epochs, or a smaller model (e.g., `distilgpt2`).
- **Rate limits / API errors**: Catch and print exceptions around the OpenAI call; ensure the `OPENAI_API_KEY` is set.
- **Widgets not visible**: Ensure Jupyter widgets are enabled and you’re in a notebook context.

---

## ✅ License & Attribution

- Dataset: `darkraipro/recipe-instructions` (see dataset license on Hugging Face).
- This project uses OpenAI and 🤗 Transformers/Datasets under their respective licenses.
- Add your preferred project license (e.g., MIT) at the repository root.

---

## 🙌 Acknowledgements

- 🤗 Hugging Face (Transformers, Datasets, Hub)
- OpenAI (Chat Completions API)
- PyTorch
- The dataset authors/maintainers

---

### Why this project?

It’s a compact, end‑to‑end example that blends **LLM fine‑tuning** with a **tiny, practical UI**—great for demos, teaching, and experimenting with recipe‑generation workflows.
