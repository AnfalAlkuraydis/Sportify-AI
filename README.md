# 🥗 Sportify AI — Clean Script

A minimal, production-ready script to **fine-tune GPT‑2** on recipe-style text and **generate** meal instructions.

## Quickstart

```bash
# 1) Create a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train (uses 10k examples of recipe instructions by default)
python sportify_ai.py train --dataset darkraipro/recipe-instructions --output_dir out/ft-gpt2-recipe --fp16

# 4) Inference
python sportify_ai.py infer --model out/ft-gpt2-recipe --prompt "Give me a high-protein dinner recipe with salmon and quinoa" --max_new_tokens 180
```

## Notes
- The script **guesses** the text column if not provided; use `--text_column` to be explicit.
- Perplexity is printed after evaluation (`eval_loss` → `exp(loss)`).
- No secrets are required. This project **does not** use OpenAI keys.
- You can switch base models with `--base_model` (e.g., `gpt2-medium`), if GPU memory allows.
