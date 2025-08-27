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

## License
Add your repository license (e.g., MIT). Respect the dataset/model licenses and API terms.
