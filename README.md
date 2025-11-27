# AI-Powered Debate System

## One-line summary
AI system that generates rebuttals and suggests debate points from an input debate prompt using NLP models (baseline TF-IDF + Logistic Regression, and experiments with LSTM/BERT). Demo available via Streamlit.

## Problem statement
Given an opponent’s statement or debate prompt, automatically produce structured rebuttals and supporting counter-arguments to help users prepare for debates and public speaking.

## Dataset
- Included sample: `data/sample_debate_data.csv` (small sample for reproduction)
- Full dataset: (if used) add link here or mention source
- Example columns: `topic`, `argument_text`, `label`, `response_text`

## Approach / Pipeline
1. Data cleaning & preprocessing (tokenization, lowercasing, stopword removal)
2. Feature representation: TF-IDF and Transformer embeddings (BERT)
3. Models tried:
   - Baseline: TF-IDF + Logistic Regression
   - Sequence models: LSTM / GRU
   - Transformer-based classifier (BERT)
4. Evaluation:
   - Classification: Accuracy, Precision, Recall, F1
   - (If generation used) BLEU / ROUGE

## Results (summary)
- Baseline (TF-IDF + LR): Accuracy = **xx%** (replace with your result)
- Best model: BERT finetune (if used) — Accuracy = **xx%** (replace)
- Example: Input → Suggested reply (see `images/demo_screenshot.png`)

## Run locally
```bash
git clone https://github.com/thesiddyyy/AI-Powered-Debate-System.git
cd AI-Powered-Debate-System
python -m venv venv
# Windows:
venv\Scripts\activate
# mac / linux:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
