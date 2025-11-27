# AI-Powered Debate System

## One-line summary
AI system that generates rebuttals and suggests debate points from an input debate prompt using NLP models (baseline TF-IDF + Logistic Regression, and experiments with LSTM/BERT). Demo available via Streamlit.

## Problem statement
Given an opponent’s statement or debate prompt, automatically produce structured rebuttals and supporting counter-arguments to help users prepare for debates and public speaking.

## Dataset
This project uses a **synthetic debate dataset** generated using ChatGPT to safely create large-scale debate-related text. Real debate datasets are limited or restricted due to copyright and privacy issues, so synthetic data was chosen for controlled, ethical experimentation.

The dataset contains **~100,000 rows**, with the following columns:

- **Topic** – main debate theme (e.g., politics, education, technology)
- **Subtopic** – more specific debate angle within the Topic
- **PeoplePointOfView** – short argument or opinion given by a simulated person
- **Sentiment** – classification of the argument (Positive / Negative / Neutral)
- **EngagementScore** – numeric score representing how engaging or impactful the argument is

I include a small sample file (`data/sample_debate_data.csv`) for reproducibility.  
The full dataset is large, so only a preview/sample is stored in the repository.


Using synthetic data allows the model to learn diverse debate patterns without exposing private or copyrighted data.


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

I trained and evaluated multiple models for the debate classification task. Below are the measured test accuracies for each approach (same dataset and test split):

- **Simple Dense (feed-forward neural network)** — **91%** (best baseline)
- **GRU (Gated Recurrent Unit)** — **91%**
- **Bidirectional LSTM** — **90%**
- **Conv1D + Pooling** — **89%**
- **BERT (pretrained transformer fine-tuned)** — **26%**

**Notes / interpretation**
- The best performing models were the Simple Dense and GRU (91% accuracy). I saved the best-performing model file under `models/best_model.pkl` (or `models/best_model.pt` if PyTorch).
- BERT produced very low accuracy (26%) on my experiments. Possible reasons and details are documented below (see *BERT notes*).

### BERT notes (why low performance)
- **Data size & fine-tuning mismatch**: BERT has many parameters. Although it *can* be fine-tuned on small datasets, in practice good BERT fine-tuning needs careful hyperparameter tuning (low learning rates, warmup, weight decay), and enough labelled examples per class. My dataset size (<< 100k samples) and current fine-tuning setup resulted in poor generalization.
- **Training/Hyperparameter issues**: Transformers are sensitive to learning rate, batch size, number of epochs, and warmup steps. Using default settings or too high a learning rate can cause BERT to fail to converge.
- **Possible class imbalance or label noise**: BERT magnifies issues if classes are imbalanced or labels noisy — cleaning/augmentation helps.
- **Compute/epochs**: Proper BERT fine-tuning often needs more GPU steps/epochs and early stopping. On a simple CPU/short runs it may underperform simpler neural nets trained longer.
- **Actionable next steps**: try frozen/partial fine-tuning (freeze lower layers), use a lighter transformer (DistilBERT), tune learning rate (e.g., 2e-5 to 5e-5 with warmup), perform stratified k-fold cross-validation, or augment data.


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
