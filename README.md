# AI-Powered Debate System

## One-line summary
A deep learning–based AI system that analyzes debate arguments and predicts key properties such as **sentiment** and **engagement score** using synthetic debate data (~100k samples). I experimented with multiple deep learning architectures (Dense, GRU, BiLSTM, Conv1D) and achieved up to **91% accuracy**.

---

## Problem statement
Debate data is difficult to obtain due to copyright, privacy, and ethics restrictions. I designed an AI system that takes a debate statement (PeoplePointOfView) and predicts useful debate characteristics such as **Sentiment** and **EngagementScore**, which can help users analyze arguments, detect tone, and measure impact.

To avoid using restricted or copyrighted debate content, I generated a large, safe **synthetic dataset** using ChatGPT.

---

## Dataset
I created a **synthetic dataset of ~100,000 debate entries** using ChatGPT prompts. This allows safe experimentation with NLP tasks without relying on real or copyrighted debate content.

Each row in the dataset contains:

- **Topic** – main debate domain (e.g., Education, Technology, Politics)
- **Subtopic** – more specific theme under the topic
- **PeoplePointOfView** – argument/opinion text from a human speaker
- **Sentiment** – Positive / Negative / Neutral
- **EngagementScore** – numerical score representing how impactful the argument is

A **small sample (`sample_debate_data.csv`)** is included for reproducibility.  
The full synthetic dataset can be regenerated using the notebook.

---

## Why Deep Learning? (ML → DL Transition)
I initially tested classical ML models:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 92% |
| SVM | 92% |
| Random Forest | 92% |

Even though ML models gave high accuracy, they were **not stable** and struggled to learn semantic patterns from longer text sequences.

So I switched to deep learning models, which understand sequence structure better. The DL models performed more naturally on synthetic language.

---

## Models Used (Deep Learning)

### **1) Simple Dense Neural Network**
- Vectorization: TF-IDF / Embedding
- **Accuracy: 91%**
- Fastest & most stable baseline DL model  
- Used for the Streamlit demo (best performer)

### **2) GRU (Gated Recurrent Unit)**
- **Accuracy: 91%**
- Best sequence model for this dataset  
- Low overfitting, handles synthetic patterns well

### **3) Bidirectional LSTM**
- **Accuracy: 90%**
- Good recall  
- Slightly slower and more prone to overfitting

### **4) Conv1D + MaxPooling**
- **Accuracy: 89%**
- Good at local text patterns  
- Fast but less sensitive to global context

### **5) BERT (Transformer Fine-Tuning)**
- **Accuracy: 26%**
- Underperformed due to:  
  - synthetic nature of text (less rich than real human language)  
  - insufficient hyperparameter tuning  
  - transformer sensitivity to learning rate, warmup, and training steps  

Transformers need higher-quality natural text + GPU-level fine-tuning.  
For this project, simpler DL models generalized better.

---

## Final Results (Deep Learning Summary)

| Model                  | Accuracy |
|------------------------|---------:|
| **Dense Neural Network** | **91%** |
| **GRU**                  | **91%** |
| Bidirectional LSTM       | 90%      |
| Conv1D + MaxPooling      | 89%      |
| BERT (fine-tuned)        | 26%      |

The **best performing approach (Dense/GRU)** is saved in the `models/` folder and used in the Streamlit demo.

---

## Project Structure

