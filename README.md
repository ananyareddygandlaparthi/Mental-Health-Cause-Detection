# 🧠 Mental Health Cause Detection

A multi-class text classification project that identifies the underlying cause of mental health conditions from free-text input, using fine-tuned **BERT** and **XLNet** transformer models.

---

## 📌 Overview

Mental health issues such as anxiety, depression, loneliness, and suicidal thoughts are increasingly discussed in online communities. However, due to the vast volume of user-generated content, it is challenging to automatically identify the underlying causes or warning signs behind these struggles.

This project analyzes Reddit posts from key mental health subreddits — **r/anxiety**, **r/depression**, **r/mentalhealth**, **r/suicidewatch**, and **r/lonely** — to detect potential causes or risk factors contributing to mental health issues. Two transformer-based text classifiers (BERT and XLNet) were fine-tuned to:

- Automatically categorize posts based on underlying mental health causes
- Support early identification of risk factors through language patterns
- Contribute to improved understanding and digital mental health analysis

---

## 🏷️ Classification Labels

| Label | Description |
|---|---|
| `drug and alcohol` | Substance use as a contributing factor |
| `early life` | Childhood experiences and upbringing |
| `personality` | Personality traits and disorders |
| `trauma and stress` | PTSD, grief, anxiety, and stressful life events |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python |
| **Models** | BERT (`bert-base-uncased`), XLNet |
| **Frameworks** | PyTorch, Hugging Face Transformers |
| **Data Handling** | pandas, scikit-learn |
| **Training** | Jupyter Notebooks (`.ipynb`) |
| **Inference** | Python script (interactive CLI) |

---

## 📂 Project Structure

```
Mental-Health-Cause-Detection/
├── LDDA1.csv                  # Dataset: Drug & Alcohol
├── LDEL1.csv                  # Dataset: Early Life
├── LDPF1.csv                  # Dataset: Personality
├── LDTS1.csv                  # Dataset: Trauma & Stress
├── bert-classifier.ipynb      # BERT fine-tuning notebook
├── bert testing.py            # Interactive CLI for BERT inference
├── XLNet_classifier.ipynb     # XLNet fine-tuning notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU

### 1. Clone the repository

```bash
git clone https://github.com/ananyareddygandlaparthi/Mental-Health-Cause-Detection.git
cd Mental-Health-Cause-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔍 Running Inference (BERT)

The `bert testing.py` script lets you type any sentence and get an instant prediction:

```bash
cd "bert classifier"
python "bert testing.py"
```

**Update the model path** in the script before running:
```python
model_path = r"path/to/final_bert_model"
```

**Example:**
```
Enter a sentence: I've been struggling since losing my job and going through a divorce.
🔹 Predicted Label: trauma and stress
```

Type `exit` to quit.

---

## 📓 Training

Open either notebook to retrain or experiment:

- `bert classifier/bert-classifier.ipynb` — fine-tunes BERT on all four datasets
- `xlnet classifier/XLNet_classifier.ipynb` — fine-tunes XLNet on all four datasets

Both notebooks include data loading, preprocessing, training, evaluation, and model saving.

---

## 📊 Results & Findings

### Overall Accuracy Comparison

| Category | BERT Accuracy | XLNet Accuracy |
|---|---|---|
| Drug and Alcohol | 0.87 | 0.96 |
| Early Life | 0.80 | 0.76 |
| Personality | 0.72 | 0.85 |
| Trauma and Stress | 0.60 | 0.77 |
| **Overall** | **0.75** | **0.84** |

---

### XLNet Classifier — Classification Report

| Metric | Drug & Alcohol | Early Life | Personality | Trauma & Stress | Macro Avg | Weighted Avg |
|---|---|---|---|---|---|---|
| Precision | 0.88 | 0.79 | 0.87 | 0.81 | 0.84 | 0.84 |
| Recall | 0.96 | 0.76 | 0.85 | 0.77 | 0.83 | 0.84 |
| F1-Score | 0.91 | 0.77 | 0.86 | 0.79 | 0.83 | 0.84 |
| Support | 45 | 29 | 47 | 39 | 160 | 160 |
| **Accuracy** | | | | | | **0.84** |

The XLNet classifier outperformed BERT with an overall accuracy of **84%**. It demonstrated high precision and recall across all classes, particularly excelling in identifying Drug and Alcohol and Personality-related posts. The balanced macro and weighted averages suggest consistent performance across categories. XLNet was trained for 10 epochs before accuracy started declining without recovery, indicating overfitting.

---

### BERT Classifier — Classification Report

| Metric | Drug & Alcohol | Early Life | Personality | Trauma & Stress | Macro Avg | Weighted Avg |
|---|---|---|---|---|---|---|
| Precision | 0.97 | 0.82 | 0.58 | 0.67 | 0.76 | 0.75 |
| Recall | 0.87 | 0.80 | 0.72 | 0.60 | 0.75 | 0.75 |
| F1-Score | 0.92 | 0.81 | 0.64 | 0.63 | 0.75 | 0.76 |
| Support | 45 | 40 | 40 | 40 | 165 | 165 |
| **Accuracy** | | | | | | **0.75** |

The BERT classifier achieved an overall accuracy of **75%**, with particularly strong performance on the Drug and Alcohol and Early Life categories. Slightly lower F1-scores were observed for Personality and Trauma and Stress, likely due to overlapping linguistic cues between these classes. BERT was trained for 5 epochs before showing signs of overfitting — at 10 epochs it showed significant overfitting with a training loss of ~0.01 while the test loss increased to 1.4.

---

### 🏁 Conclusion

Both BERT and XLNet were evaluated as transformer-based classifiers for identifying causes of mental health issues from Reddit posts. **XLNet outperformed BERT**, achieving 84% overall accuracy vs. BERT's 75%, with more consistent per-class performance — especially on Personality and Trauma and Stress.

The superior performance of XLNet is attributed to its **permutation-based language modeling**, which captures bidirectional context more effectively than BERT, enabling better understanding of nuanced language in mental health discussions. These results highlight the potential of transformer-based models to accurately identify underlying mental health causes from textual data.

---

