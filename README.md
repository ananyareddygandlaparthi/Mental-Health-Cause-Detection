# Mental-health-cause-detection

This project focuses on fine-tuning **BERT** and **XLNet** transformer models on a **labeled Reddit mental health dataset** to classify user posts into categories such as **Depression**, **Anxiety**, **Personality**, and more.

We use **HuggingFace Transformers**, **PyTorch**, and **scikit-learn** to train, evaluate, and compare the models’ performance.  
The goal is to determine which transformer performs better for mental health-related text classification.

## Dataset Description

- The dataset is a **Reddit mental health dataset** consisting of posts labeled under different categories:
  - **DA** – Depression & Anxiety  
  - **EL** – Emotional Loneliness  
  - **PF** – Psychosocial Functioning  
  - **TS** – Trauma & Stress  

Each entry mainly contains:
- `title` – post title  
- `selftext` – body of the Reddit post  
- `Label` – assigned category  

The final dataset is created by merging all CSV files and cleaning missing values.

---

How to Run the Code: Fine-Tuning BERT
- Open bert-classifier.ipynb in Jupyter or Colab.
- Run all cells sequentially.
- The notebook will load and preprocess the labeled dataset.
- Tokenize using bert-base-uncased.
- Train and validate the model on an 80:20 split.
- Display metrics like accuracy, F1 score, and confusion matrix.
- Zip best model file and download
- Unzip file and run the bert testing file after updating path of best model folder in code

How to Run the Code: Fine-Tuning XLNet
- Open xlnet-classifier.ipynb.
- Run all cells in order.
- The notebook will load the dataset and encode labels.
- Tokenize using xlnet-base-cased.
- Fine-tune the model for text classification.
- Save the best-performing model (best_model_xlnet.pth).
- Display validation metrics, loss curves, and confusion matrix.

For both models, we evaluate using:
- Accuracy
- Precision, Recall, F1 Score (macro, micro, weighted)
- Confusion Matrix (visualized)
- Training & Validation Loss Curves

