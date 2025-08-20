# Stress Detection System  

An **NLP-based system** for detecting stress from Reddit posts using **transformer models** and **machine learning techniques**. The project focuses on mental health monitoring by classifying posts into **Stressed** and **Not Stressed**, while also providing interpretability features like **keyword highlighting** and **confidence scoring**.  

---

## 📌 Features  
- **High-Accuracy Stress Detection** – Achieved up to **98% accuracy** using BERT.  
- **Transformer Models** – Fine-tuned **BERT, RoBERTa, DistilBERT, ClinicalBERT** on stress-labeled data.  
- **Data Augmentation** – Expanded dataset from ~2.8K to ~11K posts using **T5-based paraphrasing** (NLPaug).  
- **Interpretability** – Displays **confidence scores** and highlights **stress-indicative words**.  
- **Comparative Evaluation** – Benchmarked traditional ML (SVM, Random Forest, Naive Bayes, etc.) vs transformer models.  
- **UI Demo (Streamlit)** – Simple interface for real-time predictions and mental health tips.  

---

## 📊 Dataset  
- Source: **Reddit posts** (via Kaggle’s Dreaddit dataset).  
- Size: ~2,838 original samples → ~11,000 samples after augmentation.  
- Labels:  
  - `1` → Stressed  
  - `0` → Not Stressed  

---

## ⚙️ Project Workflow  

1. **Data Preprocessing**  
   - Lowercasing, stopword removal, stemming, punctuation/URL cleaning.  
   - Outlier and duplicate removal for high-quality data.  

2. **Data Augmentation**  
   - Applied **T5-based paraphrasing** twice to increase dataset diversity.  

3. **Model Training**  
   - **Traditional ML**: Logistic Regression, Random Forest, SVM, Naive Bayes.  
   - **Transformers**: BERT, RoBERTa, DistilBERT, ClinicalBERT (fine-tuned with Hugging Face).  

4. **Evaluation**  
   - Metrics: **Accuracy, Precision, Recall, F1-score**.  
   - Visualizations: Confusion Matrix, Word Clouds, Label Distribution.  

---

## 🏆 Results  

### Machine Learning Models  
| Model              | Accuracy |  
|--------------------|----------|  
| Logistic Regression | 87.8% |  
| Decision Tree       | 88.0% |  
| Random Forest       | 95.6% |  
| SVM                 | 96.7% |  
| Naive Bayes         | 85.3% |  

### Transformer Models  
| Model        | Accuracy | Precision | Recall | F1-score |  
|--------------|----------|-----------|--------|----------|  
| DistilBERT   | 96.8% | 0.956 | 0.987 | 0.971 |  
| BERT         | **98.0%** | 0.973 | 0.989 | 0.981 |  
| RoBERTa      | 96.4% | 0.953 | 0.981 | 0.967 |  
| ClinicalBERT | 96.6% | 0.959 | 0.976 | 0.968 |  

✅ **BERT achieved the best performance with 98% accuracy.**  

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Libraries & Frameworks:** Hugging Face Transformers, NLTK, NLPaug, Scikit-learn, PyTorch, Pandas, Matplotlib  
- **Deployment:** Streamlit, GitHub  
- **Version Control:** Git  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/YourUsername/Stress-Detection-System.git
   cd Stress-Detection-System
