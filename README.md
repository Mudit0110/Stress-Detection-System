# Stress Detection System  

An NLP-based project that detects stress in Reddit posts using transformer models and machine learning. The system classifies posts as Stressed or Not Stressed, with added features like confidence scores and keyword highlighting for interpretability.  

---

## Features  
- Achieves up to 98% accuracy with BERT.  
- Fine-tuned transformer models: BERT, RoBERTa, DistilBERT, ClinicalBERT.  
- Expanded dataset from ~2.8K to ~11K posts using T5-based paraphrasing.  
- Benchmarked against traditional ML models such as SVM, Random Forest, and Naive Bayes.  

---

## Dataset  
- Source: Reddit posts (Kaggle Dreaddit dataset).  
- Original size: ~2,838 posts.  
- After augmentation: ~11,000 posts.  
- Labels:  
  - 1 → Stressed  
  - 0 → Not Stressed  

---

## Results  

### Machine Learning Models  

| Model                | Accuracy (%) |  
|----------------------|--------------|  
| Logistic Regression  | 87.8         |  
| Decision Tree        | 88.0         |  
| Random Forest        | 95.6         |  
| Support Vector Machine (SVM) | 96.7 |  
| Naive Bayes          | 85.3         |  

### Transformer Models  

| Model        | Accuracy (%) | Precision | Recall | F1-score |  
|--------------|--------------|-----------|--------|----------|  
| DistilBERT   | 96.8         | 0.956     | 0.987  | 0.971    |  
| **BERT**     | **98.0**     | 0.974     | 0.990  | 0.982    |  
| RoBERTa      | 96.4         | 0.953     | 0.982  | 0.967    |  
| ClinicalBERT | 96.6         | 0.960     | 0.977  | 0.968    |  

---

## Tech Stack  
- Languages: Python  
- Libraries: Hugging Face Transformers, Scikit-learn, PyTorch, NLTK, NLPaug, Pandas, Matplotlib  
- Deployment: Streamlit

## License  
This project is licensed under the MIT License. See the LICENSE file for details.  

## Author  
**Mudit Shrivastava**  
Email: MUDIT.SHRI2002@GMAIL.COM


