# 🌐 Arabic News Documents Topic Modeling

This project focuses on **automatically identifying key topics** in Arabic news documents using **Natural Language Processing (NLP)** and **machine learning**. It addresses challenges like Arabic morphology, dialect variations, and large-scale text analysis.

---

## 🚀 Project Overview

We utilize a **multi-class Arabic news dataset** containing 111,728 documents across 5 categories:
- **Culture (13,738)**
- **Diverse (16,728)**
- **Economy (14,235)**
- **Politics (20,505)**
- **Sports (46,522)**

Documents are labeled as:
- `0`: Culture
- `1`: Diverse
- `2`: Economy
- **`3`: Politics**
- **`4`: Sports**

The dataset is split into **80% training** and **20% testing** with class balance maintained.

---

## 🔧 Key Features

1. **Advanced Text Preprocessing:**
   - Arabic punctuation removal
   - Stopword filtering using NLTK's Arabic corpus
   - Root extraction with ISRI Arabic Stemmer

2. **Feature Engineering:**
   - TF-IDF vectorization with **5,000 maximum features**
   - Stratified train-test splitting

3. **Machine Learning Pipeline:**
   - Logistic Regression classifier with `max_iter=1000`
   - Model serialization using `pickle`

4. **Performance Metrics:**
   - **Overall Accuracy:** 95.47%
   - Detailed class-wise metrics:

| Category   | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Culture    | 0.96      | 0.94   | 0.95     |
| Diverse    | 0.95      | 0.97   | 0.96     |
| Economy    | 0.91      | 0.90   | 0.90     |
| Politics   | 0.91      | 0.91   | 0.91     |
| Sports     | 0.99      | 0.99   | 0.99     |

---

## 🖥️ How to Use

### 1. Clone Repository
```bash
git clone https://github.com/AbdelkadirSellahi/arabic-news-topic-modeling.git
cd arabic-news-topic-modeling
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook Topic_Modeling_based_on_Arabic_news_Documents.ipynb
```

### 4. Make Predictions
```python
from utils import predict_topic

sample_news = "Your Arabic news text here..."
predicted_category = predict_topic(sample_news)
print(f"Predicted Category: {predicted_category}")
```

---

## 📊 Key Insights

- Sports articles were easiest to classify (F1=0.99)
- Economic/political content showed more classification complexity
- Model handles dialectal variations effectively
- Confusion matrix reveals most confusion between Economy/Politics categories

---

## 📄 Technical Approach

1. **Data Exploration:**
   - Class distribution visualization
   - Text length analysis
   - Sample inspection per category

2. **Preprocessing Pipeline:**
   ```mermaid
   graph LR
   A[Raw Text] --> B[Punctuation Removal]
   B --> C[Stopword Filtering]
   C --> D[Arabic Stemming]
   D --> E[TF-IDF Vectorization]
   ```

3. **Model Selection:**
   - Logistic Regression outperformed alternatives in initial tests
   - Handled class imbalance through stratified sampling

---

## 🤝 Contribution Guide

1. **Fork the Repository**
2. **Create Feature Branch**
3. **Test Your Changes**
4. **Commit and Push**
5. **Open Pull Request**

---

## 💬 **Contact**

Feel free to open an issue or reach out for collaboration!  

**Author**: *Abdelkadir Sellahi*

**Email**: *abdelkadirsellahi@gmail.com* 

**GitHub**: [Abdelkadir Sellahi](https://github.com/AbdelkadirSellahi)

---

## 🙏 Acknowledgments

- Dataset: [Arabic News Dataset](https://data.mendeley.com/datasets/v524p5dhpj/2)
- Core Libraries: NLTK, scikit-learn, pandas, matplotlib
- Stemming: ISRI Arabic Stemmer

---

💡 **Pro Tip:** Try the pretrained model with news from different Arabic dialects! For best results, ensure text contains at least 50 words.
