# Arabic News Documents Topic Modeling

This project automatically identifies key topics in Arabic news articles using natural language processing and machine learning. It tackles common challenges such as Arabic morphology, dialectal variation, and scaling text analysis to large datasets.

---

## Project Overview

We use a labeled dataset of 111,728 Arabic news documents, grouped into five categories:

- **Culture**: 13,738 documents  
- **Diverse**: 16,728 documents  
- **Economy**: 14,235 documents  
- **Politics**: 20,505 documents  
- **Sports**: 46,522 documents  

Each category is assigned a numeric label:
- `0`: Culture  
- `1`: Diverse  
- `2`: Economy  
- `3`: Politics  
- `4`: Sports  

The dataset is split into 80% training and 20% testing, with stratification to preserve class distribution.

---

## Key Features

### Text Preprocessing
- Removal of Arabic-specific punctuation  
- Stopword filtering using NLTK’s Arabic stopword list  
- Root extraction via the ISRI Arabic Stemmer  

### Feature Engineering
- TF-IDF vectorization limited to the top 5,000 features  
- Stratified train-test splitting to maintain class balance  

### Machine Learning Pipeline
- Logistic Regression classifier trained with `max_iter=1000`  
- Model saved using Python’s `pickle` for reuse  

### Performance Metrics
- **Overall accuracy**: 95.47%  
- Per-class breakdown:

| Category   | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Culture    | 0.96      | 0.94   | 0.95     |
| Diverse    | 0.95      | 0.97   | 0.96     |
| Economy    | 0.91      | 0.90   | 0.90     |
| Politics   | 0.91      | 0.91   | 0.91     |
| Sports     | 0.99      | 0.99   | 0.99     |

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/AbdelkadirSellahi/arabic-news-topic-modeling.git  
cd arabic-news-topic-modeling
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
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

## Key Insights

- Sports articles were classified with the highest accuracy (F1-score: 0.99).  
- Economy and politics showed more ambiguity, with lower but still strong performance.  
- The model handles dialectal differences reasonably well.  
- Most confusion occurs between Economy and Politics categories, as reflected in the confusion matrix.

---

## Technical Approach

### Data Exploration
- Visualized class distributions  
- Analyzed document length across categories  
- Manually reviewed sample texts per class  

### Preprocessing Pipeline
```
Raw Text → Punctuation Removal → Stopword Filtering → Arabic Stemming → TF-IDF Vectorization
```

### Model Selection
- Logistic Regression was selected after initial testing against alternatives.  
- Class imbalance was addressed through stratified sampling rather than resampling techniques.

---

## Contribution Guide

1. Fork the repository  
2. Create a feature branch  
3. Test your changes locally  
4. Commit and push  
5. Open a pull request  

---

## Contact

If you have questions, suggestions, or want to collaborate, feel free to open an issue or reach out directly.

**Author**: Abdelkadir Sellahi  
**Email**: abdelkadirsellahi@gmail.com  
**GitHub**: [Abdelkadir Sellahi](https://github.com/AbdelkadirSellahi)

---

## Acknowledgments

- Dataset: [Arabic News Dataset](https://data.mendeley.com/datasets/v524p5dhpj/2)  
- Core libraries: NLTK, scikit-learn, pandas, matplotlib  
- Stemming tool: ISRI Arabic Stemmer  
