# Statlog Vehicle Silhouettes – Machine Learning Model Comparison

## Problem Statement

Dataset: **Statlog Vehicle Silhouettes**  
Source: https://www.kaggle.com/datasets/patriciabrezeanu/statlog-vehicle-silhouettes

The objective of this project is to classify a given vehicle silhouette into one of four vehicle types using features extracted from the silhouette.
The vehicle may be viewed from different angles. This is a **multi-class classification problem**, where each vehicle must be classified into one of the following categories:

- Bus  
- Opel  
- Saab  
- Van  
---

## Dataset Description

- **Number of instances:** 846  
- **Number of features:** 18 numerical attributes  
- **Target variable:** Vehicle class (4 classes)  
- **Problem type:** Multi-class classification  
- **Feature type:** Continuous numerical values  

The dataset contains geometric and shape-based features extracted from vehicle silhouettes.  
The objective is to correctly classify each vehicle based on these measurements.
---

## Project Links

- **GitHub Repository:**  
  https://github.com/tusharmulkar/ml-model-app/

- **Streamlit App:**  
  https://ml-model-app-tusharmulkar.streamlit.app/

---

## Models Used

The following machine learning models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble Boosting)

---

## Model Comparison Table

| Model                | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression  | 0.6850   | 0.8715 | 0.6852    | 0.6889 | 0.6679 | 0.5918 |
| Decision Tree        | 0.7244   | 0.8154 | 0.7216    | 0.7229 | 0.7198 | 0.6335 |
| KNN                  | 0.7165   | 0.9211 | 0.7062    | 0.7171 | 0.7087 | 0.6239 |
| Naive Bayes          | 0.5118   | 0.7914 | 0.5650    | 0.5205 | 0.4906 | 0.3780 |
| Random Forest        | 0.7165   | 0.9418 | 0.6973    | 0.7172 | 0.7035 | 0.6242 |
| XGBoost              | 0.7717   | 0.9476 | 0.7640    | 0.7726 | 0.7668 | 0.6963 |

---

## Observations

### Logistic Regression
Achieved moderate performance with an accuracy of **68.5%**. The relatively high AUC indicates good class separation. However, the lower F1 score suggests that the model is not able to effectively capture non-linear patterns in the dataset. It serves as a strong baseline model.

### Decision Tree
Improved performance over Logistic Regression, achieving **72.44% accuracy**. It effectively captures nonlinear relationships within the dataset. However, its AUC (0.8154) is lower than other models, indicating comparatively weaker class discrimination.

### KNN
Both AUC and Accuracy are high, making it a strong candidate for this dataset. The high AUC (0.9211) indicates excellent class separation capability.

### Naive Bayes
Naive Bayes showed the weakest performance among all models. This is likely due to the strong independence assumption between features, which does not hold for this dataset where features are correlated.

### Random Forest (Ensemble)
Achieved strong and stable performance, indicating reliable overall classification. Ensemble bagging improves robustness compared to a single Decision Tree.

### XGBoost (Ensemble Boosting)
XGBoost achieved the best overall performance with:
- **Accuracy:** 77.17%
- **F1-score:** 0.7668
- **MCC:** 0.6963
- **AUC:** 0.9476 (highest among all models)

Boosting allows the model to sequentially correct errors from previous trees, resulting in superior performance. It is the most suitable model for this dataset.

---

## Conclusion

Among all evaluated models, **XGBoost demonstrated the best overall performance** across all evaluation metrics. The results indicate that the Statlog Vehicle Silhouettes dataset benefits from ensemble-based methods, particularly boosting techniques, which effectively capture complex nonlinear patterns and feature interactions.
