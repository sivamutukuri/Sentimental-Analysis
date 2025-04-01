# Sentiment Analysis Project

# Overview

This project performs sentiment analysis on text data using machine learning techniques. It preprocesses textual data, extracts features using TF-IDF, and employs deep learning models with PyTorch to classify sentiments.

# Features

Text preprocessing (tokenization, stopword removal, lemmatization)

Feature extraction using TF-IDF

Sentiment classification using machine learning models

Data visualization using Seaborn and Matplotlib

# Technologies Used

Python (for scripting and data handling)

NLTK & spaCy (for natural language processing)

Scikit-learn (for feature extraction and ML model training)

PyTorch (for deep learning-based sentiment analysis)

Pandas & NumPy (for data manipulation)

Matplotlib & Seaborn (for visualization)

#EDA

![image](https://github.com/user-attachments/assets/aa6d604c-0464-4d02-9d46-497d48412e7c)

The data suggests that certain countries have tweets primarily occurring during a specific time of the day.

![image](https://github.com/user-attachments/assets/b41739e9-1ab9-401a-aa49-776e5c758624)

* This chart displays the relationship between user age groups and the time they post tweets.

* The x-axis represents different age groups (0-20, 21-30, 31-45, 46-60, 60-70, 70-100).

* The y-axis represents the count of tweets.

* Different colors signify different times of the day (morning, noon, night).

* The distribution appears uniform, suggesting similar tweet activity across all age groups for each time slot.


# Ml Models

Models Used:

* Logistic Regression
* Random Forest Classifier
* Naive Bayes (MultinomialNB)
* K-Nearest Neighbors (KNN)
* XGBoost
* LightGBM
  
# Model Performance & Results

* Model	Macro-averaged F1 Score
* Logistic Regression	0.76
* Random Forest	0.77
* Naive Bayes	0.71
* K-Nearest Neighbors	0.68 (with n_neighbors=7)
* XGBoost	0.77
* LightGBM	0.78
* Best Performing Model (Based on F1-Score)

Considering only the Macro-averaged F1 score, LightGBM emerges as the best-performing model with a score of 0.78. Random Forest and XGBoost are close contenders with an F1-score of 0.77.

# Why Focus on F1-Score?

Balances Precision and Recall: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance, especially when dealing with imbalanced datasets.
Relevance to Sentiment Analysis: In sentiment analysis, you often care about correctly identifying both positive and negative sentiments (precision and recall). The F1-score captures this balance well.
Robustness to Class Imbalance: Macro-averaged F1, in particular, is less sensitive to class imbalances, making it a more reliable metric in your scenario

