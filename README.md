# Fake News Detection using Machine Learning and NLP
## Project Overview
This project focuses on detecting fake news using Machine Learning and Natural Language Processing (NLP) techniques. The system classifies news articles as fake or real to reduce the impact of misinformation.

## Problem Statement
Fake news spreads rapidly through online platforms and social media. It can mislead people and create biased opinions. This project aims to
identify fake news automatically using ML algorithms.

## Dataset Description
This project uses a Fake News Dataset obtained from Kaggle. 
The dataset contains news articles with labels indicating whether the news is fake or real. 
It is suitable for text classification using NLP techniques.

- Source: Kaggle Fake News Dataset
- Columns:
  - title: News headline
  - text: Main news content
  - label: Class label (0 = Fake, 1 = Real)

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
  
- ## Libraries Used
- NumPy
- Pandas
- Scikit-learn
- NLTK

## Project Workflow
1. Data Collection from Kaggle dataset
2. Data Cleaning and Preprocessing
3. Text Vectorization using TF-IDF
4. Splitting dataset into training and testing sets
5. Training Machine Learning models
6. Testing models on unseen data
7. Evaluating performance using accuracy and other metrics
8. Selecting the best performing model

## Text Preprocessing
- Lowercasing
- Removing punctuation
- Stopword removal
- Tokenization
- Lemmatization

## Machine Learning Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

#Results:
Among all the models, Random Forest Classifier achieved the highest accuracy and provided better performance compared to other models.

## Installation & Execution

1. Clone the repository:
   git clone https://github.com/Apurwa599/fake-news-detection.git

2. Navigate to the project directory:
   cd fake-news-detection

3. Install required libraries:
   pip install -r requirements.txt

4. Run the notebook:
   jupyter notebook fake-news-detection.ipynb

## Future Improvements:
- Use Deep Learning models like LSTM or BERT
- Deploy the model using Flask or Streamlit
- Improve accuracy with hyperparameter tuning
- Add real-time news prediction

Conclusion:
This project demonstrates the use of Machine Learning and NLP techniques
to detect fake news effectively. It helps understand text classification
and real-world data challenges.

## Acknowledgements
This project was developed as part of an internship and for learning
purposes using open-source datasets and tools.


Author: Apurwa Khare  
        MCA (AIML)



