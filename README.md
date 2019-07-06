# Quora-Question-Pair-Similarity

## 1. Business Problem
## 1.1 Description

Quora is a place to gain and share knowledge—about anything. It’s a Platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.


> Credits: Kaggle
__ Problem Statement __

- Identify which questions asked on Quora are duplicates of questions that have already been asked.
- This could be useful to instantly provide answers to questions that have already been answered.
- We are tasked with predicting whether a pair of questions are duplicates or not.

## 1.2 Sources/Useful Links
- Source : https://www.kaggle.com/c/quora-question-pairs 

____ Useful Links ____
- Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
## 1.3 Real world/Business Objectives and Constraints
1. The cost of a mis-classification can be very high.
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
3. No strict latency concerns.
4. Interpretability is partially important.
## 2. Machine Learning Probelm
### 2.1 Data
### 2.1.1 Data Overview
- Data will be in a file Train.csv 
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate 
- Size of Train.csv - 60MB 
- Number of rows in Train.csv = 404,290

## 2.1.2 Example Data point

"id","qid1","qid2","question1","question2","is_duplicate"

"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"

"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"

"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"

"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"

## 2.2 Mapping the real world problem to an ML problem
### 2.2.1 Type of Machine Leaning Problem
It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

### 2.2.2 Performance Metric
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s):

- log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
- Binary Confusion Matrix
## 2.3 Train and Test Construction
We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.

## step by step procedure to solve this case study
- First load the train.csv data set into data frame
- We find number of data points : 404290
- We find there are 6 columns in data
- Those are [id , qid1, qid2, question1, question2. is_duplicate ]
- clearly observe that is_dupicate is labeled column
- Question pairs are not Similar (is_duplicate = 0): 63.08%
- Question pairs are Similar (is_duplicate = 1): 36.92%
- We done some EDA to understand data much more rigorous
- We find Total number of Unique Questions are: 537933
- Number of unique questions that appear more than one time : 111780 (20.77953945937505%)
- Max number of times a single question is repeated : 157
- Number of duplicate questions 0
- Maximum number of times a single question is repeated: 157
- Checking whether there are any rows with null values
- There are two rows with null values in question2
## Basic Feature Extraction (before cleaning)
- Hear we done some basic feature extraction
- freq_qid1 = Frequency of qid1's
- freq_qid2 = Frequency of qid2's
- q1len = Length of q1
- q2len = Length of q2
- q1_n_words = Number of words in Question 1
- q2_n_words = Number of words in Question 2
- word_Common = (Number of common unique words in Question 1 and Question 2)
- word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
- word_share = (word_common)/(word_Total)
- freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
- freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

## Analysis of some of the extracted features
- we find some questions have only one single words.

- Minimum length of the questions in question1 : 1

- Minimum length of the questions in question2 : 1

- Number of Questions with minimum length [question1] : 67

- Number of Questions with minimum length [question2] : 24

- When we perform univariat analysis on Feature : word_shapre

- The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity

- The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

- Feature: word_Common

- The distributions of the word_Common feature in similar and non-similar questions are highly overlapping

## EDA : Advanced Feature Extraction
- we done some advanced feature extraction on data
## Preprocessing of Text
- Removing html tags
- Removing Punctuations
- Performing stemming
- Removing Stopwords
- Expanding contractions etc
- Preprocessing of Text

## Definition:

- Token: You get a token by splitting sentence a space
- Stop_Word : stop words as per NLTK.
- Word : A token that is not a stop_word

## Features:

- cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2 
cwc_min = common_word_count / (min(len(q1_words), len(q2_words))


- cwc_max__ : Ratio of common_word_count to max lenghth of word count of Q1 and Q2 
cwc_max = common_word_count / (max(len(q1_words), len(q2_words)) 

- csc_min__ : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 
csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops)) 

- csc_max__ : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops)) 

- ctc_min__ : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) 

- ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))


- last_word_eq : Check if First word of both questions is equal or not
last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])


- first_word_eq : Check if First word of both questions is equal or not
first_word_eq = int(q1_tokens[0] == q2_tokens[0])


- abs_len_diff : Abs. length difference
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))


- mean_len : Average Token Length of both Questions
mean_len = (len(q1_tokens) + len(q2_tokens))/2


- fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- Token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- Token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


- longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
- longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

- append the new features as a columns to data set

## Plotting Word clouds:

- Creating Word Cloud of Duplicates and Non-Duplicates Question pairs
- We can observe the most frequent occuring words
## Visualization :

- We visualize the data using TSNE
- We find some labels are well separated and some labels are overlaped
## Vectorizer & train and test data
- we drop some columns [qid1, qid2]

- we merge three data frames by using id column

- take X = total data

- take Y = labels

- split the data into train and test data

- Applying tfidf vectorizer on train data

- Transform the vectorizer to X_train_q1 and X_train_q2

- we find most of columns/values are zeros

- so we convert train and test vectorized data into sparse matrix

- Then merge X_train_q1 and X_train_q2 using hstack function

- done both train and test data

- save those train and test data as .npz files

## Machine Learning Models
- our loss function is logloss
- In order to find best logloss first we build a random model to find log-loss
- We observe log_loss on test data using random mode : 0.887
- In order to reduce log loss we apply some machine learnign algorithams
## Logistic Regression
- When we apply logistic regression we done hyperparameter tuning
- we find best hyperparameter C / (1/lambda) is : 1
- log-loss its reduced is : 0.5146
- we are using l1 Reglarization because the features are very high Then We plot confusion matrix and Presision and Recall matrix
## Linear SVM
- When we apply linear SVM we done hyperparameter tuning
- We find best hyperparameter alpha is : 0.00001
- Log-loss its reduced is : 0.5638
- We are using l1 reglarization because the features are very high
- Then we plot confusion matrix and Rresision and recall matrixs
## XgBoost
- When we apply XgBoost we done hyperparameter tuning
- We find best hyperparameter n_estimators : 80 and max_depth : 100
- log-loss its reduced is : 0.5914
- Then we plot confusion matrix and Rresision and recall matrixs
