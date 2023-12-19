# Online service for automating the process of product matching 
Development of an ML-product and web interface for the automatic product matching service for the company [Prosept](https://prosept.ru/).

The customer manufactures several hundred different household and industrial chemical products, which are then sold through dealers. To assess the situation, manage prices, and overall business operations, the customer periodically gathers information on how dealers sell their products. The matching of dealer products with manufacturer products is currently done manually.

**Goal**: development of an online service accessible through a web browser for automating the process of product matching.

The project is cross-functional, and this repository contains the ML-solution, while the backend and frontend are presented in separate repositories: [backend](https://github.com/K1N88/product-markup-backend), [frontend](https://github.com/sergasent/hackaton-pros).

## Table of Contents
- [User experience](#user-experience)
- [Framework](#framework)
- [Quality Metrics](#quality-metrics)
- [Technical Solution](#technical-solution)
- [Использование](#использование)
- [Stack](#stack)
- [Команда проекта](#команда-проекта)

## User experience
This service is designed for the analytics - employees of the customer. For each selected dealer product several customer products that are most likely to match that dealer product are suggested. The selection of the most likely suggestions is made using machine learning methods.

## Framework 
As part of the project, an ML-product was developed to automate the process of matching goods.
1. Data Preprocessing and Preliminary Analysis
- Checking for duplicates
- Handling missing values
2. Text Cleaning
- Separating concatenated words and digits with spaces
- Removing punctuation.
- Converting text to lowercase
3. Text Vectorization:
  3.1. Using TF-IDF with various parameters, such as removing stop words, utilizing uni- and bi-grams based on words and characters.
  3.2. Generating embeddings using the LaBSE model.
4. Calculating Distance Between Text Vectors:
- Applying cosine similarity to assess the similarity between texts.
5. Investigating metric changes under different parameters, such as the number of most similar names outputted (5-10).
6. Reranking the Result:
- Attempting to rerank the results using a trained classifier.

## Quality Metrics
- Accuracy@5 - Accuracy indicating how often the correct name is among the first five.
- Accuracy@1 - Accuracy indicating how often the correct name is proposed first.
- MRR (Mean Reciprocal Rank) - Measure of how often the correct option is proposed closer to the beginning of the list (average position of the correct item in the ranked list).

## Technical Solution
This project provides functionality for predicting the top n most similar customer product names for each specific dealer name. The prediction is based on computing the cosine similarity of the vectorized product names. To convert textual data into numerical values (vectorization), TF-IDF Vectorizer or pretrained LaBSE model were used.
### Files
- `Procept_tfidf.py`, `Procept_labse_small.py`: files containing the main function for predicting n most similar customer product names   
- `requirements.txt`: file with a list of dependencies for installing the necessary libraries  
### Data preprocessing
To standardize the names, the text cleaning function was used inside the main functions:
- to separate concatenated words and digits with spaces
- to remove punctuation
- to convert text to lowercase
- to remove the words "просепт/prosept"

## Usage
### To install the necessary libraries, run:
```sh
$ pip install -r requirements.txt
``
### Two variants of models:  
1. Procept_tfidf.py based on TF-IDF Vectorizer.
2. Procept_labse_small.py based on LaBSE en-ru model.

### Result of comparison of two models

|Model|Accuracy@1|Accuracy@5|MRR|Executing Time|  
| --- | :---: | :---: | --- | :---: |
|TF-IDF|73.97%|93.21%|0.8164|7сек|
|LaBSE|75.08%| 94.70%|0.839|120сек|

The TF-IDF model was chosen due to its faster performance.

## Stack
- `Pandas`, `Matplotlib`, `Seaborn`, `NumPy`, `Scikit-learn`, `PyTorch`
-  *TF-IDF, LaBSE*
-  *cosine similarity*

## Team
**- Грибанов Михаил - Leader** [Telegram](https://t.me/gribanov_m) 
**- Сергеев Дмитрий**   [Telegram](https://t.me/SDI84) 
**- Шубина Любовь** [Telegram](https://t.me/luybashu)
