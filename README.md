# Predicting Hepatitis C Disease Progression
**Final Project: Machine Learning Course** **Author:** Scott Gerritsen

## Project Overview
For this project, I chose to work with the **HCV (Hepatitis C Virus) Dataset** from the UCI Machine Learning Repository. My goal was to see if clinical laboratory results—things like liver enzyme levels and protein counts—could accurately categorize patients into different stages of liver disease (Hepatitis, Fibrosis, and Cirrhosis) versus healthy blood donors.

### The Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HCV+data)
* **Size:** 615 instances, 14 attributes.
* **Target:** `Category` (Ordinal scale from healthy donor to cirrhosis).

## What I Found in the Data (EDA)
Before building any models, I ran some exploratory analysis. A few things jumped out immediately:

* **The Imbalance Issue:** The data is heavily skewed. Most of the entries are healthy blood donors. This is a classic "needle in a haystack" problem for machine learning.
* **Key Indicators:** Looking at the correlation heatmap, liver-related features like **AST** (Aspartate Aminotransferase) and **ALT** (Alanine Aminotransferase) showed the strongest connections to disease progression, which makes sense from a medical perspective.
* **Missing Info:** There were some gaps in the laboratory data (specifically in the ALP and CHOL columns) that I had to address before modeling.

## Preprocessing Workflow
To get the data "model-ready," I followed these steps:

1. **Cleaning:** Dropped the `Unnamed: 0` column as it was just a row index.
2. **Imputation:** Rather than deleting rows with missing values, I filled the NaNs using the **mean** of each column to keep as much data as possible.
3. **Encoding:** I used `LabelEncoder` to turn categorical data like **Sex** and the **Category** target into numbers that the algorithms could process.
4. **Train/Test Split:** I split the data (80/20) to ensure I had a fresh set of data to test the models on.
5. **Scaling:** I applied `StandardScaler` to the numerical features. I made sure to fit the scaler *only* on the training data to prevent any data leakage.

## Modeling and Performance
I decided to compare a straightforward linear model against a more complex ensemble method.

### 1. Logistic Regression (The Baseline)
* **Overall Accuracy:** 89%
* **Performance:** It was surprisingly effective. It caught almost every healthy donor, though it did struggle to differentiate between the specific stages of Hepatitis and Cirrhosis (F1-scores for those were lower, around 0.50–0.80).

### 2. Random Forest (The Ensemble)
* **Overall Accuracy:** 86%
* **Performance:** Interestingly, this model performed slightly worse than the baseline. It had a hard time with the minority classes—in fact, it completely missed "Class 1" in my test set.

## Reflections and Lessons Learned
* **Accuracy is Deceiving:** This project was a great reminder that a high accuracy score doesn't mean a model is "good." Because of the class imbalance, the models could get 89% accuracy just by guessing "Healthy" every time. 
* **Complexity isn't always better:** I expected the Random Forest to beat Logistic Regression, but the simpler model actually generalized better here.
* **Next Steps:** If I had more time, I would definitely look into **SMOTE** (oversampling the minority classes) or adjusting **class weights** to force the model to pay more attention to the actual Hepatitis/Cirrhosis cases.

## Repository Structure
* `Scott_Gerritsen_Machine_Learning_Final_Project.ipynb`: The original development notebook.
* `src/hcv_prediction.py`: Cleaned, documented Python script of the workflow.
* `data/`: Local directory for the dataset.
