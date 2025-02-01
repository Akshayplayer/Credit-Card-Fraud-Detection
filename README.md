# Credit Card Fraud Detection 

## Problem statement 

The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.

In this project, we will analyse customer-level data which has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 

The dataset is taken from the [Kaggle Website](https://www.kaggle.com/mlg-ulb/creditcardfraud) website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

## Business Problem Overview

For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

In the banking industry, credit card fraud detection using machine learning is not just a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees, and denials of legitimate transactions.

## Understanding and Defining Fraud

Credit card fraud is any dishonest act and behavior to obtain information without the proper authorization from the account holder for financial gain. Among different ways of fraud, Skimming is the most common one, which is the way of duplicating information located on the magnetic strip of the card.  Apart from this, the other ways are:

- Manipulation/alteration of genuine cards
- Creation of counterfeit cards
- Stolen/lost credit cards
- Fraudulent telemarketing 

## Data Dictionary

The dataset can be download using this [link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.


## Project Pipeline

This pipeline outlines the steps involved in building a credit card fraud detection model using the provided dataset. Each step includes details about the techniques and considerations for that stage.

**1. Data Understanding:**

**Description:** Load the dataset and understand the features (columns) and their data types.

**Tools/Libraries:** Pandas for data loading and manipulation.

**Output:** Initial understanding of the dataset's features, data types, and potential challenges.

**2. Exploratory Data Analysis (EDA):**

**Description:** Perform univariate and bivariate analyses to uncover patterns and relationships within the data.

**Tools/Libraries:** Pandas, Matplotlib, Seaborn for data analysis and visualization.

**Specific Actions:**
- Visualize data distributions using histograms, box plots, etc.
- Analyze correlations between features using scatter plots, heatmaps, etc.
- Identify potential outliers and handle them appropriately.
- Check for class imbalance and consider techniques like under-sampling or over-sampling.

**Output:** Insights into data distributions, relationships, potential issues, and strategies for data preprocessing.

**3. Data Preprocessing:**

**Description:** Prepare the data for model training by addressing missing values, scaling/normalizing features, and encoding categorical variables.

**Tools/Libraries:** Scikit-learn for data preprocessing (StandardScaler, MinMaxScaler, LabelEncoder).

**Specific Actions:**
- Handle missing values using imputation or removal.
- Scale/normalize numerical features using StandardScaler or MinMaxScaler.
- Encode categorical features if present using LabelEncoder or one-hot encoding.

**Output:** Transformed dataset ready for model training.

**4. Feature Engineering:**

**Description:** Extract new or enhanced features by utilizing existing features to optimize the accuracy of the model.

**Specific Actions:** 
- Perform feature selection based on the calculated importance score of the features.
- Feature Importance techniques include calculating the impact of features on the model's performance using inbuilt feature importance calculation of the given models.
  
**Output:** Optimal Feature set on which the models can yield high accuracy.

**5. Train/Test Split:**

**Description:** Split the data into training and testing sets to evaluate model performance on unseen data.

**Tools/Libraries:** Scikit-learn for model selection (train_test_split).

**Specific Actions:**
- Split the data into 80% for training and 20% for testing.
- Use train_test_split with random_state for reproducibility.
- Consider stratified sampling to maintain class proportions in both sets.
- 
**Output:** Training and testing datasets ready for model training and evaluation.

**6. Model Building and Hyperparameter Tuning:**

**Description:** Machine Learning models, such as RandomForestClassifier or XGBClassifier.Run base models on the dataset. Train and evaluate different machine learning models, and optimize hyperparameters using grid search or cross-validation.

**Tools/Libraries:** Scikit-learn for model selection (GridSearchCV, cross_val_score), machine learning algorithms (RandomForestClassifier, XGBClassifier).

**Specific Actions:**
- Train models like RandomForestClassifier and XGBClassifier.
- Use GridSearchCV to search for optimal hyperparameter values.
- Perform cross-validation to evaluate model performance.
- Address class imbalance using techniques like class weights, oversampling, or undersampling.

**Output:** Trained models with optimized hyperparameters, along with their performance metrics.

**7. Model Evaluation:**

**Description:** Evaluate the performance of the final model on the testing data.

**Tools/Libraries:** Scikit-learn for evaluation metrics (accuracy_score, precision_score, recall_score, f1_score).

**Specific Actions:**
- Use metrics like accuracy, precision, recall, and F1-score.
- Confusion matrix and classification report can be used to check the performance of the model on the classes.
- Focus on recall to identify fraudulent transactions more effectively.
- 
**Output:** Final model with its performance metrics on the testing data, demonstrating its ability to detect fraud.
