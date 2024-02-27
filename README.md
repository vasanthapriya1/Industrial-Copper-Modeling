# Industrial-Copper-Modeling
The goal of this project is to develop a machine learning solution that predicts the selling price of copper based on historical transaction data and classifies the status of each transaction.

## Approach
- Data Understanding
- Identify variable types (continuous, categorical) and their distributions.
- Handle categorical variables with '00000' values and treat reference columns as categorical.
- Data preprocessing, including handling missing values and skewness.

## Exploratory Data Analysis (EDA):
- Explore the dataset to understand its structure, features, and distributions.
- Identify and visualize skewness and outliers in the dataset using appropriate techniques such as histograms, box plots, or scatter plots.

## Data Preprocessing:
- Handle skewness by applying transformations such as log transformation or box-cox transformation to make the data more normally distributed.
- Treat outliers using techniques such as winsorization or trimming to mitigate their impact on model performance.
- Perform data cleaning steps such as handling missing values, duplicate records, or irrelevant features.

## Regression Model for Selling_Price Prediction:
- Choose a suitable regression algorithm such as Linear Regression, Decision Tree Regression, or Random Forest Regression.
- Split the dataset into training and testing sets.
- Train the regression model on the training data and evaluate its performance on the testing data.
- Fine-tune the model hyperparameters if necessary to optimize performance.

## Classification Model for Lead Prediction:

- Create a binary classification model to predict lead outcomes (WON or LOST).
- Convert the STATUS variable into binary labels (1 for WON, 0 for LOST).
- Select an appropriate classification algorithm such as Logistic Regression, Random Forest Classifier, or Gradient Boosting Classifier.
- Split the dataset into training and testing sets.
- Train the classification model on the training data and evaluate its performance on the testing data.
- Fine-tune the model hyperparameters if necessary to optimize performance.

## Streamlit Web Application:
- Build a Streamlit web application where users can input values for each column.
- Incorporate the trained regression model to predict the Selling_Price based on the input values.
- Integrate the trained classification model to predict the lead status (WON or LOST) based on the input values.
- Display the predicted Selling_Price and lead status to the user.
  
## Key Technologies and Skills
- Python
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Pickle
- Streamlit

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.



