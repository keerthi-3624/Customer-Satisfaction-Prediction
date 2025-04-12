# Customer Satisfaction Prediction Project

## Project Overview

This project focuses on predicting **Customer Satisfaction Ratings** based on customer support ticket data using machine learning techniques. The goal is to classify customer satisfaction ratings into categories (e.g., 1 to 5) based on features such as ticket type, priority, resolution time, and other metadata.

The project is implemented in Python using libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

---

## Features of the Dataset

The dataset used in this project is named **`customer_support_tickets.csv`** and contains 8,469 entries with 17 columns. Below are the key columns:

### Key Columns:
1. **Ticket ID**: Unique identifier for each ticket.
2. **Customer Name & Email**: Identifying information about the customer.
3. **Customer Age & Gender**: Demographic details of the customer.
4. **Product Purchased**: The product associated with the ticket.
5. **Date of Purchase**: When the purchase was made.
6. **Ticket Type, Subject, Description**: Metadata about the customer support ticket.
7. **Ticket Status**: Current status of the ticket (e.g., Open, Closed).
8. **Resolution**: Details about how the ticket was resolved.
9. **Ticket Priority**: Priority level of the ticket (e.g., Low, Medium, High).
10. **Ticket Channel**: The communication channel used (e.g., Email, Phone).
11. **First Response Time & Time to Resolution**: Time metrics related to ticket handling.
12. **Customer Satisfaction Rating**: Target variable (ratings from 1 to 5).

---

## Libraries Used

The following Python libraries are utilized in this project:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib & seaborn**: For data visualization.
- **scikit-learn**:
  - `LabelEncoder`: Encodes categorical variables into numerical values.
  - `StandardScaler`: Scales features for better performance in modeling.
  - `RandomForestClassifier`: Used for classification modeling.
  - `train_test_split`: Splits data into training and testing sets.
  - `accuracy_score`, `classification_report`, and `confusion_matrix`: For model evaluation.

---

## Workflow

### 1. Data Loading and Exploration
- Load the dataset using `pandas.read_csv`.
- Display an overview of the dataset (`df.info()` and `df.describe()`).
- Identify missing values and handle them appropriately.

### 2. Preprocessing
- Drop unnecessary columns (e.g., Ticket ID, Customer Name, Email) to simplify analysis.
- Convert time-related columns (`First Response Time` and `Time to Resolution`) into numerical values by calculating total seconds since the earliest time.
- Fill missing values in numerical columns with their median values.
- Encode categorical columns using `LabelEncoder`.

### 3. Feature Selection
Define features (`X`) and target variable (`y`), ensuring that missing values in the target variable are handled by replacing them with the median.

### 4. Train-Test Split
Split the dataset into training and testing sets using an 80/20 ratio.

### 5. Feature Scaling
Standardize features using `StandardScaler` to ensure consistent scaling across all numerical variables.

### 6. Model Building
Train a **Random Forest Classifier** on the training data using default hyperparameters.

### 7. Evaluation
Evaluate model performance on test data using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

---

## Results

### Model Performance:
The Random Forest Classifier achieves an accuracy of approximately **72%**, indicating moderate predictive performance.

### Classification Report:
The report shows precision, recall, and F1-score for each satisfaction rating category (1 to 5). Most predictions are concentrated around rating category "3," suggesting room for improvement in classifying other categories.

### Confusion Matrix:
The confusion matrix highlights misclassifications between satisfaction ratings, providing insights into areas where model performance can be improved.

---

## How to Run the Code

1. Clone this repository to your local machine.
2. Ensure Python is installed along with required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
3. Place the dataset (`customer_support_tickets.csv`) in your working directory.
4. Open the Jupyter Notebook file (`Customer_satisfaction.ipynb`) in Jupyter Notebook or any compatible IDE.
5. Run all cells sequentially to execute preprocessing, model training, and evaluation.

---

## Example Usage

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")

# Preprocess data (refer to notebook for detailed steps)
# Define features and target variable
X = df_clean.drop(columns=['Customer Satisfaction Rating'])
y = df_clean['Customer Satisfaction Rating']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

---

## Future Improvements

1. Experiment with hyperparameter tuning for Random Forest Classifier to improve accuracy.
2. Explore other machine learning algorithms like Gradient Boosting or Neural Networks for better performance.
3. Handle class imbalance through techniques like oversampling or SMOTE (Synthetic Minority Oversampling Technique).
4. Incorporate additional features such as sentiment analysis from ticket descriptions.

---

Feel free to contribute by submitting issues or pull requests!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/57218076/19fc345a-a722-41dd-bdb6-f7ab9b5ec709/Customer_satisfaction.ipynb

---
Answer from Perplexity: pplx.ai/share
