# IMPORTING DEPENDENCIES

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
# Data Collection and Analysis
# PIMA Diabetes Dataset


#see the version of the all lib
print("numpy",np.__version__)        #numpy 2.2.4
print("pandas",pd.__version__)       #pandas 2.2.3
print("sklearn",sklearn.__version__) #sklearn 1.6.1

# Load the dataset (update the path as necessary)
diabetes_dataset = pd.read_csv('diabetes.csv')

# Print first 5 rows
print(diabetes_dataset.head())

# Dataset information
print("Shape:", diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())

# 0 --> Non-Diabetic
# 1 --> Diabetic

# Separate features and target
# Remove 'SkinThickness' and 'DiabetesPedigreeFunction' from the dataset
X = diabetes_dataset.drop(columns=['Outcome', 'SkinThickness', 'DiabetesPedigreeFunction'], axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Save the scaler
with open('scaler.sav', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
# Train SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
# Accuracy scores
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

# Save the trained model
filename = 'trained_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
print("Model saved as 'trained_model.pkl'")
