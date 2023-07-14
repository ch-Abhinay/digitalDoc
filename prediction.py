#importing required modules 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
train = pd.read_csv(r"C:\Users\nayak\Documents\Training1.csv")
test = pd.read_csv(r"C:\Users\nayak\Documents\Testing.csv")

#cleaning the unrequired data
train = train.drop(["Unnamed: 133"], axis=1)
train.prognosis.value_counts()
train.isna().sum()
test.isna().sum()

P = train[["prognosis"]]
X = train.drop(["prognosis"], axis=1)
Y = test.drop(["prognosis"], axis=1)
syms=list(train.head(0))
syms.pop()
#print(syms)
#splitting data 
xtrain, xtest, ytrain, ytest = train_test_split(X, P, test_size=0.2, random_state=42)

#using random Forest classifers (alg)
rf = RandomForestClassifier(random_state=42)
model_rf = rf.fit(xtrain, ytrain)
tr_pred_rf = model_rf.predict(xtrain)
ts_pred_rf = model_rf.predict(xtest)


#testing the accuracy 
print("training accuracy is:", accuracy_score(ytrain, tr_pred_rf))
print("testing accuracy is:", accuracy_score(ytest, ts_pred_rf))

# Prompt the user to enter symptoms
user_symptoms = input("Enter your symptoms (separated by commas): ").split(",")
for i in user_symptoms:
    if i not in syms:
        print('symptoms not in list, result may be inappropriate')
# Preprocess user input
user_input = np.zeros(len(X.columns))
for symptom in user_symptoms:
    symptom = symptom.strip().lower()
    if symptom in X.columns:
        user_input[X.columns.get_loc(symptom)] = 1

# Use the trained model to predict disease
predicted_disease = model_rf.predict([user_input])[0]

print("Predicted disease:", predicted_disease)