import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 35, 45, 50, 23, 40, 60, 48],
    'Income': [30000, 50000, 80000, 90000, 28000, 60000, 100000, 75000],
    'LoanAmount': [5000, 20000, 30000, 40000, 7000, 25000, 50000, 32000],
    'CreditHistory': [1, 1, 1, 0, 0, 1, 0, 1],
    'Creditworthy': [1, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
