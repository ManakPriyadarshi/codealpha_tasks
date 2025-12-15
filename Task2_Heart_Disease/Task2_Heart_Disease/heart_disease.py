import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Age': [45, 54, 65, 35, 50, 60, 40, 70],
    'BP': [130, 140, 150, 120, 135, 145, 125, 155],
    'Cholesterol': [230, 250, 260, 200, 240, 270, 210, 280],
    'HeartDisease': [1, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
