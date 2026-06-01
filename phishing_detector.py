import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("phishing_emails.csv")

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

email = input("\nEnter email text: ")

email_features = vectorizer.transform([email])
prediction = model.predict(email_features)

print("\nPrediction:", prediction[0].upper())