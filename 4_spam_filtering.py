# Practical 4: E-mail Spam Filtering using TF-IDF + Naive Bayes

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset (Kaggle SMS spam file is latin-1, not utf-8)
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]].dropna()

# Step 2: Split into training (75%) and testing (25%) data
X_train, X_test, y_train, y_test = train_test_split(
    df.message, df.label, test_size=0.25, random_state=1)

# Step 3: Convert text to TF-IDF numbers
vec = TfidfVectorizer(stop_words="english")
X_train_tfidf = vec.fit_transform(X_train)        # learn vocabulary from training data only

# Step 4: Train the Naive Bayes classifier
model = MultinomialNB().fit(X_train_tfidf, y_train)

# Step 5: Evaluate on the test data
pred = model.predict(vec.transform(X_test))
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Step 6: Try new messages
messages = [
    "Congratulations you win free cash prize click now",
    "team meeting tomorrow please review the report",
]
for m in messages:
    print(f"'{m}' ->", model.predict(vec.transform([m]))[0])
